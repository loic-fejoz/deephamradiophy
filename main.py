import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars
import os
import math
import argparse
import torch.nn.functional as F

try:
    from torchview import draw_graph
    import graphviz
    HAS_TORCHVIEW = True
except ImportError:
    HAS_TORCHVIEW = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rrc_pulse(n_samples, rolloff, samples_per_symbol, filter_span):
    """
    Generates a Root-Raised Cosine (RRC) impulse response.
    """
    T = samples_per_symbol
    t = torch.arange(-filter_span * T / 2, filter_span * T / 2 + 1) / T
    
    pulse = torch.zeros_like(t)
    
    # Handle the t=0 case
    pulse[t == 0] = 1.0 - rolloff + 4 * rolloff / np.pi
    
    # Handle the t = +/- T / (4 * alpha) cases
    if rolloff > 0:
        special_t = 1.0 / (4 * rolloff)
        pulse[torch.abs(t - special_t) < 1e-6] = (rolloff / np.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
        )
        pulse[torch.abs(t + special_t) < 1e-6] = pulse[torch.abs(t - special_t) < 1e-6]
    
    # General case
    mask = (t != 0)
    if rolloff > 0:
        mask &= (torch.abs(torch.abs(t) - 1.0 / (4 * rolloff)) > 1e-6)
    
    tm = t[mask]
    pulse[mask] = (
        torch.sin(np.pi * tm * (1 - rolloff)) + 4 * rolloff * tm * torch.cos(np.pi * tm * (1 + rolloff))
    ) / (np.pi * tm * (1 - (4 * rolloff * tm)**2))
    
    # Normalize energy to 1
    pulse = pulse / torch.sqrt(torch.sum(pulse**2))
    return pulse

# --- 1. Secondary Modules for Advanced Sync (Phase 11) ---

class STN1D(nn.Module):
    def __init__(self, input_channels, L, target_N):
        super(STN1D, self).__init__()
        self.target_N = target_N
        self.L = L
        
        # Localization network to estimate timing offset
        # Larger kernels help 'see' the signal even when misaligned
        self.localization = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, padding=7),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        
        # Determine localization output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, L)
            loc_out = self.localization(dummy_input)
            n_flatten = loc_out.size(1)
            
        self.fc_loc = nn.Sequential(
            nn.Linear(n_flatten, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize the weights/bias to output 0 (centered)
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.zero_()

    def forward(self, x):
        # x: (Batch, 2, L)
        batch_size = x.size(0)
        
        # Predict normalized offset theta in [-1, 1]
        features = self.localization(x)
        theta = torch.tanh(self.fc_loc(features)) # (Batch, 1)
        
        # Create 1D sampling grid
        # We want to crop a window of target_N from the search window L
        scale = self.target_N / self.L
        
        # Base grid for target_N samples in [-1, 1]
        grid = torch.linspace(-1, 1, self.target_N, device=x.device).view(1, 1, self.target_N, 1).expand(batch_size, -1, -1, -1)
        
        # Map back to L-window coordinates: grid_mapped = grid * scale + theta_scaled
        # Since grid_sample expects input in [-1, 1], we effectively zoom and shift
        # theta_scaled range restricted to [-(1-scale), (1-scale)] to keep window inside [-1, 1]
        theta_scaled = theta * (1.0 - scale)
        grid = grid * scale + theta_scaled.view(-1, 1, 1, 1)
        
        # Construct the 4D sampling grid (Batch, H_out, W_out, 2)
        # Using concat instead of in-place assignment to help torchview visualization
        zero_grid = torch.zeros_like(grid)
        sampling_grid = torch.cat([grid, zero_grid], dim=-1) # (Batch, 1, target_N, 2)
        
        # x_4d: (Batch, Channels, H=1, W=L)
        x_4d = x.unsqueeze(2)
        aligned = F.grid_sample(x_4d, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return aligned.squeeze(2) # (Batch, 2, target_N)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (Batch, L, hidden_dim)
        # Compute soft attention weights over the sequence
        # attn_weights: (Batch, L, 1)
        attn_weights = F.softmax(self.query(x), dim=1)
        
        # Context vector via weighted sum
        context = torch.sum(x * attn_weights, dim=1)
        return context

# --- 2. The "End-to-End" Architecture ---

# Encoder (Transmitter)
class Encoder(nn.Module):
    def __init__(self, M, N_SAMPLES, args):
        super(Encoder, self).__init__()
        self.M = M
        self.N_SAMPLES = N_SAMPLES 
        self.args = args
        
        # Architecture Parameters
        self.latent_dim = 16
        self.conv_dim = 32
        
        # Use nn.Embedding to map message index to a latent vector
        self.embedding = nn.Embedding(M, self.latent_dim)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(self.latent_dim, self.conv_dim, kernel_size=N_SAMPLES),
            nn.ReLU(),
            nn.Conv1d(self.conv_dim, 2, kernel_size=3, padding=1),
        )

    def forward(self, messages_indices):
        x = self.embedding(messages_indices) # (batch, latent_dim)
        x = x.unsqueeze(-1) # (batch, latent_dim, 1)
        
        waveform_iq = self.upsample(x) # (batch, 2, N_SAMPLES)
        
        # Phase 12: RRC Pulse Shaping
        if self.args.rolloff > 0:
            T_s = 4 
            pulse = get_rrc_pulse(self.N_SAMPLES, self.args.rolloff, T_s, self.args.filter_span).to(waveform_iq.device)
            pulse = pulse.view(1, 1, -1)
            
            i = F.conv1d(waveform_iq[:, 0:1, :], pulse, padding='same')
            q = F.conv1d(waveform_iq[:, 1:2, :], pulse, padding='same')
            waveform_iq = torch.cat([i, q], dim=1)

        # --- Power Constraint ---
        energy = torch.mean(waveform_iq**2, dim=(1, 2), keepdim=True)
        norm_factor = torch.sqrt(energy)
        waveform_iq = waveform_iq / (norm_factor + 1e-8)

        return waveform_iq

# Decoder (Receiver)
class Decoder(nn.Module):
    def __init__(self, M, N_SAMPLES, args):
        super(Decoder, self).__init__()
        self.M = M
        self.N_SAMPLES = N_SAMPLES
        self.args = args
        
        # Architecture Parameters
        self.cnn_dim = 64
        self.gru_dim = 2 * 64
        
        # Phase 12: Matched Filter Front-end
        if self.args.rolloff > 0:
            T_s = 4
            pulse = get_rrc_pulse(self.N_SAMPLES, self.args.rolloff, T_s, self.args.filter_span)
            self.matched_filter = nn.Parameter(pulse.view(1, 1, -1), requires_grad=False)
        else:
            self.matched_filter = None

        # Phase 11: STN Alignment
        self.stn = STN1D(2, N_SAMPLES + args.max_offset, N_SAMPLES)

        self.cnn = nn.Sequential(
            nn.Conv1d(2, self.cnn_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(self.cnn_dim, self.cnn_dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        # GRU for sequence modeling
        # input_size matches the output channels of the last CNN layer
        self.gru = nn.GRU(input_size=self.cnn_dim, hidden_size=self.gru_dim, batch_first=True, bidirectional=True)
        # Attention for selective pooling (multiplier 2 for bidirectional GRU)
        self.pool = AttentionPooling(self.gru_dim * 2)
        self.fc = nn.Linear(self.gru_dim * 2, M)

    def forward(self, x_noisy):
        # x_noisy: (Batch, 2, L)
        
        # Phase 12: Matched Filtering
        if self.matched_filter is not None:
            # Move filter to device if needed (parameters do this automatically normally)
            i = F.conv1d(x_noisy[:, 0:1, :], self.matched_filter, padding='same')
            q = F.conv1d(x_noisy[:, 1:2, :], self.matched_filter, padding='same')
            x_noisy = torch.cat([i, q], dim=1)

        # Phase 11: Coarse Alignment (STN)
        x_aligned = self.stn(x_noisy)
        
        # Feature extraction
        x = self.cnn(x_aligned)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        features = self.pool(out)
        logits = self.fc(features)
        return logits

# --- The Channel Layer ---

def apply_phase_noise(waveform_iq, max_phase_deg=180.0):
    """Applies a random constant phase rotation to each waveform in the batch."""
    batch_size = waveform_iq.size(0)
    # Random angle between -max_phase and +max_phase
    phase = (torch.rand(batch_size, 1, 1, device=waveform_iq.device) * 2 - 1) * (np.pi * max_phase_deg / 180.0)
    
    # Rotation matrix for I and Q
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    
    i = waveform_iq[:, 0:1, :]
    q = waveform_iq[:, 1:2, :]
    
    i_rot = i * cos_p - q * sin_p
    q_rot = i * sin_p + q * cos_p
    
    return torch.cat([i_rot, q_rot], dim=1)

def apply_frequency_offset(waveform_iq, max_freq_step=0.1):
    """Applies a random linear phase ramp (frequency offset)."""
    batch_size, _, n_samples = waveform_iq.size()
    # Random frequency step per batch
    freq = (torch.rand(batch_size, 1, 1, device=waveform_iq.device) * 2 - 1) * max_freq_step
    
    # Time vector [0, 1, ..., N-1]
    t = torch.arange(n_samples, device=waveform_iq.device).view(1, 1, -1)
    phase_ramp = freq * t
    
    cos_p = torch.cos(phase_ramp)
    sin_p = torch.sin(phase_ramp)
    
    i = waveform_iq[:, 0:1, :]
    q = waveform_iq[:, 1:2, :]
    
    i_rot = i * cos_p - q * sin_p
    q_rot = i * sin_p + q * cos_p
    
    return torch.cat([i_rot, q_rot], dim=1)

def add_awgn(waveform_iq, snr_db):
    """Adds Additive White Gaussian Noise to the I/Q samples (Batch, 2, N)."""
    snr_linear = 10**(snr_db / 10.0)
    noise_power = 1.0 / snr_linear
    noise_std = np.sqrt(noise_power / 2) 
    
    noise = torch.randn_like(waveform_iq) * noise_std
    return waveform_iq + noise

def indices_to_bits(indices, K):
    """
    Converts message indices (0 to 2^K-1) to bit tensors.
    """
    batch_size = indices.size(0)
    bits = []
    for i in range(K):
        # Extract i-th bit
        # 2^i bit is (indices >> i) & 1
        bits.append((indices >> i) & 1)
    
    # Returns (batch, K)
    return torch.stack(bits, dim=1).float()

def calculate_bitwise_ber(predicted_logits, true_indices, K):
    """
    Calculates actual Bit Error Rate by comparing binary representations.
    """
    _, predicted_indices = torch.max(predicted_logits, 1)
    
    pred_bits = indices_to_bits(predicted_indices, K)
    true_bits = indices_to_bits(true_indices, K)
    
    # Hamming distance
    bit_errors = (pred_bits != true_bits).sum().item()
    total_bits = true_indices.size(0) * K
    return bit_errors / total_bits, bit_errors

def get_spectral_penalty(waveform_iq, bw_limit=0.5):
    """
    Computes energy outside the allowed bandwidth limit.
    bw_limit: Fraction of the total bandwidth allowed (0.0 to 1.0).
    """
    batch_size, _, n_samples = waveform_iq.size()
    if n_samples <= 1:
        return torch.tensor(0.0, device=waveform_iq.device)

    # Convert to complex representation: I + jQ
    z = torch.complex(waveform_iq[:, 0, :], waveform_iq[:, 1, :])
    
    # Compute FFT
    Z = torch.fft.fft(z, dim=1)
    Z_shifted = torch.fft.fftshift(Z, dim=1)
    
    # Power Spectral Density
    psd = torch.abs(Z_shifted)**2
    
    # Create mask of allowed bins (central ones)
    total_bins = n_samples
    allowed_bins = max(1, int(total_bins * bw_limit))
    start_bin = (total_bins - allowed_bins) // 2
    end_bin = start_bin + allowed_bins
    
    mask = torch.ones(total_bins, device=waveform_iq.device)
    mask[start_bin:end_bin] = 0.0 # Bins inside the limit have 0 penalty
    
    # Energy outside the mask
    # Normalize by total energy (N^2 in FFT domain for unit energy in time domain)
    # We want the penalty to be the fraction of power outside the mask (0 to 1) 
    total_energy = torch.sum(psd)
    out_energy = torch.sum(psd * mask)
    penalty = out_energy / (total_energy + 1e-8)
    return penalty

def get_dc_penalty(waveform_iq):
    """
    Penalizes the DC component (mean of the signal).
    DC power = |mean(I)|^2 + |mean(Q)|^2
    """
    # Mean over time dimension
    dc_i = torch.mean(waveform_iq[:, 0, :], dim=1)
    dc_q = torch.mean(waveform_iq[:, 1, :], dim=1)
    
    # Square magnitude (power) of DC component
    dc_power = dc_i**2 + dc_q**2
    
    # Return average across batch
    return torch.mean(dc_power)

def get_papr(waveform_iq):
    """
    Computes the Peak-to-Average Power Ratio (PAPR) in dB.
    PAPR = 10 * log10(max(|x|^2) / mean(|x|^2))
    """
    # Instantaneous power: I^2 + Q^2
    power = torch.sum(waveform_iq**2, dim=1) # (Batch, N_SAMPLES)
    
    peak_power, _ = torch.max(power, dim=1)
    mean_power = torch.mean(power, dim=1)
    
    papr = 10 * torch.log10(peak_power / (mean_power + 1e-8) + 1e-8)
    return torch.mean(papr) # Return average PAPR across batch

def get_papr_penalty(waveform_iq):
    """
    Computes a penalty proportional to the variance of the instantaneous power.
    A constant envelope signal has 0 variance in power.
    """
    power = torch.sum(waveform_iq**2, dim=1) # (Batch, N_SAMPLES)
    
    # Since we normalize in the Encoder to mean_power = 2.0 (2 channels, mean=1.0)
    # we want power to be 2.0 everywhere for constant envelope.
    papr_penalty = torch.mean((power - 2.0)**2)
    
    return papr_penalty

def apply_multipath_fading(waveform_iq, n_taps=3, fading_scale=0.0, use_circular=True):
    """
    Simulates Rayleigh Fading using a Tapped Delay Line model.
    n_taps: Number of paths.
    fading_scale: Strength of the fading (0.0 = no fading, 1.0 = full fading).
    use_circular: If True, uses circular convolution (assumes cyclic prefix/OFDM).
                 If False, uses linear convolution (standard temporal fading).
    """
    if fading_scale <= 0 or n_taps <= 1:
        return waveform_iq
    
    batch_size, channels, n_samples = waveform_iq.size()
    device = waveform_iq.device
    
    coeffs_real = torch.randn(batch_size, n_taps, device=device) / math.sqrt(2 * n_taps)
    coeffs_imag = torch.randn(batch_size, n_taps, device=device) / math.sqrt(2 * n_taps)
    
    output = waveform_iq * (1.0 - fading_scale)
    faded = torch.zeros_like(waveform_iq)
    
    if use_circular:
        for t in range(n_taps):
            shifted_i = torch.roll(waveform_iq[:, 0, :], shifts=t, dims=1)
            shifted_q = torch.roll(waveform_iq[:, 1, :], shifts=t, dims=1)
            a = coeffs_real[:, t].view(-1, 1)
            b = coeffs_imag[:, t].view(-1, 1)
            faded[:, 0, :] += (shifted_i * a - shifted_q * b)
            faded[:, 1, :] += (shifted_i * b + shifted_q * a)
    else:
        # Linear Convolution using F.conv1d
        # We process one batch item at a time or use group convolution
        # To keep it simple and efficient, we can use group convolution where groups = batch_size
        # Each channel/batch gets its own unique kernels (taps)
        
        # waveform_iq is (Batch, 2, N)
        # We need kernels of size (Batch*2, 1, n_taps)
        # but the kernels are complex.
        # This is slightly complex to vectorize perfectly in one call, 
        # let's use the loop but optimize the I/Q complex logic.
        for t in range(n_taps):
            # Zero-padding for linear delay
            a = coeffs_real[:, t].view(-1, 1)
            b = coeffs_imag[:, t].view(-1, 1)
            
            # Shifted version with zero padding at the start
            shifted_i = F.pad(waveform_iq[:, 0, :-t], (t, 0)) if t > 0 else waveform_iq[:, 0, :]
            shifted_q = F.pad(waveform_iq[:, 1, :-t], (t, 0)) if t > 0 else waveform_iq[:, 1, :]
            
            faded[:, 0, :] += (shifted_i * a - shifted_q * b)
            faded[:, 1, :] += (shifted_i * b + shifted_q * a)
            
    output += fading_scale * faded
    
    energy = torch.mean(output**2, dim=(1, 2), keepdim=True)
    output = output / (torch.sqrt(energy) + 1e-8)
    
    return output

def apply_timing_offset(waveform_iq, max_offset=0):
    """
    Randomly shifts the signal within a larger window.
    Pads the signal to length N + max_offset and shifts it by [0, max_offset].
    """
    if max_offset <= 0:
        return waveform_iq
    
    batch_size, channels, n_samples = waveform_iq.size()
    device = waveform_iq.device
    
    # We want to return a window of size N + max_offset
    window_size = n_samples + max_offset
    output = torch.zeros(batch_size, channels, window_size, device=device)
    
    for b in range(batch_size):
        offset = torch.randint(0, max_offset + 1, (1,)).item()
        output[b, :, offset:offset + n_samples] = waveform_iq[b]
        
    return output

# --- Post-Training Utilities ---

def evaluate_at_snr(encoder, decoder, snr_db, args, device, num_batches=100):
    """Evaluates BER for a specific SNR."""
    M = 2**args.K
    total_bit_errors = 0
    total_bits_evaluated = 0
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for _ in range(num_batches):
            messages_indices = torch.randint(0, M, (args.batch_size,)).to(device)
            
            encoded_signal = encoder(messages_indices)
            
            # Channel impairments (SNR, Fading, Offsets) matches curriculum
            distorted_signal = apply_multipath_fading(encoded_signal, n_taps=args.n_taps, 
                                                     fading_scale=args.fading_scale, 
                                                     use_circular=args.use_circular)
            distorted_signal = apply_phase_noise(distorted_signal, max_phase_deg=args.max_phase_deg)
            distorted_signal = apply_frequency_offset(distorted_signal, max_freq_step=args.max_freq_step)
            distorted_signal = apply_timing_offset(distorted_signal, max_offset=args.max_offset)
            noisy_signal = add_awgn(distorted_signal, snr_db)
            decoded_logits = decoder(noisy_signal)
            ber, bit_errors = calculate_bitwise_ber(decoded_logits, messages_indices, args.K)
            total_bit_errors += bit_errors
            total_bits_evaluated += args.batch_size * args.K
            
    return total_bit_errors / total_bits_evaluated

def generate_ber_table(encoder, decoder, args, device):
    """Generates and prints a Markdown table of BER vs SNR and saves a plot."""
    print(f"\n| SNR (dB) | BER under fading scale {args.fading_scale} |")
    print("|----------|-------------------------------|")
    
    start = int(max(args.snr_start, args.snr_end))
    end = int(min(args.snr_start, args.snr_end))
    
    snrs = []
    bers = []
    
    for snr in range(start, end - 1, -1):
        ber = evaluate_at_snr(encoder, decoder, float(snr), args, device)
        print(f"| {snr:8d} | {ber:.6f} |")
        snrs.append(snr)
        bers.append(ber)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogy(snrs, bers, 'b-o', markersize=4, label=f'Fading Scale: {args.fading_scale}')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title(f'BER vs SNR Waterfall (K={args.K}, N={args.n_samples})')
    plt.legend()
    
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    plot_filename = f"output/{prefix_str}ber_db_vs_snr.png"
    plt.savefig(plot_filename)
    print(f"\nBER waterfall plot saved to {plot_filename}")
    
    if not args.no_gui:
        plt.show()
    else:
        plt.close()

def export_to_sigmf(encoder, args):
    """Exports learned waveforms to SigMF format."""
    import json
    import time
    
    M = 2**args.K
    encoder.eval()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        all_indices = torch.arange(M).long().to(device)
        waveforms = encoder(all_indices).cpu().numpy() # (M, 2, N_SAMPLES)
        
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    data_filename = f"output/{prefix_str}learned_waveforms.sigmf-data"
    meta_filename = f"output/{prefix_str}learned_waveforms.sigmf-meta"
    
    # Flatten and interleave I/Q: (M, 2, N) -> (M, N, 2) -> (M*N*2,)
    iq_interleaved = waveforms.transpose(0, 2, 1).flatten().astype(np.float32)
    
    with open(data_filename, 'wb') as f:
        f.write(iq_interleaved.tobytes())
        
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": 1000000,
            "core:version": "1.0.0",
            "core:description": f"Learned waveforms for K={args.K}, N={args.n_samples} PHY Autoencoder",
            "core:recorder": "Deep-HAM Radio PHY Autoencoder",
            "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "captures": [{ "core:sample_start": 0, "core:frequency": 433000000 }],
        "annotations": []
    }
    
    for i in range(M):
        meta["annotations"].append({
            "core:sample_start": i * args.n_samples,
            "core:sample_count": args.n_samples,
            "core:label": f"Message {i}"
        })
        
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"SigMF capture saved to {data_filename} and {meta_filename}")

# --- Training Loop ---
def train_autoencoder(args):
    M = 2**args.K
    encoder = Encoder(M, args.n_samples, args).to(DEVICE)
    decoder = Decoder(M, args.n_samples, args).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

    # Generate all possible messages for evaluation and plotting
    all_messages_indices = torch.arange(M).long().to(DEVICE)

    train_losses = []
    ber_history = []
    snr_history = []

    for epoch in range(args.epochs):
        # --- Noise Curriculum with Jitter ---
        # Target curriculum SNR
        target_snr = args.snr_start - (args.snr_start - args.snr_end) * (epoch / (args.epochs - 1))
        
        # Jitter the SNR to force robustness across a wider range
        current_snr_db = np.random.uniform(target_snr - args.snr_jitter, target_snr + args.snr_jitter)
        
        snr_history.append(current_snr_db)

        # Simulate a batch of messages
        messages_indices = torch.randint(0, M, (args.batch_size,)).to(DEVICE)

        optimizer.zero_grad()

        # Transmitter
        encoded_signal = encoder(messages_indices)
        
        # Decide fading scale for this batch
        cur_fading = args.fading_scale
        if args.fading_jitter:
            cur_fading = np.random.uniform(0.0, args.fading_scale)

        # Apply Channel
        distorted = apply_multipath_fading(encoded_signal, n_taps=args.n_taps, 
                                          fading_scale=cur_fading, 
                                          use_circular=args.use_circular)
        distorted = apply_phase_noise(distorted, max_phase_deg=args.max_phase_deg)
        distorted = apply_frequency_offset(distorted, max_freq_step=args.max_freq_step)
        
        # Timing Offset (New Phase 7) - Signal now becomes (Batch, 2, N + max_offset)
        distorted_signal = apply_timing_offset(distorted, max_offset=args.max_offset)
        
        noisy_signal = add_awgn(distorted_signal, current_snr_db)
        
        # Receiver
        decoded_logits = decoder(noisy_signal)

        # Loss Function: Categorical Cross-Entropy + Spectral Penalty
        ce_loss = criterion(decoded_logits, messages_indices)
        
        bw_penalty = get_spectral_penalty(encoded_signal, bw_limit=args.bw_limit)
        papr_penalty = get_papr_penalty(encoded_signal)
        dc_penalty = get_dc_penalty(encoded_signal)
        
        # Calculate total loss with penalty warmup (Phase 11/12 discovery aid)
        warmup = min(1.0, (epoch + 1) / 1000.0)
        loss = ce_loss + warmup * (args.bw_penalty * bw_penalty + 
                                   args.papr_penalty * papr_penalty + 
                                   args.dc_penalty * dc_penalty)
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Calculate Precision BER (Bitwise)
        ber, _ = calculate_bitwise_ber(decoded_logits, messages_indices, args.K)
        ber_history.append(ber)

        if (epoch + 1) % 50 == 0:
            avg_papr = get_papr(encoded_signal)
            print(f"Epoch {epoch+1}/{args.epochs}, SNR: {current_snr_db:.2f} dB, Loss: {loss.item():.4f}, BER: {ber:.4f}, PAPR: {avg_papr:.2f} dB, DC Penalty: {dc_penalty:.2f}, BW Penalty: {bw_penalty:.2f}")

    print("Training finished!")
    return encoder, decoder, train_losses, ber_history, snr_history, all_messages_indices

# --- Visualization ---
def visualize_learned_waveform(encoder, all_messages_indices, args):
    M = 2**args.K
    N_SAMPLES = args.n_samples
    encoder.eval() # Set encoder to evaluation mode
    with torch.no_grad():
        # Get the transmitted signals for all M messages
        waveform_iq_tensor = encoder(all_messages_indices)
        avg_papr = get_papr(waveform_iq_tensor)
        print(f"Final average PAPR of learned waveforms: {avg_papr:.2f} dB")
        learned_waveforms_iq = waveform_iq_tensor.cpu().numpy()
    
    # Plotting for N_SAMPLES > 1 (i.e., not a simple constellation point)
    print("\nVisualizing learned waveforms...")
    
    if N_SAMPLES == 1:
        # Simple I/Q constellation plot (like QAM)
        # learned_waveforms_iq shape is (M, 2, 1), squeeze to (M, 2)
        learned_waveforms_iq = np.squeeze(learned_waveforms_iq, axis=-1)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(learned_waveforms_iq[:, 0], learned_waveforms_iq[:, 1], c='blue', marker='o')
        for i in range(M):
            plt.annotate(str(i), (learned_waveforms_iq[i, 0], learned_waveforms_iq[i, 1]))
        plt.title(f"Learned Constellation (M={M}, N_SAMPLES={N_SAMPLES})")
        plt.xlabel("In-phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.grid(True)
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
    else:
        # Plot each waveform (I and Q components over time)
        # This will be more illustrative for spread spectrum/temporal codes
        cols = 4
        rows = math.ceil(M / cols)
        plt.figure(figsize=(15, 2.5 * rows))
        for i in range(M):
            plt.subplot(rows, cols, i + 1)
            
            # waveform shape is (2, N_SAMPLES)
            waveform = learned_waveforms_iq[i]
            
            plt.plot(waveform[0], label='I') # In-phase component
            plt.plot(waveform[1], label='Q', linestyle='--') # Quadrature component
            plt.title(f"Message {i}")
            plt.ylim(-2, 2) # Keep y-axis consistent for comparison
            plt.grid(True)
            if i == 0: # Only show legend once
                plt.legend()
    plt.suptitle(f"Learned Waveforms for {M} Messages (N_SAMPLES={N_SAMPLES})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs("output", exist_ok=True)
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    filename = f"output/{prefix_str}waveforms_K{args.K}_M{M}_N{N_SAMPLES}.png"
    plt.savefig(filename)
    print(f"Waveform figure saved to {filename}")
    
    if not args.no_gui:
        plt.show()
    else:
        plt.close()

def visualize_waterfall(encoder, args):
    """
    Simulates a 'on-the-air' waterfall by encoding a random packet.
    """
    M = 2**args.K
    N_SAMPLES = args.n_samples
    num_symbols = (args.packet_bytes * 8) // args.K
    
    print(f"\nVisualizing waterfall for a {args.packet_bytes}-byte packet ({num_symbols} symbols)...")
    
    # Generate random symbols
    indices = torch.randint(0, M, (num_symbols,)).to(DEVICE)
    
    encoder.eval()
    with torch.no_grad():
        # Encode (num_symbols, 2, N_SAMPLES)
        waveforms = encoder(indices)
        
        # Reshape to a single continuous stream (2, num_symbols * N_SAMPLES)
        stream = waveforms.transpose(0, 1).reshape(2, -1).unsqueeze(0) # (1, 2, L)
        
        # We show the CLEAN signal as requested (no channel impairments)
        signal = stream.squeeze(0).cpu().numpy()
        complex_signal = signal[0] + 1j * signal[1]
    
    plt.figure(figsize=(12, 8))
    # Waterfall plot (Spectrogram)
    n_fft = max(16, 2**int(np.ceil(np.log2(N_SAMPLES * 2))))
    
    plt.subplot(2, 1, 1)
    plt.specgram(complex_signal, NFFT=n_fft, Fs=1.0, noverlap=n_fft//2, cmap='viridis')
    plt.title(f"Waterfall (Spectrogram) of Clean {args.packet_bytes}-byte Packet (TX Signal)")
    plt.ylabel("Normalized Frequency")
    plt.xlabel("Sample Index")
    plt.colorbar(label='Intensity (dB)')
    
    plt.subplot(2, 1, 2)
    plt.plot(np.real(complex_signal[:1000]), label='I (first 1000 samples)')
    plt.plot(np.imag(complex_signal[:1000]), label='Q', alpha=0.5)
    plt.title("Time-domain signal (fragment)")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    filename = f"output/{prefix_str}waterfall_K{args.K}_N{N_SAMPLES}.png"
    plt.savefig(filename)
    print(f"Waterfall figure saved to {filename}")
    if not args.no_gui:
        plt.show()
    else:
        plt.close()

def visualize_spectrum(encoder, all_messages_indices, args):
    """Computes and plots the Power Spectral Density of all learned waveforms."""
    encoder.eval()
    with torch.no_grad():
        waveform_iq = encoder(all_messages_indices)
    
    batch_size, _, n_samples = waveform_iq.size()
    if n_samples <= 1:
        print("Skipping spectral visualization (N_SAMPLES <= 1)")
        return

    # Convert to complex and compute FFT
    z = torch.complex(waveform_iq[:, 0, :], waveform_iq[:, 1, :])
    Z = torch.fft.fft(z, dim=1)
    Z_shifted = torch.fft.fftshift(Z, dim=1)
    
    # Average PSD over all messages
    psd = torch.abs(Z_shifted)**2
    avg_psd = torch.mean(psd, dim=0).cpu().numpy()
    
    # Frequencies (normalized to -0.5 to 0.5)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_samples))
    
    plt.figure(figsize=(10, 6))
    # PSD in dB, normalized to peak
    psd_db = 10 * np.log10(avg_psd / (np.max(avg_psd) + 1e-12) + 1e-12)
    plt.plot(freqs, psd_db)
    
    # Draw mask limits
    plt.axvline(-args.bw_limit/2, color='r', linestyle='--', label=f'Allowed BW ({args.bw_limit*100:.0f}%)')
    plt.axvline(args.bw_limit/2, color='r', linestyle='--')
    
    # Fill the penalty area for clarity
    plt.fill_between(freqs, -100, 0, where=(np.abs(freqs) > args.bw_limit/2), 
                     color='red', alpha=0.1, label='Penalty Zone')
    
    plt.title(f"Normalized Power Spectral Density (N={n_samples}, BW Limit={args.bw_limit})")
    plt.xlabel("Normalized Frequency (relative to Fs)")
    plt.ylabel("Relative Power (dB)")
    plt.ylim(-60, 5)
    plt.grid(True)
    plt.legend()
    
    os.makedirs("output", exist_ok=True)
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    filename = f"output/{prefix_str}spectrum_K{args.K}_N{n_samples}_BW{args.bw_limit}.png"
    plt.savefig(filename)
    print(f"Spectrum figure saved to {filename}")
    if not args.no_gui:
        plt.show()
    else:
        plt.close()

def draw_model_architecture(args):
    """
    Uses torchview to generate a visualization of the Encoder and Decoder.
    """
    if not HAS_TORCHVIEW:
        print("Error: torchview and graphviz are required for --draw-graph.")
        print("Install them with: uv add torchview graphviz")
        return

    # Mock device for graph generation
    device = torch.device("cpu")
    
    M = 2**args.K
    encoder = Encoder(M, args.n_samples, args).to(device)
    decoder = Decoder(M, args.n_samples, args).to(device)

    print("Generating architecture diagrams in 'output/'...")
    
    # 1. Visualize Encoder
    # Encoder takes indices (Batch,)
    dummy_indices = torch.randint(0, M, (1,)).long().to(device)
    try:
        enc_graph = draw_graph(encoder, input_data=dummy_indices, device='cpu', 
                               graph_name="Encoder", expand_nested=True)
        enc_graph.visual_graph.render(os.path.join("output", "architecture_encoder"), format="png")
    except Exception as e:
        print(f"Error drawing Encoder: {e}")
    
    # 2. Visualize Decoder
    # Decoder input: (Batch, Channels, Time)
    L = args.n_samples + args.max_offset
    dummy_signal = torch.zeros(1, 2, L).to(device)
    try:
        dec_graph = draw_graph(decoder, input_data=dummy_signal, device='cpu', 
                               graph_name="Decoder", expand_nested=True)
        dec_graph.visual_graph.render(os.path.join("output", "architecture_decoder"), format="png")
    except Exception as e:
        print(f"Error drawing Decoder: {e}")

    print("Done. Files saved to output/architecture_encoder.png and output/architecture_decoder.png")

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep-HAM Radio PHY Autoencoder Training')
    parser.add_argument('-K', type=int, default=4, help='Number of bits per message')
    parser.add_argument('-N', '--n-samples', type=int, default=16, help='Number of I/Q samples per message')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--snr-start', type=float, default=10.0, help='Start SNR in dB')
    parser.add_argument('--snr-end', type=float, default=-20.0, help='End SNR in dB')
    parser.add_argument('--max-phase-deg', type=float, default=5.0, help='Max phase ambiguity in degrees (default: 5.0 for stable clock)')
    parser.add_argument('--max-freq-step', type=float, default=0.05, help='Max frequency drift step (default: 0.05 for TCXO-like drift)')
    parser.add_argument('--bw-penalty', type=float, default=0.5, help='Weight of the spectral penalty (default: 0.0)')
    parser.add_argument('--bw-limit', type=float, default=0.5, help='Allowed bandwidth fraction (0.0-1.0, default: 0.5)')
    parser.add_argument('--n-taps', type=int, default=3, help='Number of multipath taps (default: 3)')
    parser.add_argument('--fading-scale', type=float, default=0.5, help='Strength of fading effect (0.0-1.0, default: 0.5)')
    parser.add_argument('--use-circular', action='store_true', help='Use circular convolution for fading (OFDM-like)')
    parser.set_defaults(use_circular=False)
    parser.add_argument('--papr-penalty', type=float, default=1.0, help='Weight of the PAPR constraint (default: 1.0)')
    parser.add_argument('--dc-penalty', type=float, default=0.0, help='Weight of the DC blocking penalty (default: 0.0)')
    parser.add_argument('--snr-jitter', type=float, default=2.0, help='Range of SNR randomization in dB during training (default: 2.0)')
    parser.add_argument('--fading-jitter', action='store_true', help='Randomize fading scale between 0 and --fading-scale per batch (default: False)')
    parser.add_argument('--prefix', type=str, default="", help='Prefix for output filenames')
    parser.add_argument('--packet-bytes', type=int, default=20, help='Number of bytes for waterfall visualization (default: 20)')
    parser.add_argument('--max-offset', type=int, default=0, help='Max random timing offset in samples (default: 0)')
    parser.add_argument('--rolloff', type=float, default=0.0, help='RRC roll-off factor (0.0=off, 0.35 typical, default: 0.0)')
    parser.add_argument('--filter-span', type=int, default=8, help='RRC filter span in symbols (default: 8)')
    parser.add_argument('--draw-graph', action='store_true', help='Draw model architecture diagram and exit (default: False)')
    parser.add_argument('--no-gui', action='store_true', help='Do not display plots, only save them (default: False)')
    args = parser.parse_args()

    if args.no_gui:
        plt.switch_backend('Agg')

    print(f"Using device: {DEVICE}")

    if args.draw_graph:
        draw_model_architecture(args)
        import sys
        sys.exit(0)

    encoder_model, decoder_model, losses, bers, snrs, all_msg_indices = train_autoencoder(args)

    M = 2**args.K
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.plot(bers)
    ax.set_title("BER History vs Epoch/SNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BER")
    
    # Add secondary x-axis for SNR
    def epoch_to_snr(x):
        return args.snr_start - (args.snr_start - args.snr_end) * (x / max(1, args.epochs - 1))

    def snr_to_epoch(x):
        return (args.snr_start - x) * (args.epochs - 1) / (args.snr_start - args.snr_end)

    secax = ax.secondary_xaxis('top', functions=(epoch_to_snr, snr_to_epoch))
    secax.set_xlabel('Corresponding SNR (dB)')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("output", exist_ok=True)
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    filename = f"output/{prefix_str}results_K{args.K}_M{M}_N{args.n_samples}_SNR{args.snr_start:.0f}to{args.snr_end:.0f}.png"
    plt.savefig(filename)
    print(f"Training results figure saved to {filename}")
    
    if not args.no_gui:
        plt.show()
    else:
        plt.close()

    # Visualize the learned waveforms (or constellations if N_SAMPLES=1)
    visualize_learned_waveform(encoder_model, all_msg_indices, args)
    
    # Visualize the spectrum
    visualize_spectrum(encoder_model, all_msg_indices, args)

    # Visualize the waterfall
    visualize_waterfall(encoder_model, args)

    # Deployment and Analysis
    # Save models
    prefix_str = f"{args.prefix}_" if args.prefix else ""
    enc_path = f"output/{prefix_str}encoder.pth"
    dec_path = f"output/{prefix_str}decoder.pth"
    torch.save(encoder_model.state_dict(), enc_path)
    torch.save(decoder_model.state_dict(), dec_path)
    print(f"Models saved to {enc_path} and {dec_path}")

    # SigMF Export
    export_to_sigmf(encoder_model, args)

    # Final detailed SNR vs BER table
    print(f"\nEvaluating detailed BER table from {args.snr_start} to {args.snr_end} dB:")
    generate_ber_table(encoder_model, decoder_model, args, DEVICE)
