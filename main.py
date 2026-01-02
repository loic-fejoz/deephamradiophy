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

# --- 1. The "End-to-End" Architecture ---

# Encoder (Transmitter)

# Encoder (Transmitter)
class Encoder(nn.Module):
    def __init__(self, M, N_SAMPLES):
        super(Encoder, self).__init__()
        self.M = M
        self.N_SAMPLES = N_SAMPLES 

        # We start with a linear layer to map one-hot bits to a latent space
        # then we reshape to (Batch, Channels, Length) for 1D convolutions.
        # We compute the latent dimension from M and N_SAMPLES to ensure 
        # enough expressive power regardless of spreading factor.
        self.latent_dim = max(M, N_SAMPLES)
        self.fc = nn.Linear(M, self.latent_dim)
        
        # 1D CNN to shape the waveform
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1), # Output 2 channels for I and Q
        )

    def forward(self, x):
        # x is (batch, M)
        x = self.fc(x)
        
        # Reshape to (batch, 1, latent_len)
        x = x.view(-1, 1, self.latent_dim)
        x = torch.nn.functional.interpolate(x, size=self.N_SAMPLES, mode='linear', align_corners=False)
        
        waveform_iq = self.net(x) # (batch, 2, N_SAMPLES)
        
        # --- Power Constraint ---
        # Average energy over the 2 channels and N samples
        energy = torch.mean(waveform_iq**2, dim=(1, 2), keepdim=True)
        norm_factor = torch.sqrt(energy)
        waveform_iq = waveform_iq / (norm_factor + 1e-8)

        return waveform_iq

# Decoder (Receiver)
class Decoder(nn.Module):
    def __init__(self, M, N_SAMPLES):
        super(Decoder, self).__init__()
        self.M = M
        self.N_SAMPLES = N_SAMPLES

        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Linear head to get logits for M messages
        self.fc = nn.Linear(16 * N_SAMPLES, M)

    def forward(self, x_noisy):
        # x_noisy is (batch, 2, N_SAMPLES)
        features = self.net(x_noisy)
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

def apply_multipath_fading(waveform_iq, n_taps=3, fading_scale=0.0):
    """
    Simulates Rayleigh Fading using a Tapped Delay Line model.
    n_taps: Number of paths.
    fading_scale: Strength of the fading (0.0 = no fading, 1.0 = full fading).
    """
    if fading_scale <= 0 or n_taps <= 1:
        return waveform_iq
    
    batch_size, channels, n_samples = waveform_iq.size()
    device = waveform_iq.device
    
    # Generate complex Rayleigh coefficients for each tap and each batch
    # Rayleigh amplitude: sqrt(X^2 + Y^2) where X, Y ~ N(0, 1/sqrt(2))
    # Equivalent to complex Gaussian H ~ CN(0, 1)
    # We want the total power across all taps to be 1.0
    coeffs_real = torch.randn(batch_size, n_taps, device=device) / math.sqrt(2 * n_taps)
    coeffs_imag = torch.randn(batch_size, n_taps, device=device) / math.sqrt(2 * n_taps)
    
    # Mix with fading_scale: (1 - scale)*Direct + scale*Faded
    # We apply fading per batch (slow fading assumption: constant over message duration)
    output = waveform_iq * (1.0 - fading_scale)
    
    # Tapped delay line
    faded = torch.zeros_like(waveform_iq)
    for t in range(n_taps):
        # Shift the waveform for each tap
        # We'll use circular shift for simplicity in this discrete model
        shifted_i = torch.roll(waveform_iq[:, 0, :], shifts=t, dims=1)
        shifted_q = torch.roll(waveform_iq[:, 1, :], shifts=t, dims=1)
        
        # Complex multiply: (I + jQ) * (A + jB) = (IA - QB) + j(IB + QA)
        a = coeffs_real[:, t].view(-1, 1)
        b = coeffs_imag[:, t].view(-1, 1)
        
        faded[:, 0, :] += (shifted_i * a - shifted_q * b)
        faded[:, 1, :] += (shifted_i * b + shifted_q * a)
        
    output += fading_scale * faded
    
    # Power Normalization: Maintain constant average power
    # Fading can cause deep nulls or peaks; we normalize to keep SNR definition consistent
    energy = torch.mean(output**2, dim=(1, 2), keepdim=True)
    output = output / (torch.sqrt(energy) + 1e-8)
    
    return output

# --- Training Loop ---
def train_autoencoder(args):
    M = 2**args.K
    encoder = Encoder(M, args.n_samples).to(DEVICE)
    decoder = Decoder(M, args.n_samples).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

    # Generate all possible messages for evaluation and plotting
    all_messages_indices = torch.arange(M).long().to(DEVICE)
    all_messages_one_hot = torch.eye(M).to(DEVICE)[all_messages_indices]

    train_losses = []
    ber_history = []
    snr_history = []

    for epoch in range(args.epochs):
        # --- Noise Curriculum ---
        current_snr_db = args.snr_start - (args.snr_start - args.snr_end) * (epoch / (args.epochs - 1))
        current_snr_db = max(args.snr_end, min(args.snr_start, current_snr_db))
        snr_history.append(current_snr_db)

        # Simulate a batch of messages
        messages_indices = torch.randint(0, M, (args.batch_size,)).to(DEVICE)
        one_hot_messages = torch.eye(M).to(DEVICE)[messages_indices]

        optimizer.zero_grad()

        # Transmitter
        encoded_signal = encoder(one_hot_messages)
        
        # Channel
        distorted_signal = apply_multipath_fading(encoded_signal, n_taps=args.n_taps, fading_scale=args.fading_scale)
        distorted_signal = apply_phase_noise(distorted_signal, max_phase_deg=args.max_phase_deg)
        distorted_signal = apply_frequency_offset(distorted_signal, max_freq_step=args.max_freq_step)
        noisy_signal = add_awgn(distorted_signal, current_snr_db)
        
        # Receiver
        decoded_logits = decoder(noisy_signal)

        # Loss Function: Categorical Cross-Entropy + Spectral Penalty
        ce_loss = criterion(decoded_logits, messages_indices)
        
        bw_penalty = get_spectral_penalty(encoded_signal, bw_limit=args.bw_limit)
        loss = ce_loss + args.bw_penalty * bw_penalty
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Calculate BER (Bit Error Rate) for monitoring
        _, predicted_messages = torch.max(decoded_logits, 1)
        
        # Convert message errors to bit errors
        # Assuming Gray coding for converting message errors to bit errors would be more accurate
        # but for simplicity, we'll just count how many bits would be wrong if the message is wrong.
        num_message_errors = (predicted_messages != messages_indices).sum().item()
        num_bit_errors = 0
        
        if num_message_errors > 0:
            # A simple approximation: if a message is wrong, assume K/2 bits are wrong on average
            # For a more precise calculation, you'd convert to binary and compare bit by bit.
            num_bit_errors = num_message_errors * (args.K / 2) # Average bits wrong per message error

        total_bits = args.batch_size * args.K
        ber = num_bit_errors / total_bits
        ber_history.append(ber)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, SNR: {current_snr_db:.2f} dB, Loss: {loss.item():.4f}, BER: {ber:.4f}")

    print("Training finished!")
    return encoder, decoder, train_losses, ber_history, snr_history, all_messages_one_hot

# --- Visualization ---
def visualize_learned_waveform(encoder, all_messages_one_hot, args):
    M = 2**args.K
    N_SAMPLES = args.n_samples
    encoder.eval() # Set encoder to evaluation mode
    with torch.no_grad():
        # Get the transmitted signals for all M messages
        learned_waveforms_iq = encoder(all_messages_one_hot).cpu().numpy()
    
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
    filename = f"output/waveforms_K{args.K}_M{M}_N{N_SAMPLES}.png"
    plt.savefig(filename)
    print(f"Waveform figure saved to {filename}")
    
    plt.show()

def visualize_spectrum(encoder, all_messages_one_hot, args):
    """Computes and plots the Power Spectral Density of all learned waveforms."""
    encoder.eval()
    with torch.no_grad():
        waveform_iq = encoder(all_messages_one_hot)
    
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
    filename = f"output/spectrum_K{args.K}_N{n_samples}_BW{args.bw_limit}.png"
    plt.savefig(filename)
    print(f"Spectrum figure saved to {filename}")
    plt.show()

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
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    encoder_model, decoder_model, losses, bers, snrs, all_msg_one_hot = train_autoencoder(args)

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
    filename = f"output/results_K{args.K}_M{M}_N{args.n_samples}_SNR{args.snr_start:.0f}to{args.snr_end:.0f}.png"
    plt.savefig(filename)
    print(f"Training results figure saved to {filename}")
    
    plt.show()

    # Visualize the learned waveforms (or constellations if N_SAMPLES=1)
    visualize_learned_waveform(encoder_model, all_msg_one_hot, args)
    
    # Visualize the spectrum
    visualize_spectrum(encoder_model, all_msg_one_hot, args)

    # Example of how to evaluate BER at a specific SNR after training
    print("\nEvaluating BER at target SNR_END_DB:")
    test_snr_db = args.snr_end 
    num_test_batches = 100
    total_bit_errors = 0
    total_bits_evaluated = 0

    encoder_model.eval()
    decoder_model.eval()

    with torch.no_grad():
        for _ in tqdm(range(num_test_batches), desc="BER Evaluation"):
            messages_indices = torch.randint(0, M, (args.batch_size,)).to(DEVICE)
            one_hot_messages = torch.eye(M).to(DEVICE)[messages_indices]

            encoded_signal = encoder_model(one_hot_messages)
            distorted_signal = apply_multipath_fading(encoded_signal, n_taps=args.n_taps, fading_scale=args.fading_scale)
            distorted_signal = apply_phase_noise(distorted_signal, max_phase_deg=args.max_phase_deg)
            distorted_signal = apply_frequency_offset(distorted_signal, max_freq_step=args.max_freq_step)
            noisy_signal = add_awgn(distorted_signal, test_snr_db)
            decoded_logits = decoder_model(noisy_signal)

            _, predicted_messages = torch.max(decoded_logits, 1)
            
            num_message_errors = (predicted_messages != messages_indices).sum().item()
            total_bit_errors += num_message_errors * (args.K / 2) # Approximation
            total_bits_evaluated += args.batch_size * args.K

    final_ber = total_bit_errors / total_bits_evaluated
    print(f"Final BER at {test_snr_db:.2f} dB: {final_ber:.6f}")
