
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def get_spectral_penalty(waveform_iq, bw_limit=0.5):
    batch_size, _, n_samples = waveform_iq.size()
    z = torch.complex(waveform_iq[:, 0, :], waveform_iq[:, 1, :])
    Z = torch.fft.fft(z, dim=1)
    Z_shifted = torch.fft.fftshift(Z, dim=1)
    psd = torch.abs(Z_shifted)**2
    total_bins = n_samples
    allowed_bins = int(total_bins * bw_limit)
    start_bin = (total_bins - allowed_bins) // 2
    end_bin = start_bin + allowed_bins
    mask = torch.ones(total_bins, device=waveform_iq.device)
    mask[start_bin:end_bin] = 0.0
    penalty = torch.mean(psd * mask)
    return penalty, psd, mask

# Test with N=64
N = 64
bw_limit = 0.5

# 1. DC signal (should have 0 penalty)
dc_signal = torch.zeros(1, 2, N)
dc_signal[:, 0, :] = 1.0 # Constant I
penalty_dc, psd_dc, mask = get_spectral_penalty(dc_signal, bw_limit=bw_limit)
print(f"DC Penalty: {penalty_dc.item():.4f} (Expected: 0.0)")

# 2. High frequency signal (should have high penalty)
t = torch.arange(N).float()
freq = 0.45 # Near Nyquist
hf_signal = torch.zeros(1, 2, N)
hf_signal[:, 0, :] = torch.cos(2 * np.pi * freq * t)
hf_signal[:, 1, :] = torch.sin(2 * np.pi * freq * t)
penalty_hf, psd_hf, _ = get_spectral_penalty(hf_signal, bw_limit=bw_limit)
print(f"HF Penalty: {penalty_hf.item():.4f} (Expected: High)")

# 3. Visualize
freqs = np.linspace(-0.5, 0.5, N)
plt.figure(figsize=(10, 6))
plt.plot(freqs, psd_dc[0].numpy(), label='DC PSD')
plt.plot(freqs, psd_hf[0].numpy(), label='HF PSD')
plt.plot(freqs, mask.numpy() * 100, label='Mask (x100)', alpha=0.3)
plt.axvline(-bw_limit/2, color='r', linestyle='--')
plt.axvline(bw_limit/2, color='r', linestyle='--')
plt.legend()
plt.title("Spectral Penalty Verification")
plt.savefig("test_spectrum.png")
print("Saved verification plot to test_spectrum.png")
