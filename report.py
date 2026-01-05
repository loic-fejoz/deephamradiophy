import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from tqdm import tqdm
from main import Encoder, Decoder, evaluate_at_snr, DEVICE

def parse_args():
    parser = argparse.ArgumentParser(description='Deep-HAM Radio PHY Report Generator')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for model files (e.g. experiment_1)')
    parser.add_argument('-K', type=int, default=4, help='Number of bits per message')
    parser.add_argument('-N', '--n-samples', type=int, default=256, help='Number of I/Q samples per message')
    parser.add_argument('--max-offset', type=int, default=64, help='Max random timing offset in samples')
    parser.add_argument('--rolloff', type=float, default=0.33, help='RRC roll-off factor')
    parser.add_argument('--filter-span', type=int, default=8, help='RRC filter span')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for evaluation')
    parser.add_argument('--n-taps', type=int, default=3, help='Number of multipath taps')
    parser.add_argument('--use-circular', action='store_true', help='Use circular convolution')
    parser.add_argument('--max-phase-deg', type=float, default=5.0, help='Max phase ambiguity')
    parser.add_argument('--max-freq-step', type=float, default=0.05, help='Max frequency drift')
    
    # Sweep ranges
    parser.add_argument('--snr-start', type=float, default=10.0)
    parser.add_argument('--snr-end', type=float, default=-15.0)
    parser.add_argument('--snr-step', type=float, default=1.0)
    parser.add_argument('--fading-start', type=float, default=0.0)
    parser.add_argument('--fading-end', type=float, default=1.0)
    parser.add_argument('--fading-step', type=float, default=0.2)
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Generating report for prefix: {args.prefix}")
    
    M = 2**args.K
    encoder = Encoder(M, args.n_samples, args).to(DEVICE)
    decoder = Decoder(M, args.n_samples, args).to(DEVICE)
    
    # Load weights
    enc_path = f"output/{args.prefix}_encoder.pth"
    dec_path = f"output/{args.prefix}_decoder.pth"
    
    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        print(f"Error: Model files not found at {enc_path} or {dec_path}")
        return
        
    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE))
    encoder.eval()
    decoder.eval()
    print("Models loaded successfully.")

    # 3D Sweep
    snr_range = np.arange(args.snr_start, args.snr_end - 0.1, -args.snr_step)
    fading_range = np.arange(args.fading_start, args.fading_end + 0.01, args.fading_step)
    
    SNR, FADING = np.meshgrid(snr_range, fading_range)
    BER = torch.zeros(len(fading_range), len(snr_range))
    
    print(f"Starting 3D sweep (SNR: {args.snr_start} to {args.snr_end}, Fading: {args.fading_start} to {args.fading_end})...")
    
    for i, f_scale in enumerate(tqdm(fading_range, desc="Fading Sweep")):
        # Temporarily override args.fading_scale for evaluate_at_snr
        args.fading_scale = f_scale
        for j, s_db in enumerate(snr_range):
            ber = evaluate_at_snr(encoder, decoder, s_db, args, DEVICE, num_batches=20)
            BER[i, j] = ber

    # 3D Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(SNR, FADING, BER.numpy(), cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Fading Scale')
    ax.set_zlabel('Bit Error Rate (BER)')
    ax.set_title(f'BER vs SNR and Fading Scale (Prefix: {args.prefix})')
    
    # Log scale for BER axis is tricky in 3D, keeping linear but user asked for Z as upward axis
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plot_filename = f"output/{args.prefix}_3d_ber.png"
    plt.savefig(plot_filename)
    print(f"3D Plot saved to {plot_filename}")

    # Generate Report
    report_path = f"output/{args.prefix}_report.md"
    with open(report_path, 'w') as f:
        f.write(f"# Evaluation Report: {args.prefix}\n\n")
        
        f.write("## Training Parameters\n")
        f.write(f"- **K (Bits per Message)**: {args.K} (M={M})\n")
        f.write(f"- **N (IQ Samples)**: {args.n_samples}\n")
        f.write(f"- **Max Timing Offset**: {args.max_offset}\n")
        f.write(f"- **RRC Roll-off**: {args.rolloff}\n")
        f.write(f"- **RRC Filter Span**: {args.filter_span}\n")
        f.write(f"- **Multipath Taps**: {args.n_taps}\n")
        f.write(f"- **Phase Ambiguity**: {args.max_phase_deg}°\n")
        f.write(f"- **Freq Drift Step**: {args.max_freq_step}\n\n")

        f.write("## Training Artifacts\n")
        f.write(f"![Waveforms]({args.prefix}_waveforms_K{args.K}_M{M}_N{args.n_samples}.png)\n")
        f.write(f"![Spectrum]({args.prefix}_spectrum_K{args.K}_N{args.n_samples}_BW0.5.png)\n")
        f.write(f"![Waterfall]({args.prefix}_waterfall_K{args.K}_N{args.n_samples}.png)\n\n")
        
        f.write("## Persistence & Deployment\n")
        f.write(f"- Encoder Model: [encoder.pth]({args.prefix}_encoder.pth)\n")
        f.write(f"- Decoder Model: [decoder.pth]({args.prefix}_decoder.pth)\n")
        f.write(f"- SigMF Data: [sigmf-data]({args.prefix}_learned_waveforms.sigmf-data)\n")
        f.write(f"- SigMF Meta: [sigmf-meta]({args.prefix}_learned_waveforms.sigmf-meta)\n\n")
        
        f.write("## 3D BER Analysis\n")
        f.write("This plot shows the BER performance across both Signal-to-Noise Ratio (SNR) and Fading Scale (multipath intensity).\n\n")
        f.write(f"![3D BER Waterfall]({args.prefix}_3d_ber.png)\n\n")
        
        f.write("### BER Data Table\n")
        # Header
        f.write("| Fading \\ SNR | " + " | ".join([f"{s:.1f}" for s in snr_range]) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(snr_range)) + "|\n")
        # Rows
        for i, f_scale in enumerate(fading_range):
            row = [f"{f_scale:.1f}"] + [f"{BER[i, j]:.4f}" for j in range(len(snr_range))]
            f.write("| " + " | ".join(row) + " |\n")

    print(f"Markdown report generated at {report_path}")

if __name__ == "__main__":
    main()
