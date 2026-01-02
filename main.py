import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars
import os

# --- Configuration Parameters ---
K = 3           # Number of bits per message (log2(M))
M = 2**K        # Number of possible messages (e.g., 16-ary, so 4 bits per message)
N_SAMPLES = 1   # Number of I/Q samples per message (bandwidth expansion factor)
                # Higher N_SAMPLES allows for more complex spreading/waveform
BATCH_SIZE = 1024
NUM_EPOCHS = 2000
LEARNING_RATE = 0.001
SNR_START_DB = 10.0 # Start training at high SNR
SNR_END_DB = 0.0  # End training at low SNR (sub-noise floor)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. The "End-to-End" Architecture ---

# Encoder (Transmitter)
class Encoder(nn.Module):
    def __init__(self, M, N_SAMPLES):
        super(Encoder, self).__init__()
        self.M = M
        self.N_SAMPLES = N_SAMPLES # Number of complex I/Q samples per message

        # Input: one-hot vector for M messages
        # Output: N_SAMPLES * 2 (for I and Q components)
        self.net = nn.Sequential(
            nn.Linear(M, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_SAMPLES * 2) # Output I/Q pairs
        )

    def forward(self, x):
        # x is a one-hot vector of shape (batch_size, M)
        waveform_iq = self.net(x)
        
        # Reshape to (batch_size, N_SAMPLES, 2) for complex representation
        # waveform_iq = waveform_iq.view(-1, self.N_SAMPLES, 2)

        # --- Power Constraint (Normalization Layer) ---
        # Calculate energy for each waveform: Sum of squares of I and Q components
        # Reshape for energy calculation: (batch_size, N_SAMPLES * 2) -> (batch_size, N_SAMPLES * 2)
        energy = torch.mean(waveform_iq**2, dim=1, keepdim=True) # Average energy per sample
        
        # Normalize to ensure average energy per sample is 1 (or desired power)
        # This implicitly sets the average power of the transmitted signal.
        # We want the *total* power of the N_SAMPLES to be normalized, not each sample individually.
        # Let's normalize total energy per message to 1 (average over N_SAMPLES)
        norm_factor = torch.sqrt(energy) # Size (batch_size, 1)
        waveform_iq = waveform_iq / (norm_factor + 1e-8) # Add small epsilon for stability

        return waveform_iq

# Decoder (Receiver)
class Decoder(nn.Module):
    def __init__(self, M, N_SAMPLES):
        super(Decoder, self).__init__()
        self.M = M
        self.N_SAMPLES = N_SAMPLES

        # Input: N_SAMPLES * 2 (noisy I/Q samples)
        # Output: M (logits for M messages)
        self.net = nn.Sequential(
            nn.Linear(N_SAMPLES * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, M) # Output logits for M messages
        )

    def forward(self, x_noisy):
        # x_noisy is (batch_size, N_SAMPLES * 2)
        logits = self.net(x_noisy)
        return logits

# --- The Channel Layer ---
def add_awgn(waveform_iq, snr_db):
    """
    Adds Additive White Gaussian Noise to the I/Q samples.
    
    Args:
        waveform_iq (Tensor): Transmitted I/Q samples (batch_size, N_SAMPLES * 2).
        snr_db (float): Signal-to-noise ratio in decibels.
        
    Returns:
        Tensor: Noisy I/Q samples.
    """
    # Calculate linear SNR from dB
    snr_linear = 10**(snr_db / 10.0)
    
    # Assuming the transmitted signal (waveform_iq) has an average power of 1
    # Noise power = Signal power / SNR_linear = 1 / SNR_linear
    noise_power = 1.0 / snr_linear
    
    # Standard deviation of noise (for I and Q components independently)
    # The noise is complex, so we split the power between I and Q.
    # Each component (I and Q) gets half the total noise power.
    # variance of real noise = noise_power / 2
    noise_std = np.sqrt(noise_power / 2) 
    
    # Generate noise with same shape as waveform_iq
    noise = torch.randn_like(waveform_iq) * noise_std
    
    return waveform_iq + noise

# --- Training Loop ---
def train_autoencoder():
    encoder = Encoder(M, N_SAMPLES).to(DEVICE)
    decoder = Decoder(M, N_SAMPLES).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    # Generate all possible messages for evaluation and plotting
    all_messages_indices = torch.arange(M).long().to(DEVICE)
    all_messages_one_hot = torch.eye(M).to(DEVICE)[all_messages_indices]

    train_losses = []
    ber_history = []
    snr_history = []

    for epoch in range(NUM_EPOCHS):
        # --- Noise Curriculum ---
        # Linearly interpolate SNR_dB from START to END
        current_snr_db = SNR_START_DB - (SNR_START_DB - SNR_END_DB) * (epoch / (NUM_EPOCHS - 1))
        
        # Ensure it doesn't go below END_DB or above START_DB
        current_snr_db = max(SNR_END_DB, min(SNR_START_DB, current_snr_db))
        snr_history.append(current_snr_db)

        # Simulate a batch of messages
        messages_indices = torch.randint(0, M, (BATCH_SIZE,)).to(DEVICE)
        one_hot_messages = torch.eye(M).to(DEVICE)[messages_indices] # Convert to one-hot

        optimizer.zero_grad()

        # Transmitter
        encoded_signal = encoder(one_hot_messages)
        
        # Channel
        noisy_signal = add_awgn(encoded_signal, current_snr_db)
        
        # Receiver
        decoded_logits = decoder(noisy_signal)

        # Loss Function Selection: Categorical Cross-Entropy
        loss = criterion(decoded_logits, messages_indices)
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
            num_bit_errors = num_message_errors * (K / 2) # Average bits wrong per message error

        total_bits = BATCH_SIZE * K
        ber = num_bit_errors / total_bits
        ber_history.append(ber)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, SNR: {current_snr_db:.2f} dB, Loss: {loss.item():.4f}, BER: {ber:.4f}")

    print("Training finished!")
    return encoder, decoder, train_losses, ber_history, snr_history, all_messages_one_hot

# --- Visualization ---
def visualize_learned_waveform(encoder, all_messages_one_hot):
    encoder.eval() # Set encoder to evaluation mode
    with torch.no_grad():
        # Get the transmitted signals for all M messages
        learned_waveforms_iq = encoder(all_messages_one_hot).cpu().numpy()
    
    # Plotting for N_SAMPLES > 1 (i.e., not a simple constellation point)
    print("\nVisualizing learned waveforms...")
    
    if N_SAMPLES == 1:
        # Simple I/Q constellation plot (like QAM)
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
        plt.figure(figsize=(15, 10))
        for i in range(M):
            plt.subplot(M // 4, 4, i + 1) # Arrange in a grid, e.g., 4x4 for M=16
            
            # Reshape from (N_SAMPLES * 2) to (N_SAMPLES, 2) for I and Q
            waveform = learned_waveforms_iq[i].reshape(N_SAMPLES, 2)
            
            plt.plot(waveform[:, 0], label='I') # In-phase component
            plt.plot(waveform[:, 1], label='Q', linestyle='--') # Quadrature component
            plt.title(f"Message {i}")
            plt.ylim(-2, 2) # Keep y-axis consistent for comparison
            plt.grid(True)
            if i == 0: # Only show legend once
                plt.legend()
        plt.suptitle(f"Learned Waveforms for {M} Messages (N_SAMPLES={N_SAMPLES})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    os.makedirs("output", exist_ok=True)
    filename = f"output/waveforms_K{K}_M{M}_N{N_SAMPLES}.png"
    plt.savefig(filename)
    print(f"Waveform figure saved to {filename}")
    
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    encoder_model, decoder_model, losses, bers, snrs, all_msg_one_hot = train_autoencoder()

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
    # SNR curriculum is linear: current_snr_db = SNR_START_DB - (SNR_START_DB - SNR_END_DB) * (epoch / (NUM_EPOCHS - 1))
    def epoch_to_snr(x):
        return SNR_START_DB - (SNR_START_DB - SNR_END_DB) * (x / max(1, NUM_EPOCHS - 1))

    def snr_to_epoch(x):
        return (SNR_START_DB - x) * (NUM_EPOCHS - 1) / (SNR_START_DB - SNR_END_DB)

    secax = ax.secondary_xaxis('top', functions=(epoch_to_snr, snr_to_epoch))
    secax.set_xlabel('Corresponding SNR (dB)')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("output", exist_ok=True)
    filename = f"output/results_K{K}_M{M}_N{N_SAMPLES}_SNR{SNR_START_DB:.0f}to{SNR_END_DB:.0f}.png"
    plt.savefig(filename)
    print(f"Training results figure saved to {filename}")
    
    plt.show()

    # Visualize the learned waveforms (or constellations if N_SAMPLES=1)
    visualize_learned_waveform(encoder_model, all_msg_one_hot)

    # Example of how to evaluate BER at a specific SNR after training
    print("\nEvaluating BER at target SNR_END_DB:")
    test_snr_db = SNR_END_DB # Or any other SNR
    num_test_batches = 100
    total_bit_errors = 0
    total_bits_evaluated = 0

    encoder_model.eval()
    decoder_model.eval()

    with torch.no_grad():
        for _ in tqdm(range(num_test_batches), desc="BER Evaluation"):
            messages_indices = torch.randint(0, M, (BATCH_SIZE,)).to(DEVICE)
            one_hot_messages = torch.eye(M).to(DEVICE)[messages_indices]

            encoded_signal = encoder_model(one_hot_messages)
            noisy_signal = add_awgn(encoded_signal, test_snr_db)
            decoded_logits = decoder_model(noisy_signal)

            _, predicted_messages = torch.max(decoded_logits, 1)
            
            num_message_errors = (predicted_messages != messages_indices).sum().item()
            total_bit_errors += num_message_errors * (K / 2) # Approximation
            total_bits_evaluated += BATCH_SIZE * K

    final_ber = total_bit_errors / total_bits_evaluated
    print(f"Final BER at {test_snr_db:.2f} dB: {final_ber:.6f}")
