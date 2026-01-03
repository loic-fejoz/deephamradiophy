# AGENTS.md: Project "Deep-PHY"

**Objective:** Discover a novel Physical Layer (PHY) for VHF/UHF Ham Radio using End-to-End Deep Learning (Autoencoders). Target sub-noise floor performance (negative SNR) and high resilience to terrestrial impairments.



## 1. Project Vision

Design a non-traditional modulation scheme that moves beyond FSK, LoRa, and BPSK. By treating the transmitter (Encoder) and receiver (Decoder) as a single differentiable neural network, we aim to "evolve" a waveform optimized for low-speed, high-robustness amateur radio communication.

## 2. Technical Stack

* **Language:** Python 3.10+
* **Framework:** PyTorch (for differentiable DSP)
* **Analysis:** NumPy, Matplotlib, SciPy (FFT Analysis)
* **Environment:** Discrete-time baseband simulation ( sampling)

## 3. Core Architecture: The Autoencoder Pipe

The simulation engine must implement the following pipeline:

1. **Input:** Message  (One-hot encoded).
2. **Encoder (TX):** Dense or 1D-CNN layers mapping bits to  complex samples.
3. **Normalization Layer:** Enforces average power constraints ().
4. **Differentiable Channel:** * **AWGN:** Variable SNR (Curriculum based).
* **Phase Noise:** Stochastic phase rotation.
* **Frequency Offset:** Simulation of TCXO drift.
5. **Decoder (RX):** Dense or 1D-CNN layers mapping noisy samples back to  message probabilities.
6. **Loss Function:** Categorical Cross-Entropy.



## 4. Phase 1: The Initial Experiment (MVP)

The goal of the first script is to prove the machine can learn a resilient constellation in pure noise.

### Implementation Checklist:

* [x] Implement `Encoder` with 2-3 hidden layers.
* [x] Implement `add_awgn` with linear SNR conversion.
* [x] Implement **Noise Curriculum**: Linear interpolation of SNR over training.
* [x] Implement `visualize_learned_waveform`: Plot I/Q temporal signals and a Scatter plot of the learned "constellation."

## 5. Current Focus: Robustness & Real-World Bridge

With simple constellations working, we must now move towards waveforms that can survive hardware impairments.

### Step 2: Stochastic Impairments (Phase & Frequency)
*   [x] Add stochastic **Phase Noise** and **Frequency Offset** to the channel layer.
*   **Goal:** Force the Decoder to become invariant to rotation and small frequency shifts.
*   **Metric:** Measure BER degradation vs Frequency Offset (Hz).

### Step 3: Spectral Masking & Bandwidth Efficiency
*   [x] Add an FFT-based penalty to the Loss Function.
*   **Goal:** Keep the AI from "spreading" into neighbors' channels. Ensure the waveform fits within a 12.5kHz mask.
*   **Constraint:** Use a 1D-CNN or larger $N$ to discover frequency-diversity schemes.

### Step 4: Multipath & Fading (Terrestrial Modeling)
*   [x] Add a Tapped Delay Line model (Rayleigh Fading) to the channel.
*   [x] **Goal:** Discover a waveform that resists "deep fades" (Rayleigh Fading).
*   **Result:** Discovered PPM-like waveforms with high spectral redundancy.

### Step 5: PAPR Optimization for Hardware
*   [x] Implement Peak-to-Average Power Ratio (PAPR) calculation.
*   [x] Add a penalty to the loss function to favor constant-envelope waveforms.
*   [x] Verified: PAPR reduced from 1.58 dB to 0.39 dB in training.
*   **Goal:** Ensure the waveform can be transmitted by cheap, non-linear Class-C amateur radio amplifiers without massive distortion.

### Step 6: Waterfall Visualization & Experiment Diagnostics
*   [x] Implement `visualize_waterfall` to show "on-the-air" packet behavior.
*   [x] Add `--prefix` CLI argument for better experiment tracking.
*   **Result:** Verified the combinatorial nature of learned MPPM waveforms.

### Step 7: Timing Discovery & Synchronization
*   [x] Introduce random **Sample Offsets** (delay) in the channel.
*   [x] **Goal:** The model must discover its own "Preamble" or "Sync Word" to align the message in time.
*   [x] **Result:** Successfully trained with 8-sample random shifts; Decoder learned to "scan" the window for features.

### Step 8: High-Expressive Encoder & Sequence Decoder
*   [x] Replace simple linear interpolation with **Transposed Convolutions** (`ConvTranspose1d`) and **Embeddings**.
*   [x] **Goal:** Allow the network to learn its own upsampling/shaping filters. Investigate sequence-aware models (GRU).
*   [x] **Result:** Improved convergence significantly; Bi-GRU handles timing offsets natively.

### Step 9: Precision Metrics & Gray Coding
*   [x] Implement actual bitwise comparison for BER calculation instead of approximations.
*   [x] **Goal:** Enable discovery of **Gray Coding** where similar waveforms map to messages with low Hamming distance.
*   [x] **Result:** Bitwise BER integrated and used for gradient tracking.

### Step 10: Real-time Dynamics & Channel Toggles
*   [x] Add symbol-to-symbol phase jitter and time-varying frequency drift (in progress/partial).
*   [x] **Goal:** Increase robustness for longer packets. Add a toggle for **Linear vs. Circular** multipath convolution.
*   [x] **Result:** Linear convolution fading model implemented.

### Step 11: Advanced Timing Synchronization
*   **TODO:** Implement correlation-based pre-processors or **Spatial Transformer Networks (STN)**.
*   **Goal:** Achieve high-accuracy timing alignment for large `max_offset` values (e.g., > 100 samples).

### Step 12: Pulse Shaping & Nyquist Filtering
*   **TODO:** Replace simple I/Q pulses with **Root-Raised Cosine (RRC)** filtering or a learned differentiable filter.
*   **Goal:** Minimize Out-of-Band (OOB) emissions while maintaining Inter-Symbol Interference (ISI) control.
*   **Metric:** Adjacent Channel Power Ratio (ACPR).

## 6. Target Metrics (Ongoing)

* **Sensitivity:** Successful decode at $-20$ dB SNR (AWGN).
* **Speed:** "Modest" throughput (50 bps – 500 bps).
* **Bandwidth:** Fits within a standard 12.5 kHz (Narrow FM) channel.

## 7. Documentation Standards

*   **README.md**: Maintain a "Hypotheses and Choices" section. Every physical parameter added to the simulation (Phase Noise, Drift, Bandwidth) must be grounded in real-world Ham Radio hardware specifications (e.g., TCXO ppm, 12.5kHz channel spacing).

