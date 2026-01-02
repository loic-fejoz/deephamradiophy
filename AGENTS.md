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

* [ ] Implement `Encoder` with 2-3 hidden layers.
* [ ] Implement `add_awgn` with linear SNR conversion.
* [ ] Implement **Noise Curriculum**: Start at +10dB, decay to -15dB over 1000 epochs.
* [ ] Implement `visualize_learned_waveform`: Plot I/Q temporal signals and a Scatter plot of the learned "constellation."


## 5. Next Steps & TODOs (Research Roadmap)

### Step 2: Spectral Analysis & MFSK Discovery

* [ ] **TODO:** Add an FFT block to the visualization suite.
* [ ] **Goal:** Check if the AI is using Frequency Diversity (MFSK) or Time Diversity (Spreading).
* [ ] **Constraint:** Introduce a "Bandwidth Mask" to penalize the AI for using frequencies outside a designated range.

### Step 3: Timing & Sync (The "Real World" Bridge)

* [ ] **TODO:** Implement a random time-delay in the channel.
* [ ] **Goal:** Force the AI to discover/learn a "Preamble" or a self-synchronizing waveform (like a Zadoff-Chu sequence or Barker code).

### Step 4: Multipath & Fading (Terrestrial Modeling)

* [ ] **TODO:** Add a Tapped Delay Line model (Rayleigh Fading) to the channel.
* [ ] **Goal:** Discover a waveform that resists "deep fades" common on 2m/70cm bands in urban environments.

### Step 5: PAPR Optimization for Hardware

* [ ] **TODO:** Add a penalty to the Loss Function for high Peak-to-Average Power Ratio.
* [ ] **Goal:** Ensure the resulting waveform can be transmitted by cheap, non-linear Class-C amateur radio amplifiers without massive distortion.


## 6. Target Metrics

* **Sensitivity:** Successful decode at  dB SNR (AWGN).
* **Speed:** "Modest" throughput (50 bps – 500 bps).
* **Bandwidth:** Fits within a standard 3 kHz (SSB) or 12.5 kHz (Narrow FM) channel.

