# Deep-HAM Radio PHY

Experimentation with Machine Learning (ML) techniques by developing novel Physical Layer (PHY) modulations using Deep Learning Autoencoders for Amateur Radio communication.

## Objective

The goal of this project is to discover non-traditional modulation schemes that move beyond standard FSK, LoRa, or BPSK. By treating the transmitter (Encoder) and receiver (Decoder) as a single differentiable neural network, we aim to "evolve" waveforms optimized for high robustness and sub-noise floor performance (negative SNR) on VHF/UHF bands. It is highly inspired by [RADE Radio Autoencoder](https://freedv.org/radio-autoencoder/).

## Architecture

The system implements an end-to-end differentiable DSP pipeline:

1.  **Input:** Message $M$ (One-hot encoded).
2.  **Encoder (TX):** Neural network mapping bits to complex I/Q samples.
3.  **Normalization:** Enforces average power constraints.
4.  **Channel:** Differentiable simulation of AWGN, phase noise, and frequency offsets (TBD).
5.  **Decoder (RX):** Neural network mapping noisy samples back to message probabilities.
6.  **Loss:** Categorical Cross-Entropy.

## Hypotheses and Choices

To ground the autoencoder in real-world Ham Radio physics, we make the following assumptions:

| Parameter | Default Value | Hypothesis / Choice |
| :--- | :--- | :--- |
| **Phase Noise** | $5.0^\circ$ | Represents the integrated phase jitter of a locked synthesizer in a 12.5 kHz narrowband FM channel. |
| **Freq Offset** | $0.05$ | Simulates a $\pm 1$ ppm TCXO drift (approx. $150$ Hz error at $144$ MHz/VHF) under a $12$ kHz sampling assumption. |
| **Latency ($N$)** | $16$ | Spreading factor for $K=4$. We assume a low-speed robust payload where bandwidth can be traded for sensitivity. |
| **SNR range** | $+10 \to -20$ dB | The goal is to discover modulations that remain decodable even when the signal is deep in the noise floor. |
- **PAPR (Peak-to-Average Power Ratio)**: Most Ham Radio power amplifiers are non-linear (Class-C). We optimize for low PAPR so the AI discovers signals that don't "splatter" when amplified.

## Technical Stack

*   **Python 3.12+**
*   **PyTorch** (Differentiable DSP and GPU acceleration)
*   **uv** (Package management and CUDA environment handling)
*   **NumPy, Matplotlib** (Analysis and visualization)

## Usage

You can run the simulation using `uv`:

```bash
uv run python main.py -K 4 -N 16 --snr-end -20
```

### Main Parameters:
- `-K`: Spreading Factor / Bits per message (default: 4)
- `-N`: Number of complex I/Q samples per message (default: 16)
- `--snr-start`: Training start SNR in dB (default: 10)
- `--snr-end`: Training target SNR in dB (default: -20)
- `--epochs`: Number of training iterations (default: 2000)
- `--bw-penalty`: Weight of the spectral penalty (default: 0.0)
- `--bw-limit`: Fraction of total bandwidth allowed (0.0-1.0, default: 0.5)
- `--fading-scale`: Strength of fading (0.0-1.0, default: 0.5).
- `--papr-penalty`: Weight of constant-envelope constraint (default: 1.0).
- `--max-offset`: Maximum random timing shift in samples (default: 0).
- `--max-phase-deg`: Max phase noise in degrees (default: 5.0)
- `--max-freq-step`: Max frequency drift (default: 0.05)

---

## Experiments

This section documents the various attempts and iterations in discovering resilient waveforms.

### 1. Initial Constellation Learning (AWGN Only)

The first phase focuses on proving the model can learn classical constellations in pure noise across different message sizes and with positive or null SNRs.

And indeed the results show that the system is able to learn the usual constellations.
The constellation is not perfectly align on I/Q axis as it obviously does not matter for the learning process.

#### K=1 (2-ary, 1 bit per message)
*   **Samples per message (N):** 1
*   **SNR Curriculum:** 10dB $\to$ 0dB

![BER Results K1](output/results_K1_M2_N1_SNR10to0.png)
![Waveforms K1](output/waveforms_K1_M2_N1.png)

#### 4-PSK K=2 (4-ary, 2 bits per message)
*   **Samples per message (N):** 1
*   **SNR Curriculum:** 10dB $\to$ 0dB

![BER Results K2](output/results_K2_M4_N1_SNR10to0.png)
![Waveforms K2](output/waveforms_K2_M4_N1.png)

#### 8-PSK — K=3 (8-ary, 3 bits per message)
*   **Samples per message (N):** 1
*   **SNR Curriculum:** 10dB $\to$ 0dB

![BER Results K3](output/results_K3_M8_N1_SNR10to0.png)
![Waveforms K3](output/waveforms_K3_M8_N1.png)

#### 16-PSK — K=4 (16-ary, 4 bits per message)
*   **Samples per message (N):** 1
*   **SNR Curriculum:** 10dB $\to$ 0dB

![BER Results K4](output/results_K4_M16_N1_SNR10to0.png)
![Waveforms K4](output/waveforms_K4_M16_N1.png)

### 2. Redicosvering pulse-position modulation

The second phase focuses on discovering pulse-position modulation (PPM) waveforms that can resist multipath fading. Thus Rice fading is used to simulate multipath fading.

Also, to ensure the respect of a given bandwidth, a special penalty has been added based on used bandwidth.
- **Spectral Penalty**: FFT-based energy leakage detection for regulatory compliance.
- **PAPR Optimization**: Penalty for high Peak-to-Average Power Ratio to enable use of efficient non-linear Class-C amplifiers.

The results show that the system is able to rediscover pulse-position modulation.

```bash
uv run python main.py -K 4 -N 64 --fading-scale 0.9 --n-taps 5 --epochs 3000 --bw-penalty 2.0
```

![BER Results K4](output/results_K4_M16_N64_SNR10to-20.png)
![Waveforms K4](output/waveforms_K4_M16_N64.png)
![Spectrum K4](output/spectrum_K4_N64_BW0.5.png)


### 3. Adding Peak-to-Average Power Ratio (PAPR) penalty

As we do not want to cheat on available power, a PAPR penalty has been added.
The PAPR has forced the system to spread-out the system over time.
Also the fading penalty has forced the system to come up with several pulse.
With that, I learned a new modulation scheme that I did not know before (but seems to be well-known by the number of journals' articles): Multi-Pulse Position Modulation (MPPM)!

```bash
uv run python main.py -K 4 -N 64 --fading-scale 0.9 --n-taps 5 --epochs 3000 --bw-penalty 2.0 --papr-penalty 1.0
```

![BER Results K4](output/results_K4_M16_N64_PAPR1.0_SNR10to-20.png)
![Waveforms K4](output/waveforms_K4_M16_N64_PAPR1.0.png)
![Spectrum K4](output/spectrum_K4_N64_BW0.5_PAPR1.0.png)
![Waterfall of a 20 bytes packet](output/waterfall1_waterfall_K4_N64.png)