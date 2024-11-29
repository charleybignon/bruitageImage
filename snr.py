import numpy as np

def calculate_snr(original, noisy):
    if original.max() > 1.0:
        original = original / 255.0
    if noisy.max() > 1.0:
        noisy = noisy / 255.0

    noise = noisy - original
    P_signal = np.mean(noisy ** 2)
    P_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(P_signal / P_noise)
    return snr

def calculate_snr_gain(original_snr, denoised_snr):
    return denoised_snr - original_snr
