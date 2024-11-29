from snr import calculate_snr
import numpy as np

def test_calculate_snr():
    original = np.ones((5, 5))
    noisy = original + np.random.normal(0, 0.1, (5, 5))
    snr = calculate_snr(original, noisy)
    print(f"Test du SNR : {snr:.2f} dB")
