import numpy as np

def gaussian_noise(x, sigma):
    noise = np.random.randn(*x.shape).astype(np.float32) * sigma
    return x + noise

def time_shift(x, shift_samples):
    shift = np.random.randint(-shift_samples, shift_samples + 1)
    if shift > 0:
        return np.concatenate([np.zeros(shift, dtype=np.float32), x[:-shift]])
    elif shift < 0:
        return np.concatenate([x[-shift:], np.zeros(-shift, dtype=np.float32)])
    return x.copy()

def specaugment(
    x,
    n_freq_masks=2,
    n_time_masks=2,
    max_freq_mask=8,
    max_time_mask=64,
):
    """Waveform-compatible spectral and temporal masking.

    Frequency masks are applied to real-FFT bins and transformed back to the
    time domain. Temporal masks then zero contiguous sample ranges. This is an
    adaptation of the SpecAugment idea for a common waveform input, not a
    reshape of time samples masquerading as a spectrogram.
    """
    x_aug = np.asarray(x, dtype=np.float32).copy()
    spectrum = np.fft.rfft(x_aug)
    freq_bins = len(spectrum)
    for _ in range(n_freq_masks):
        mask_len = np.random.randint(1, min(max_freq_mask, freq_bins - 1) + 1)
        start_max = freq_bins - mask_len
        f0 = np.random.randint(1, start_max + 1)
        spectrum[f0:f0 + mask_len] = 0
    x_aug = np.fft.irfft(spectrum, n=len(x_aug)).astype(np.float32)

    for _ in range(n_time_masks):
        mask_len = np.random.randint(1, min(max_time_mask, len(x_aug)) + 1)
        t0 = np.random.randint(0, len(x_aug) - mask_len + 1)
        x_aug[t0:t0 + mask_len] = 0
    return x_aug

def freq_axis_flip(x):
    """Negative control that reverses FFT magnitudes across frequency bins."""
    x = np.asarray(x, dtype=np.float32)
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)[::-1]
    phase = np.angle(spectrum)
    flipped = magnitude * np.exp(1j * phase)
    flipped[0] = flipped[0].real
    if len(x) % 2 == 0:
        flipped[-1] = flipped[-1].real
    return np.fft.irfft(flipped, n=len(x)).astype(np.float32)

def apply_augmentation(x_batch, aug_type, aug_params=None):
    if aug_type == "none":
        return x_batch.copy()
    if aug_params is None:
        aug_params = {}
    out = []
    for x in x_batch:
        if aug_type == "noise_001":
            out.append(gaussian_noise(x, sigma=0.01))
        elif aug_type == "noise_005":
            out.append(gaussian_noise(x, sigma=0.05))
        elif aug_type == "noise_01":
            out.append(gaussian_noise(x, sigma=0.1))
        elif aug_type == "shift_5":
            out.append(time_shift(x, shift_samples=5))
        elif aug_type == "shift_20":
            out.append(time_shift(x, shift_samples=20))
        elif aug_type == "shift_50":
            out.append(time_shift(x, shift_samples=50))
        elif aug_type == "specaugment":
            out.append(specaugment(x))
        elif aug_type == "combined":
            x1 = gaussian_noise(x, sigma=0.05)
            x2 = time_shift(x1, shift_samples=20)
            out.append(x2)
        elif aug_type == "freq_flip":
            out.append(freq_axis_flip(x))
        else:
            out.append(x.copy())
    return np.array(out, dtype=np.float32)

AUGMENTATION_CONDITIONS = [
    "none",
    "noise_001", "noise_005", "noise_01",
    "shift_5", "shift_20", "shift_50",
    "specaugment",
    "combined",
    "freq_flip",
]
