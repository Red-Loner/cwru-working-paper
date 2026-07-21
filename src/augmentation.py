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

def specaugment(x, n_freq_masks=2, n_time_masks=2, max_freq_mask=8, max_time_mask=8):
    x_aug = x.copy()
    original_length = len(x_aug)
    freq_bins = 32
    time_frames = original_length // freq_bins
    spec = x_aug.reshape(freq_bins, time_frames)
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_bins)
        f0 = np.random.randint(0, max(1, freq_bins - max_freq_mask))
        mask_len = np.random.randint(1, max_freq_mask + 1)
        spec[f0:f0 + mask_len, :] = 0
    for _ in range(n_time_masks):
        t0 = np.random.randint(0, max(1, time_frames - max_time_mask))
        mask_len = np.random.randint(1, max_time_mask + 1)
        spec[:, t0:t0 + mask_len] = 0
    return spec.flatten()

def freq_axis_flip(x):
    freq_bins = 32
    time_frames = len(x) // freq_bins
    spec = x.reshape(freq_bins, time_frames)
    return np.flip(spec, axis=0).flatten()

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
