import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os

def load_best_fourier_params():
    if os.path.exists("best_fourier_params.json"):
        with open("best_fourier_params.json", "r") as f:
            params = json.load(f)
            return {
                "low_cutoff": int(params["low_cut"]),
                "high_cutoff": int(params["high_cut"]),
                "use_log": bool(params["logmode"]),
                "min_val": float(params["min_val"]),
                "max_val": float(params["max_val"])
            }
    else:
        return {
            "low_cutoff": 20,
            "high_cutoff": 60,
            "use_log": True,
            "min_val": 0.4,
            "max_val": 1.2
        }

def analyze_fourier(image_path):
    p = load_best_fourier_params()
    ratio = analyze_fourier_raw(
        image_path,
        low_cutoff=p["low_cutoff"],
        high_cutoff=p["high_cutoff"],
        use_log=p["use_log"]
    )
    prob_fake = (ratio - p["min_val"]) / (p["max_val"] - p["min_val"])
    prob_fake = np.clip(prob_fake, 0, 1)
    return prob_fake

def analyze_fourier_raw(image_path, low_cutoff=20, high_cutoff=60, use_log=True):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    if use_log:
        magnitude = np.log1p(magnitude)

    h, w = image.shape
    cy, cx = h // 2, w // 2

    # Считаем сумму центральных (low) и периферийных (high) частот
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    low_freq_mask = dist < low_cutoff
    high_freq_mask = (dist >= low_cutoff) & (dist <= high_cutoff)

    low_energy = np.sum(magnitude[low_freq_mask])
    high_energy = np.sum(magnitude[high_freq_mask]) + 1e-6

    ratio = low_energy / high_energy
    return ratio