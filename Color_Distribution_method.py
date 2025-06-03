import cv2
import numpy as np
from scipy.stats import entropy
import json
import os


def load_best_color_params():
    if os.path.exists("best_color_params.json"):
        with open("best_color_params.json", "r") as f:
            return json.load(f)
    else:
        return {
            "entropy_min": 3.5, "entropy_max": 6.5,
            "std_min": 20.0, "std_max": 70.0
        }

def analyze_color_distribution(image_path):
    p = load_best_color_params()
    entropy_val, std_cbcr = analyze_color_raw(image_path)

    e_score = (entropy_val - p["entropy_min"]) / (p["entropy_max"] - p["entropy_min"])
    s_score = 1 - (std_cbcr - p["std_min"]) / (p["std_max"] - p["std_min"])

    prob_fake = 0.5 * np.clip(e_score, 0, 1) + 0.5 * np.clip(s_score, 0, 1)
    return np.clip(prob_fake, 0, 1)

def analyze_color_raw(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot open image: {image_path}")

    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    entropies = []
    stds = []

    for i in range(3):  # Y, Cr, Cb
        channel = image_ycrcb[:, :, i]
        hist = cv2.calcHist([image_ycrcb], [i], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-8)
        entropies.append(entropy(hist, base=2))

        stds.append(np.std(channel))

    mean_entropy = np.mean(entropies)
    std_cbcr = np.mean([stds[1], stds[2]])  # Cr и Cb

    return mean_entropy, std_cbcr
