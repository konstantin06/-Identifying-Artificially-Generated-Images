import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def load_best_artifact_params():
    if os.path.exists("best_artifacts_params.json"):
        with open("best_artifacts_params.json", "r") as f:
            p = json.load(f)
            return {
                "blur_kernel": int(p["blur_kernel"]),
                "min_val": float(p["min_val"]),
                "max_val": float(p["max_val"])
            }
    else:
        return {
            "blur_kernel": 3,
            "min_val": 4.0,
            "max_val": 10.0
        }

def analyze_generation_artifacts(image_path, show=False):
    p = load_best_artifact_params()
    val = analyze_artifacts_raw(image_path, blur_kernel=p["blur_kernel"])
    prob_fake = (val - p["min_val"]) / (p["max_val"] - p["min_val"])
    prob_fake = np.clip(prob_fake, 0, 1)

    if show:
        print(f"mean_laplacian: {val:.2f} → prob_fake = {prob_fake:.2f}")
    return prob_fake

def analyze_artifacts_raw(image_path, blur_kernel=0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if blur_kernel > 0:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    abs_lap = np.abs(laplacian)
    mean_val = np.mean(abs_lap)
    return mean_val
