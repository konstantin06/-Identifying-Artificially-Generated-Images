import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_ubyte
from skimage.morphology import disk
from skimage.measure import label, regionprops
import json
import os

def load_best_entropy_params():
    path = "best_entropy_params.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        # fallback параметры по умолчанию
        return {
            "tolerance": 0.15,
            "radius": 7,
            "min_size": 40,
            "max_size": 250
        }

def analyze_entropy_complexity(image_path, tolerance=None, radius=None, min_size=None, max_size=None, show_images=False):
    from skimage import io, filters, img_as_ubyte
    from skimage.morphology import disk
    from skimage.measure import label, regionprops
    import numpy as np

    # === Загружаем лучшие параметры, если они не переданы вручную
    if any(param is None for param in [tolerance, radius, min_size, max_size]):
        best = load_best_entropy_params()
        tolerance = best["tolerance"]
        radius = best["radius"]
        min_size = best["min_size"]
        max_size = best["max_size"]

    # === Обработка изображения
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    red = img_as_ubyte(image[:, :, 0])
    green = img_as_ubyte(image[:, :, 1])
    blue = img_as_ubyte(image[:, :, 2])

    selem = disk(radius)
    entropy_r = filters.rank.entropy(red, selem)
    entropy_g = filters.rank.entropy(green, selem)
    entropy_b = filters.rank.entropy(blue, selem)

    diff_rg = np.abs(entropy_r - entropy_g)
    diff_rb = np.abs(entropy_r - entropy_b)
    diff_gb = np.abs(entropy_g - entropy_b)

    mask = (diff_rg < tolerance) & (diff_rb < tolerance) & (diff_gb < tolerance)
    labeled = label(mask)
    regions = regionprops(labeled)
    cluster_sizes = [r.area for r in regions]
    mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

    # === Переводим в prob_fake с обученной шкалой
    prob_fake = 1 - min(max((mean_cluster_size - min_size) / (max_size - min_size), 0), 1)

    if show_images:
        import matplotlib.pyplot as plt
        result_image = image.copy()
        result_image[mask] = [255, 0, 0]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[1].imshow(result_image)
        ax[1].set_title("Matched Entropy Regions (Red)")
        plt.tight_layout()
        plt.show()

    return prob_fake


def analyze_entropy_complexity_raw(image_path, radius=7, tolerance=0.15):
    from skimage import io, filters, img_as_ubyte
    from skimage.morphology import disk
    from skimage.measure import label, regionprops
    import numpy as np

    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    red = img_as_ubyte(image[:, :, 0])
    green = img_as_ubyte(image[:, :, 1])
    blue = img_as_ubyte(image[:, :, 2])

    selem = disk(radius)
    entropy_r = filters.rank.entropy(red, selem)
    entropy_g = filters.rank.entropy(green, selem)
    entropy_b = filters.rank.entropy(blue, selem)

    diff_rg = np.abs(entropy_r - entropy_g)
    diff_rb = np.abs(entropy_r - entropy_b)
    diff_gb = np.abs(entropy_g - entropy_b)

    mask = (diff_rg < tolerance) & (diff_rb < tolerance) & (diff_gb < tolerance)
    labeled = label(mask)
    regions = regionprops(labeled)
    cluster_sizes = [r.area for r in regions]
    mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

    return mean_cluster_size

