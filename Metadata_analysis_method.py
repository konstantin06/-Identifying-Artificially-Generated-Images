from PIL import Image
import piexif
import os
import numpy as np
import json


def load_best_metadata_weights():
    if os.path.exists("best_metadata_weights.json"):
        with open("best_metadata_weights.json", "r") as f:
            return json.load(f)
    else:
        return {
            "has_exif": 0.5,
            "has_camera_info": 0.2,
            "suspicious_software": 0.3,
            "has_icc": 0.05,
            "low_dpi": 0.1,
            "weird_size": 0.1,
            "suspicious_quant": 0.2
        }

def analyze_metadata(image_path):
    weights = load_best_metadata_weights()
    features = analyze_metadata_features(image_path)
    prob_fake = sum(features[k] * weights.get(k, 0) for k in features)
    return min(prob_fake, 1.0)

def analyze_metadata_features(image_path):
    result = {
        "has_exif": 0,
        "has_camera_info": 0,
        "suspicious_software": 0,
        "low_dpi": 0,
        "weird_size": 0,
        "has_icc": 0,
        "suspicious_quant": 0
    }

    try:
        image = Image.open(image_path)
        width, height = image.size
        if width % 8 != 0 or height % 8 != 0:
            result["weird_size"] = 1

        result["has_icc"] = int("icc_profile" in image.info)

        dpi = image.info.get("dpi", (0, 0))[0]
        if dpi < 100:
            result["low_dpi"] = 1

        if "exif" in image.info:
            result["has_exif"] = 1
            try:
                exif_data = piexif.load(image.info["exif"])
                for ifd in exif_data:
                    if isinstance(exif_data[ifd], dict):
                        for tag, value in exif_data[ifd].items():
                            tag_name = piexif.TAGS[ifd].get(tag, {}).get("name", "").lower()
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode(errors="ignore")
                                except:
                                    continue
                            if tag_name == "software":
                                if any(k in str(value).lower() for k in ["ai", "diffusion", "gan", "midjourney", "dalle"]):
                                    result["suspicious_software"] = 1
                            if tag_name in ["make", "model"]:
                                result["has_camera_info"] = 1
            except:
                pass

        try:
            tables = image.quantization
            for table in tables.values():
                if len(set(table)) < 10:
                    result["suspicious_quant"] = 1
        except:
            pass

    except:
        pass

    return result
