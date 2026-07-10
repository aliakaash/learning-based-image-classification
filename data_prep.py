"""
data_prep.py
Generates a synthetic, fully-offline image dataset (geometric shapes) using OpenCV
so the whole pipeline can run without downloading anything from the internet.

If you have your own real dataset, skip this script entirely and just place
your images at:
    data/train/<class_name>/*.jpg
    data/test/<class_name>/*.jpg
The rest of the pipeline (preprocessing.py, models.py, train.py, app.py) will
work unchanged as long as config.CLASS_NAMES matches your folder names.

Run:
    python data_prep.py
"""

import os
import random
import numpy as np
import cv2

from config import (
    TRAIN_DIR, TEST_DIR, CLASS_NAMES, IMG_SIZE,
    IMAGES_PER_CLASS_TRAIN, IMAGES_PER_CLASS_TEST, RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _random_color():
    return (
        random.randint(40, 255),
        random.randint(40, 255),
        random.randint(40, 255),
    )


def _blank_canvas():
    h, w = IMG_SIZE
    # random-ish background so the model can't cheat on background color alone
    bg = random.randint(0, 40)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    # light noise for realism
    noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
    canvas = cv2.add(canvas, noise)
    return canvas


def _draw_circle(canvas):
    h, w = IMG_SIZE
    r = random.randint(int(0.2 * min(h, w)), int(0.4 * min(h, w)))
    cx = random.randint(r, w - r)
    cy = random.randint(r, h - r)
    cv2.circle(canvas, (cx, cy), r, _random_color(), thickness=-1)
    return canvas


def _draw_square(canvas):
    h, w = IMG_SIZE
    side = random.randint(int(0.3 * min(h, w)), int(0.6 * min(h, w)))
    x1 = random.randint(0, w - side)
    y1 = random.randint(0, h - side)
    angle = random.uniform(-25, 25)
    rect = ((x1 + side / 2, y1 + side / 2), (side, side), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], _random_color())
    return canvas


def _draw_triangle(canvas):
    h, w = IMG_SIZE
    size = random.randint(int(0.3 * min(h, w)), int(0.45 * min(h, w)))
    half = size // 2 + 2
    cx = random.randint(half, max(half, w - half))
    cy = random.randint(half, max(half, h - half))
    pts = np.array([
        [cx, cy - size // 2],
        [cx - size // 2, cy + size // 2],
        [cx + size // 2, cy + size // 2],
    ])
    cv2.fillPoly(canvas, [pts], _random_color())
    return canvas


def _draw_star(canvas):
    h, w = IMG_SIZE
    cx, cy = w // 2 + random.randint(-8, 8), h // 2 + random.randint(-8, 8)
    outer_r = random.randint(int(0.25 * min(h, w)), int(0.4 * min(h, w)))
    inner_r = outer_r * 0.4
    pts = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(canvas, [pts], _random_color())
    return canvas


_DRAW_FN = {
    "circle": _draw_circle,
    "square": _draw_square,
    "triangle": _draw_triangle,
    "star": _draw_star,
}


def generate_split(out_dir, n_per_class):
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        draw_fn = _DRAW_FN.get(class_name)
        if draw_fn is None:
            raise ValueError(
                f"No synthetic generator for class '{class_name}'. "
                f"Either update CLASS_NAMES in config.py to one of "
                f"{list(_DRAW_FN.keys())}, or supply your own real dataset."
            )
        for i in range(n_per_class):
            canvas = _blank_canvas()
            img = draw_fn(canvas)
            # small random blur to mimic real-world noise
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            path = os.path.join(class_dir, f"{class_name}_{i:04d}.jpg")
            cv2.imwrite(path, img)
        print(f"  {class_name}: {n_per_class} images -> {class_dir}")


def dataset_already_present():
    if not os.path.isdir(TRAIN_DIR):
        return False
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir) or len(os.listdir(class_dir)) == 0:
            return False
    return True


if __name__ == "__main__":
    if dataset_already_present():
        print("Dataset already present in data/train — skipping generation.")
        print("Delete the data/ folder if you want to regenerate the synthetic set.")
    else:
        print("Generating synthetic training set...")
        generate_split(TRAIN_DIR, IMAGES_PER_CLASS_TRAIN)
        print("Generating synthetic test set...")
        generate_split(TEST_DIR, IMAGES_PER_CLASS_TEST)
        print("Done. Dataset ready at data/train and data/test.")
