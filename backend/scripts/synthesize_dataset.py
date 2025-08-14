import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm
import csv

# CONFIG
DATASET_SIZE = 20000
IMG_WIDTH, IMG_HEIGHT = 160, 50
FONTS_DIR = "./assets/fonts"  # Add .ttf/.otf fonts here
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
CHARS = string.digits + "."
MIN_LEN, MAX_LEN = 2, 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load available fonts
fonts = [
    os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")
]
if not fonts:
    raise Exception("No fonts found in assets/fonts. Add some .ttf files.")


def random_number_string():
    length = random.randint(MIN_LEN, MAX_LEN)

    # 50% chance to have a decimal point (only if length > 2)
    has_decimal = random.random() < 0.5 and length > 2

    if has_decimal:
        # decimal point position: not first or last
        decimal_pos = random.randint(1, length - 2)
        int_part_len = decimal_pos
        frac_part_len = length - decimal_pos - 1

        int_part = "".join(random.choices(string.digits, k=int_part_len))
        frac_part = "".join(random.choices(string.digits, k=frac_part_len))

        return int_part + "." + frac_part
    else:
        # Just digits only
        return "".join(random.choices(string.digits, k=length))


def add_augmentations(img):
    # Convert to OpenCV
    img = np.array(img)

    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))

    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if random.random() < 0.3:
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    if random.random() < 0.3:
        img = cv2.convertScaleAbs(
            img, alpha=random.uniform(0.7, 1.3), beta=random.randint(-20, 20)
        )

    return Image.fromarray(img)


def generate_image(text, font_path):
    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    font_size = random.randint(24, 36)
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (IMG_WIDTH - text_width) // 2
    y = (IMG_HEIGHT - text_height) // 2

    draw.text((x, y), text, font=font, fill=0)
    return img


# Generate data
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])

    for i in tqdm(range(DATASET_SIZE), desc="Generating images"):
        label = random_number_string()
        font = random.choice(fonts)
        img = generate_image(label, font)
        img = add_augmentations(img)
        filename = f"img_{i:05}.png"
        img_path = os.path.join(OUTPUT_DIR, filename)
        img.save(img_path)
        writer.writerow([filename, label])
