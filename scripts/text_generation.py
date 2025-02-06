import os
import random
import json
import string

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/System/Library/Fonts/Supplemental/"
ACCEPTABLE_FONTS_FILE = "synthetic_dataset/acceptable_fonts.txt"
OUTPUT_DIR = "synthetic_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

with open(ACCEPTABLE_FONTS_FILE, "r") as f:
    acceptable_fonts = {line.strip() for line in f.readlines()}

available_fonts = [
    os.path.join(FONT_DIR, font) for font in acceptable_fonts if font.endswith(('.ttf', '.otf'))
]

if not available_fonts:
    raise ValueError("No acceptable fonts found. Check your acceptable_fonts.txt file.")

print(f"Loaded {len(available_fonts)} fonts from {ACCEPTABLE_FONTS_FILE}.")


def generate_random_text():
    length = random.randint(1, 10)  # Random length between 1 and 10
    chars = string.ascii_uppercase + string.digits  # Alphanumeric characters
    text = ''.join(random.choices(chars, k=length))  # Generate random alphanumeric string

    # Decide if we add hyphens (max 2)
    if length > 2 and random.choice([True, False]):
        hyphen_positions = sorted(random.sample(range(1, length), k=random.randint(1, 2)))
        for pos in reversed(hyphen_positions):
            text = text[:pos] + '-' + text[pos:]

    return text


def generate_text(num_samples=100):
    for i in range(num_samples):
        stacked_text = random.random() < 0.10  # 10% probability for stacked text
        num_lines = random.randint(2, 4) if stacked_text else 1  # Stack 2-4 texts if stacking

        texts = [generate_random_text() for _ in range(num_lines)]
        font_size = random.randint(60, 80)
        font_path = random.choice(available_fonts)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            print(f"Skipping incompatible font: {font_path}")
            continue

        text_sizes = [font.getsize(text) for text in texts]
        max_text_width = max(w for w, h in text_sizes)
        total_text_height = sum(h for w, h in text_sizes) + (10 * (len(texts) - 1))  # Adding spacing

        padding = 10
        img_size = (max_text_width + 2 * padding, total_text_height + 2 * padding)

        alternate_white = i % 2 == 0

        if alternate_white:
            image = Image.new("RGBA", img_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            white_box_coords = (padding, padding, max_text_width + padding, total_text_height + padding)
            draw.rectangle(white_box_coords, fill=(255, 255, 255, 255))
        else:
            image = Image.new("RGBA", img_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

        text_y = padding
        label_data = []

        for text in texts:
            text_width, text_height = font.getsize(text)
            text_x = padding
            draw.text((text_x, text_y), text, font=font, fill="black")

            underline = random.choice([True, False])
            if underline:
                line_y = text_y + text_height + 2
                draw.line([(text_x, line_y), (text_x + text_width, line_y)], fill="black", width=2)

            label_data.append({
                "text": text,
                "position": (text_x, text_y),
                "underlined": underline
            })

            text_y += text_height + 10  # Move down for the next text

        angle = random.choice([0, 90])
        image = image.rotate(angle, expand=True)

        image_filename = f"synthetic_{i:06d}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image.save(image_path)

        final_label_data = {
            "texts": label_data,
            "font": os.path.basename(font_path),
            "rotation": angle,
            "image_path": image_filename,
            "white_background": alternate_white
        }
        label_filename = f"synthetic_{i:06d}.json"
        label_path = os.path.join(LABEL_DIR, label_filename)

        with open(label_path, "w") as label_file:
            json.dump(final_label_data, label_file, indent=4)

        print(
            f"Generated {image_filename} | {len(texts)} Text(s) | Font: {os.path.basename(font_path)} | Rotation: {angle}° | White BG: {alternate_white}")


