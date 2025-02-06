import os
import random
import json
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
    return f"{random.choice('PME')} - {random.randint(100, 999)}{random.choice('ABCD')}"

def generate_text(num_samples=100):
    for i in range(num_samples):
        text = generate_random_text()
        font_size = random.randint(20, 80)
        font_path = random.choice(available_fonts)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            print(f"Skipping incompatible font: {font_path}")
            continue

        text_width, text_height = font.getsize(text)
        padding = 10
        img_size = (text_width + 2 * padding, text_height + 2 * padding)

        alternate_white = i % 2 == 0

        if alternate_white:
            image = Image.new("RGBA", img_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            white_box_coords = (padding, padding, text_width + padding, text_height + padding)
            draw.rectangle(white_box_coords, fill=(255, 255, 255, 255))
        else:
            image = Image.new("RGBA", img_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

        text_x, text_y = padding, padding
        draw.text((text_x, text_y), text, font=font, fill="black")

        underline = random.choice([True, False])
        if underline:
            line_y = text_y + text_height + 2
            draw.line([(text_x, line_y), (text_x + text_width, line_y)], fill="black", width=2)

        angle = random.choice([0, 90])
        image = image.rotate(angle, expand=True)

        image_filename = f"synthetic_{i:06d}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image.save(image_path)

        label_data = {
            "text": text,
            "font": os.path.basename(font_path),
            "rotation": angle,
            "underlined": underline,
            "image_path": image_filename,
            "white_background": alternate_white
        }
        label_filename = f"synthetic_{i:06d}.json"
        label_path = os.path.join(LABEL_DIR, label_filename)

        with open(label_path, "w") as label_file:
            json.dump(label_data, label_file, indent=4)

        print(f"Generated {image_filename} | Text: '{text}' | Font: {os.path.basename(font_path)} | Rotation: {angle}° | Underlined: {underline} | White BG: {alternate_white}")

