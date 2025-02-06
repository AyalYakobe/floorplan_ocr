import os
import random
from PIL import Image, ImageChops, ImageDraw

TEXT_IMAGES_DIR = "synthetic_dataset/images"
BACKGROUND_IMAGES_DIR = "Clippings/Sliced"
OUTPUT_DIR = "synthetic_dataset/final_images"

GREY_COLOR = (128, 128, 128, 255)  # Grey color for filling transparent areas

def fill_transparent_space(image, fill_color):
    background = Image.new("RGBA", image.size, fill_color)
    background.paste(image, (0, 0), image)
    return background

def apply_backgrounds_to_text_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text_images = [os.path.join(TEXT_IMAGES_DIR, f) for f in os.listdir(TEXT_IMAGES_DIR) if f.endswith(".png")]
    background_images = [os.path.join(BACKGROUND_IMAGES_DIR, f) for f in os.listdir(BACKGROUND_IMAGES_DIR) if f.endswith(".png")]

    if not text_images:
        raise ValueError("No text images found.")
    if not background_images:
        raise ValueError("No background images found.")

    for i, text_image_path in enumerate(text_images):
        text_image = Image.open(text_image_path).convert("RGBA")
        background_path = random.choice(background_images)
        background = Image.open(background_path).convert("RGBA")

        scale_factor = random.uniform(0.2, 0.4)  # Smaller text to fit inside
        smaller_text_size = (int(text_image.width * scale_factor), int(text_image.height * scale_factor))
        text_image = text_image.resize(smaller_text_size, Image.LANCZOS)

        max_x = background.width - text_image.width
        max_y = background.height - text_image.height

        random_x = random.randint(0, max_x) if max_x > 0 else 0
        random_y = random.randint(0, max_y) if max_y > 0 else 0

        final_image = background.copy()
        final_image.paste(text_image, (random_x, random_y), text_image)

        # Fill any remaining transparent areas with grey
        final_image = fill_transparent_space(final_image, GREY_COLOR)

        output_path = os.path.join(OUTPUT_DIR, os.path.basename(text_image_path))
        final_image.save(output_path)

        print(f"Created: {output_path} | Background: {os.path.basename(background_path)} | Text Scale: {scale_factor:.2f} | Position: ({random_x}, {random_y})")