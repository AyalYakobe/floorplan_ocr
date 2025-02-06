import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm

TEXT_IMAGES_DIR = "Synthetic_Dataset/images"
BACKGROUND_IMAGES_DIR = "Clippings/Sliced"
OUTPUT_DIR = "Synthetic_Dataset/final_images"

GREY_COLOR = (128, 128, 128, 255)  # Grey color for filling transparent areas
BOUNDING_SHAPES = ["rectangle", "circle", None]  # Include None to allow no shape
BOUNDING_SHAPE_PROBABILITY = 0.6  # 60% chance of having a bounding shape

MAX_BOUNDARY_THICKNESS = 5  # Maximum boundary thickness
MIN_BOUNDARY_THICKNESS = 2  # Minimum boundary thickness
EXTRA_PADDING = 250  # Significantly more padding for the shape

def fill_transparent_space(image, fill_color):
    """Fills any transparent areas in the image with the specified fill color."""
    background = Image.new("RGBA", image.size, fill_color)
    background.paste(image, (0, 0), image)
    return background

def resize_text_image_to_background(text_image, bg_width, bg_height):
    """Resize text image to ensure it occupies between 20% and 80% of the smaller background dimension."""
    smaller_bg_dimension = min(bg_width, bg_height)

    # Define minimum and maximum text dimensions
    min_text_size = smaller_bg_dimension * 0.35
    max_text_size = smaller_bg_dimension * 0.8

    # Scale factors
    scale_factor_width = max_text_size / text_image.width
    scale_factor_height = max_text_size / text_image.height
    scale_factor_min_width = min_text_size / text_image.width
    scale_factor_min_height = min_text_size / text_image.height

    # Final scale factor
    scale_factor = max(
        min(scale_factor_width, scale_factor_height),  # Ensure it fits within max size
        max(scale_factor_min_width, scale_factor_min_height)  # Ensure it exceeds min size
    )

    # Resize the text image
    new_text_size = (int(text_image.width * scale_factor), int(text_image.height * scale_factor))
    resized_image = text_image.resize(new_text_size, Image.LANCZOS)

    return resized_image

def add_bounding_shape(image, background_width, background_height):
    """Adds a significantly larger bounding rectangle or circle around the text."""
    if random.random() < BOUNDING_SHAPE_PROBABILITY:
        bounding_shape = random.choice(["rectangle", "circle"])
        width, height = image.size

        # Generous padding
        padding_x = max(width * 0.5, 450)  # At least 50% text width or 300 pixels
        padding_y = max(height * 0.5, 450)  # At least 50% text height or 300 pixels
        boundary_thickness = random.randint(MIN_BOUNDARY_THICKNESS, MAX_BOUNDARY_THICKNESS)

        # Create a larger canvas
        new_size = (int(width + 2 * padding_x), int(height + 2 * padding_y))
        shape_overlay = Image.new("RGBA", new_size, (0, 0, 0, 0))  # Transparent background
        draw = ImageDraw.Draw(shape_overlay)

        # Define the bounding box for the shape
        shape_bbox = (
            int(padding_x),
            int(padding_y),
            int(new_size[0] - padding_x),
            int(new_size[1] - padding_y),
        )

        # Draw the bounding shape
        if bounding_shape == "rectangle":
            draw.rectangle(shape_bbox, outline="black", width=boundary_thickness)
        elif bounding_shape == "circle":
            draw.ellipse(shape_bbox, outline="black", width=boundary_thickness)

        # Place the text image at the center
        text_x = int((new_size[0] - image.width) / 2)
        text_y = int((new_size[1] - image.height) / 2)
        shape_overlay.paste(image, (text_x, text_y), image)

        # Ensure the new image fits the background
        scale_factor = min(
            background_width / new_size[0],
            background_height / new_size[1],
        )
        if scale_factor < 1:
            new_width = int(new_size[0] * scale_factor)
            new_height = int(new_size[1] * scale_factor)
            shape_overlay = shape_overlay.resize((new_width, new_height), Image.LANCZOS)

        return shape_overlay, bounding_shape
    else:
        # If no shape is added, return the original image and None
        return image, None

def apply_backgrounds_to_text_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text_images = [os.path.join(TEXT_IMAGES_DIR, f) for f in os.listdir(TEXT_IMAGES_DIR) if f.endswith(".png")]
    background_images = [os.path.join(BACKGROUND_IMAGES_DIR, f) for f in os.listdir(BACKGROUND_IMAGES_DIR) if f.endswith(".png")]

    if not text_images:
        raise ValueError("No text images found.")
    if not background_images:
        raise ValueError("No background images found.")

    for text_image_path in tqdm(text_images, desc="Applying Backgrounds to Text Images"):
        text_image = Image.open(text_image_path).convert("RGBA")
        background_path = random.choice(background_images)
        background = Image.open(background_path).convert("RGBA")

        # Get the dimensions of the background
        bg_width, bg_height = background.size

        # Resize the text image
        text_image = resize_text_image_to_background(text_image, bg_width, bg_height)

        # Apply bounding shape
        text_image, bounding_shape = add_bounding_shape(text_image, bg_width, bg_height)

        # Center the text and shape on the background
        max_x = bg_width - text_image.width
        max_y = bg_height - text_image.height
        random_x = random.randint(0, max_x) if max_x > 0 else 0
        random_y = random.randint(0, max_y) if max_y > 0 else 0

        final_image = background.copy()
        final_image.paste(text_image, (random_x, random_y), text_image)

        # Fill any remaining transparent areas with grey
        final_image = fill_transparent_space(final_image, GREY_COLOR)

        output_path = os.path.join(OUTPUT_DIR, os.path.basename(text_image_path))
        final_image.save(output_path)

