import os
import random
from PIL import Image, ImageDraw

TEXT_IMAGES_DIR = "synthetic_dataset/images"
BACKGROUND_IMAGES_DIR = "Clippings/Sliced"
OUTPUT_DIR = "synthetic_dataset/final_images"

GREY_COLOR = (128, 128, 128, 255)  # Grey color for filling transparent areas
BOUNDING_SHAPES = ["rectangle", "circle", None]  # Include None to allow no shape
BOUNDING_SHAPE_PROBABILITY = 0.6  # 60% chance of having a bounding shape
ARROW_PROBABILITY = 0.5  # 50% chance of adding an arrow/line if there is a shape

MAX_BOUNDARY_THICKNESS = 5  # Maximum boundary thickness
MIN_BOUNDARY_THICKNESS = 2  # Minimum boundary thickness
EXTRA_PADDING = 150  # Significantly more padding for the shape

def fill_transparent_space(image, fill_color):
    """Fills any transparent areas in the image with the specified fill color."""
    background = Image.new("RGBA", image.size, fill_color)
    background.paste(image, (0, 0), image)
    return background

def add_bounding_shape(image, background_width, background_height):
    """Adds a larger bounding rectangle or circle around the text with proper padding."""
    if random.random() > BOUNDING_SHAPE_PROBABILITY:
        return image, None  # Skip adding a bounding shape

    bounding_shape = random.choice(["rectangle", "circle"])
    width, height = image.size
    padding = EXTRA_PADDING  # Use large padding to ensure the text is well-separated
    boundary_thickness = random.randint(MIN_BOUNDARY_THICKNESS, MAX_BOUNDARY_THICKNESS)

    # Create a new canvas with larger dimensions to accommodate the padding
    new_size = (width + 2 * padding, height + 2 * padding)
    shape_overlay = Image.new("RGBA", new_size, (0, 0, 0, 0))  # Transparent overlay
    draw = ImageDraw.Draw(shape_overlay)

    # Define the bounding box for the shape
    shape_bbox = (padding, padding, new_size[0] - padding, new_size[1] - padding)

    if bounding_shape == "rectangle":
        draw.rectangle(shape_bbox, outline="black", width=boundary_thickness)
    elif bounding_shape == "circle":
        draw.ellipse(shape_bbox, outline="black", width=boundary_thickness)

    # Paste the text image at the center of the padded area
    shape_overlay.paste(image, (padding, padding), image)

    # Resize if the new image with the shape exceeds the background size
    new_width, new_height = shape_overlay.size
    if new_width > background_width or new_height > background_height:
        scale_factor = min(background_width / new_width, background_height / new_height)
        new_size = (int(new_width * scale_factor), int(new_height * scale_factor))
        shape_overlay = shape_overlay.resize(new_size, Image.LANCZOS)

    return shape_overlay, bounding_shape


def add_random_arrow_or_line(draw, canvas_size, shape_bbox):
    """Draws a line or arrow extending outward from the bounding shape."""
    x1, y1, x2, y2 = shape_bbox
    shape_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Choose a starting point from the shape's edge
    edge_positions = [
        (shape_center[0], y1),  # Top center
        (shape_center[0], y2),  # Bottom center
        (x1, shape_center[1]),  # Left center
        (x2, shape_center[1])   # Right center
    ]
    start_point = random.choice(edge_positions)

    # Choose a random direction outward from the shape
    arrow_length = random.randint(50, 100)
    if start_point == (shape_center[0], y1):  # Top edge
        end_point = (start_point[0], start_point[1] - arrow_length)
    elif start_point == (shape_center[0], y2):  # Bottom edge
        end_point = (start_point[0], start_point[1] + arrow_length)
    elif start_point == (x1, shape_center[1]):  # Left edge
        end_point = (start_point[0] - arrow_length, start_point[1])
    else:  # Right edge
        end_point = (start_point[0] + arrow_length, start_point[1])

    # Ensure arrow stays within the image bounds
    end_point = (
        max(0, min(canvas_size[0], end_point[0])),
        max(0, min(canvas_size[1], end_point[1]))
    )

    # Draw either a line or an arrow
    if random.random() < 0.5:
        draw.line([start_point, end_point], fill="black", width=5)
    else:
        draw.line([start_point, end_point], fill="black", width=5)
        arrowhead_size = 10
        draw.polygon([
            (end_point[0] - arrowhead_size, end_point[1] - arrowhead_size),
            (end_point[0] + arrowhead_size, end_point[1] - arrowhead_size),
            (end_point[0], end_point[1] + arrowhead_size)
        ], fill="black")

def apply_backgrounds_to_text_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text_images = [os.path.join(TEXT_IMAGES_DIR, f) for f in os.listdir(TEXT_IMAGES_DIR) if f.endswith(".png")]
    background_images = [os.path.join(BACKGROUND_IMAGES_DIR, f) for f in os.listdir(BACKGROUND_IMAGES_DIR) if f.endswith(".png")]

    if not text_images:
        raise ValueError("No text images found.")
    if not background_images:
        raise ValueError("No background images found.")

    for text_image_path in text_images:
        text_image = Image.open(text_image_path).convert("RGBA")
        background_path = random.choice(background_images)
        background = Image.open(background_path).convert("RGBA")

        scale_factor = random.uniform(0.4, 0.4)  # Smaller text to fit inside
        smaller_text_size = (int(text_image.width * scale_factor), int(text_image.height * scale_factor))
        text_image = text_image.resize(smaller_text_size, Image.LANCZOS)

        # Apply bounding shape randomly
        text_image, bounding_shape = add_bounding_shape(text_image, background.width, background.height)

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

        print(f"Created: {output_path} | Background: {os.path.basename(background_path)} | Text Scale: {scale_factor:.2f} | Position: ({random_x}, {random_y}) | Shape: {bounding_shape or 'None'}")

