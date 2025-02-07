import os
import random
import time
from PIL import Image, ImageDraw
from tqdm import tqdm

class TextImageProcessor:
    def __init__(self, text_images_dir, background_images_dir, output_dir):
        self.text_images_dir = text_images_dir
        self.background_images_dir = background_images_dir
        self.output_dir = output_dir

        self.GREY_COLOR = (128, 128, 128, 255)  # Grey color for filling transparent areas
        self.BOUNDING_SHAPES = ["rectangle", "circle", None]  # Include None to allow no shape
        self.BOUNDING_SHAPE_PROBABILITY = 0.6  # 60% chance of having a bounding shape

        self.MAX_BOUNDARY_THICKNESS = 5  # Maximum boundary thickness
        self.MIN_BOUNDARY_THICKNESS = 2  # Minimum boundary thickness
        self.EXTRA_PADDING = 250  # Extra padding for the bounding shape

    def fill_transparent_space(self, image):
        """Fills any transparent areas in the image with the specified fill color."""
        background = Image.new("RGBA", image.size, self.GREY_COLOR)
        background.paste(image, (0, 0), image)
        return background

    def resize_text_image_to_background(self, text_image, bg_width, bg_height):
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
        return text_image.resize(new_text_size, Image.LANCZOS)

    def add_bounding_shape(self, image, background_width, background_height):
        """Adds a significantly larger bounding rectangle or circle around the text."""
        if random.random() < self.BOUNDING_SHAPE_PROBABILITY:
            bounding_shape = random.choice(["rectangle", "circle"])
            width, height = image.size

            # Generous padding
            padding_x = max(width * 0.5, 450)  # At least 50% text width or 450 pixels
            padding_y = max(height * 0.5, 450)  # At least 50% text height or 450 pixels
            boundary_thickness = random.randint(self.MIN_BOUNDARY_THICKNESS, self.MAX_BOUNDARY_THICKNESS)

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
            return image, None

    def apply_backgrounds_to_text_images(self):
        os.makedirs(self.output_dir, exist_ok=True)

        text_images = [os.path.join(self.text_images_dir, f) for f in os.listdir(self.text_images_dir) if f.endswith(".png")]
        background_images = [os.path.join(self.background_images_dir, f) for f in os.listdir(self.background_images_dir) if f.endswith(".png")]

        if not text_images:
            raise ValueError("No text images found.")
        if not background_images:
            raise ValueError("No background images found.")

        for text_image_path in tqdm(text_images, desc="Applying Backgrounds to Text Images"):
            text_image = Image.open(text_image_path).convert("RGBA")
            background_path = random.choice(background_images)
            background = Image.open(background_path).convert("RGBA")

            # Fit dimensions and shapes
            bg_width, bg_height = background.size
            text_image = self.resize_text_image_to_background(text_image, bg_width, bg_height)
            text_image, bounding_shape = self.add_bounding_shape(text_image, bg_width, bg_height)

            # Center the text and shape on the background
            max_x = bg_width - text_image.width
            max_y = bg_height - text_image.height
            random_x = random.randint(0, max_x) if max_x > 0 else 0
            random_y = random.randint(0, max_y) if max_y > 0 else 0

            final_image = background.copy()
            final_image.paste(text_image, (random_x, random_y), text_image)
            final_image = self.fill_transparent_space(final_image)

            output_path = os.path.join(self.output_dir, os.path.basename(text_image_path))
            final_image.save(output_path)

    def estimate_execution_time(self, num_samples):
        start_time = time.time()
        self.apply_backgrounds_to_text_images()
        end_time = time.time()

        elapsed_time_minutes = (end_time - start_time) / 60
        print(f"Execution Time for {num_samples} samples: {elapsed_time_minutes:.2f} minutes")

        estimated_time_500k = (elapsed_time_minutes / num_samples) * 500000
        print(f"Estimated Execution Time for 500,000 samples: {estimated_time_500k:.2f} minutes")


