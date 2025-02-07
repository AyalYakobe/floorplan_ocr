import os
from PIL import Image

class ImageProcessor:
    def __init__(self, input_dir, output_dir=None, grid_size=5, target_size=(256, 256)):
        """Initializes with input directory, optional output directory, grid size, and target resize dimensions."""
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.grid_size = grid_size
        self.target_size = target_size

        os.makedirs(self.output_dir, exist_ok=True)

    def slice_images(self):
        """Slices images into a grid and saves the slices."""
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if not image_files:
            print("No images found in the directory.")
            return

        slice_counter = 1
        for image_file in image_files:
            input_path = os.path.join(self.input_dir, image_file)
            image = Image.open(input_path)
            width, height = image.size

            slice_width = width // self.grid_size
            slice_height = height // self.grid_size

            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    left, upper = col * slice_width, row * slice_height
                    right, lower = min(left + slice_width, width), min(upper + slice_height, height)

                    sliced_image = image.crop((left, upper, right, lower))

                    slice_filename = f"slice{slice_counter}.png"
                    slice_path = os.path.join(self.output_dir, slice_filename)
                    sliced_image.save(slice_path, quality=95)

                    print(f"Saved slice: {slice_path}")
                    slice_counter += 1

        print("All images sliced successfully.")

    def resize_png_images(self):
        """Resizes PNG images in the output directory to a fixed size."""
        png_files = [f for f in os.listdir(self.output_dir) if f.lower().endswith('.png')]

        if not png_files:
            print("No PNG images found to resize.")
            return

        for image_file in png_files:
            input_path = os.path.join(self.output_dir, image_file)

            with Image.open(input_path) as image:
                image = image.convert("RGBA")

                if not isinstance(self.target_size, tuple) or len(self.target_size) != 2:
                    raise ValueError(f"Invalid target size: {self.target_size}. Expected a tuple (width, height).")

                resized_image = image.resize(self.target_size, Image.LANCZOS)
                resized_image.save(input_path, format="PNG")

                print(f"Resized and overwritten {image_file} with size {self.target_size}.")

        print("All PNG images resized and overwritten successfully.")
