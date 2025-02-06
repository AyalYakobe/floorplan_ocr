import os
from PIL import Image

def slice_images_in_directory(input_dir, output_dir, grid_size=5):
    """Slices all images in input_dir into a grid and saves them in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No images found in the directory.")
        return

    slice_counter = 1  # Counter for naming slices sequentially
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        image = Image.open(input_path)
        width, height = image.size

        slice_width = width // grid_size
        slice_height = height // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                left = col * slice_width
                upper = row * slice_height
                right = min(left + slice_width, width)
                lower = min(upper + slice_height, height)

                sliced_image = image.crop((left, upper, right, lower))

                # Use a shorter naming convention: slice1.png, slice2.png, etc.
                slice_filename = f"slice{slice_counter}.png"
                slice_path = os.path.join(output_dir, slice_filename)
                sliced_image.save(slice_path, quality=95)

                print(f"Saved slice: {slice_path}")
                slice_counter += 1

    print("All images sliced successfully.")

def resize_png_images_uniform(input_dir, target_size=(256, 256)):
    """Resizes all PNG images in input_dir to a fixed size (256x256) and overwrites them."""
    os.makedirs(input_dir, exist_ok=True)  # Ensure directory exists
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    if not png_files:
        print("No PNG images found to resize.")
        return

    for image_file in png_files:
        input_path = os.path.join(input_dir, image_file)

        # Open image and enforce RGB mode for consistency
        with Image.open(input_path) as image:
            image = image.convert("RGBA")  # Keep transparency if present

            # Ensure target_size is a valid tuple
            if not isinstance(target_size, tuple) or len(target_size) != 2:
                raise ValueError(f"Invalid target size: {target_size}. Expected a tuple (width, height).")

            # Resize to fixed size (256x256 pixels)
            resized_image = image.resize(target_size, Image.LANCZOS)

            # Overwrite the original PNG image
            resized_image.save(input_path, format="PNG")

            print(f"Resized and overwritten {image_file} with size {target_size}.")

    print("All PNG images resized and overwritten successfully.")
