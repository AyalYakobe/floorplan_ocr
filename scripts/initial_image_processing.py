import os
from PIL import Image

def slice_and_save_image(input_image, output_dir, grid_size=5):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(input_image)
    width, height = image.size
    slice_width = width // grid_size
    slice_height = height // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            left = col * slice_width
            upper = row * slice_height
            right = left + slice_width
            lower = upper + slice_height

            sliced_image = image.crop((left, upper, right, lower))
            slice_filename = f"slice_{row}_{col}.png"
            slice_path = os.path.join(output_dir, slice_filename)
            sliced_image.save(slice_path)

            print(f"Saved: {slice_path}")