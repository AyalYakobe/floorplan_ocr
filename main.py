from Scripts.combine_text_w_image import apply_backgrounds_to_text_images
from Scripts.initial_image_processing import slice_images_in_directory, resize_png_images_uniform
from Scripts.text_generation import generate_text
import time

NUM_SAMPLES = 20

if __name__ == '__main__':

    # ------ Step .1 ------- Unecessary in the future
    # slice_images_in_directory("Clippings/Raw", "Clippings/Sliced", grid_size=2)
    # resize_png_images_uniform("Clippings/Sliced", (256, 256))

    # ------ Step .2 -------
    start_time = time.time()  # Start timer
    generate_text(num_samples=NUM_SAMPLES)

    # ------ Step .3 -------
    apply_backgrounds_to_text_images()

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f"Execution Time for {NUM_SAMPLES} samples: {elapsed_time_minutes:.2f} minutes")

    estimated_time_500k = (elapsed_time_minutes / NUM_SAMPLES) * 500000
    print(f"Estimated Execution Time for 500,000 samples: {estimated_time_500k:.2f} minutes")
