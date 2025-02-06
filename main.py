from scripts.combine_text_w_image import apply_backgrounds_to_text_images
from scripts.initial_image_processing import slice_images_in_directory, resize_png_images_uniform
from scripts.text_generation import generate_text


if __name__ == '__main__':

    # ------ Step .1 -------
    # generate_text(num_samples=20)

    # ------ Step .2 ------- Unecessary in the future
    slice_images_in_directory("Clippings/Raw", "Clippings/Sliced", grid_size=2)
    resize_png_images_uniform("Clippings/Sliced", (256, 256))

    # ------ Step .3 ------- Unecessary in the future
    # apply_backgrounds_to_text_images()
