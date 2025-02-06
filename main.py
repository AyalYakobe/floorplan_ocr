from scripts.combine_text_w_image import apply_backgrounds_to_text_images
from scripts.initial_image_processing import slice_and_save_image
from scripts.text_generation import generate_text


if __name__ == '__main__':

    # ------ Step .1 -------
    # generate_text(num_samples=20)

    # ------ Step .2 ------- Unecessary in the future
    # slice_and_save_image("t1.png", "Clippings/Sliced", grid_size=2)

    # ------ Step .3 ------- Unecessary in the future
    apply_backgrounds_to_text_images()
