import json

from stem.util.system import start_time

from Scripts.combine_text_w_image import TextImageProcessor
from Scripts.simple_evaluation import load_synthetic_data, evaluate_ocr, IMAGE_PATH, LABEL_PATH, simple_evaluator
from Scripts.initial_image_processing import ImageProcessor
from Scripts.text_generation import SyntheticTextGenerator
import time
from sklearn.metrics import accuracy_score
import os
from Scripts.surya_evaluator import convert_to_coco, evaluate_images, split_images
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from PIL import Image
import Scripts.surya_evaluator as evaluator
import time
NUM_SAMPLES = 20


if __name__ == '__main__':

    # ------ Step .1 ------- Unecessary in the future
    # processor = ImageProcessor("Clippings/Raw", "Clippings/Sliced", grid_size=2, target_size=(256, 256))
    # processor.slice_images()
    # processor.resize_png_images()

    # ------ Step .2 -------
    # generator = SyntheticTextGenerator(
    #     font_dir="/System/Library/Fonts/Supplemental/",
    #     acceptable_fonts_file="Synthetic_Dataset/acceptable_fonts.txt",
    #     output_dir="Synthetic_Dataset",
    #     num_samples=NUM_SAMPLES
    # )
    # generator.generate_text_images()

    # ------ Step .3 -------
    # processor = TextImageProcessor(
    #     text_images_dir="Synthetic_Dataset/images",
    #     background_images_dir="Clippings/Sliced",
    #     output_dir="Synthetic_Dataset/final_images"
    # )
    # processor.estimate_execution_time(NUM_SAMPLES)

    # # ------ Step .4 -------
    simple_evaluator()

    # ------ Step .5 -------
    # convert_to_coco('Synthetic_Dataset/final_images', 'Synthetic_Dataset/labels', 'Synthetic_Dataset/coco_annotations.json')
    # evaluator.evaluate_ocr_custom()
