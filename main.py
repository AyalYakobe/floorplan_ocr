import json

from Scripts.combine_text_w_image import apply_backgrounds_to_text_images
from Scripts.simple_evaluation import load_synthetic_data, evaluate_ocr, IMAGE_PATH, LABEL_PATH, simple_evaluator
from Scripts.initial_image_processing import slice_images_in_directory, resize_png_images_uniform
from Scripts.text_generation import generate_text
import time

from sklearn.metrics import accuracy_score

import os
from Scripts.surya_evaluator import convert_to_coco, evaluate_images, split_images
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from PIL import Image

import Scripts.surya_evaluator as evaluator




NUM_SAMPLES = 20

if __name__ == '__main__':

    # ------ Step .1 ------- Unecessary in the future
    # slice_images_in_directory("Clippings/Raw", "Clippings/Sliced", grid_size=2)
    # resize_png_images_uniform("Clippings/Sliced", (256, 256))

    # ------ Step .2 -------
    # start_time = time.time()  # Start timer
    # generate_text(num_samples=NUM_SAMPLES)
    #
    # # ------ Step .3 -------
    # apply_backgrounds_to_text_images()
    #
    # end_time = time.time()
    # elapsed_time_minutes = (end_time - start_time) / 60
    # print(f"Execution Time for {NUM_SAMPLES} samples: {elapsed_time_minutes:.2f} minutes")
    #
    # estimated_time_500k = (elapsed_time_minutes / NUM_SAMPLES) * 500000
    # print(f"Estimated Execution Time for 500,000 samples: {estimated_time_500k:.2f} minutes")

    # # ------ Step .4 -------
    # simple_evaluator()

    # ------ Step .5 -------
    convert_to_coco('Synthetic_Dataset/final_images', 'Synthetic_Dataset/labels', 'Synthetic_Dataset/coco_annotations.json')
    # evaluator.evaluate_ocr_custom()
