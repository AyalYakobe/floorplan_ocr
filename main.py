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




NUM_SAMPLES = 20

if __name__ == '__main__':

    import Scripts.surya_evaluator as evaluator

    print(dir(evaluator))

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
    # convert_to_coco('Synthetic_Dataset/final_images', 'Synthetic_Dataset/labels', 'Synthetic_Dataset/coco_annotations.json')
    start_time = time.time()  # Start time tracking

    final_images_dir = "Synthetic_Dataset/final_images"

    # Load ground truth from COCO annotations
    ground_truth = evaluator.load_ground_truth("Synthetic_Dataset/coco_annotations.json")

    # Split images for baseline and fine-tuned evaluation
    baseline_images, fine_tuned_images = split_images(final_images_dir)

    # Load OCR Models
    baseline_recognition = RecognitionPredictor()
    fine_tuned_recognition = RecognitionPredictor()
    detection_predictor = DetectionPredictor()

    # Evaluate Baseline Images
    print("\nEvaluating Baseline OCR...")
    baseline_results = evaluate_images(baseline_recognition, baseline_images, final_images_dir)
    with open("baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=4)

    # Evaluate Fine-Tuned Images
    print("\nEvaluating Fine-Tuned OCR...")
    fine_tuned_results = evaluate_images(fine_tuned_recognition, fine_tuned_images, final_images_dir)
    with open("fine_tuned_results.json", "w") as f:
        json.dump(fine_tuned_results, f, indent=4)

    print("\nBaseline and Fine-Tuned Results Saved!")

    # Debugging check for missing images
    print(f"Total images in ground truth: {len(ground_truth)}")
    print(f"Total images in baseline OCR results: {len(baseline_results)}")
    print(f"Total images in fine-tuned OCR results: {len(fine_tuned_results)}")

    # Load Results for Comparison
    with open("baseline_results.json", "r") as f:
        baseline_results = json.load(f)

    with open("fine_tuned_results.json", "r") as f:
        fine_tuned_results = json.load(f)

    # Evaluate performance
    baseline_metrics = evaluator.evaluate_results(baseline_results, ground_truth)
    fine_tuned_metrics = evaluator.evaluate_results(fine_tuned_results, ground_truth)

    print("\nBaseline Model Performance:")
    print(
        f"Accuracy: {baseline_metrics[0]:.4f}, Precision: {baseline_metrics[1]:.4f}, Recall: {baseline_metrics[2]:.4f}, F1 Score: {baseline_metrics[3]:.4f}")

    print("\nFine-Tuned Model Performance:")
    print(
        f"Accuracy: {fine_tuned_metrics[0]:.4f}, Precision: {fine_tuned_metrics[1]:.4f}, Recall: {fine_tuned_metrics[2]:.4f}, F1 Score: {fine_tuned_metrics[3]:.4f}")

    # End time tracking and print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time

    num_samples = len(baseline_images) + len(fine_tuned_images)  # Total samples tested
    print(f"\nTotal Evaluation Time: {elapsed_time:.2f} seconds for {num_samples} samples")
    # Estimate time for 500K samples
    if num_samples > 0:
        estimated_time_500k = (elapsed_time / num_samples) * 500000
        estimated_hours = estimated_time_500k / 3600  # Convert to hours
        print(f"Estimated Time for 500K samples: {estimated_time_500k:.2f} seconds (~{estimated_hours:.2f} hours)")

