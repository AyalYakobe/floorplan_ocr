import os
import pytesseract
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps
from difflib import SequenceMatcher

# Define dataset paths
IMAGE_PATH = "Synthetic_Dataset/final_images"
LABEL_PATH = "Synthetic_Dataset/labels"
OUTPUT_CSV = "Results/ocr_evaluation_results.csv"

# Function to load synthetic dataset
def load_synthetic_data(image_path, label_path):
    """ Load images and ground truth text from synthetic dataset """
    images = []
    ground_truths = []

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print("Error: Dataset paths do not exist.")
        return images, ground_truths

    for label_file in os.listdir(label_path):
        if label_file.endswith(".json"):
            label_filepath = os.path.join(label_path, label_file)

            with open(label_filepath, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            image_filename = label_data["image_path"]
            img_filepath = os.path.join(image_path, image_filename)

            if os.path.exists(img_filepath):
                images.append(img_filepath)
                ground_truth_text = " ".join([entry["text"] for entry in label_data["texts"]])
                ground_truths.append(ground_truth_text)

    return images, ground_truths

def perform_ocr(image_path):
    """ Run OCR on an image and extract text """
    image = Image.open(image_path).convert("L")  # Convert image to grayscale
    image = ImageOps.autocontrast(image)  # Improve contrast
    text = pytesseract.image_to_string(image, config="--psm 6")  # OCR config optimized for text blocks
    return text.strip()

def calculate_cer(gt_text, pred_text):
    """ Calculate Character Error Rate (CER) """
    return 1 - SequenceMatcher(None, gt_text, pred_text).ratio()

def calculate_wer(gt_text, pred_text):
    """ Calculate Word Error Rate (WER) """
    gt_words = gt_text.split()
    pred_words = pred_text.split()
    return 1 - SequenceMatcher(None, gt_words, pred_words).ratio()

def evaluate_ocr(images, ground_truths):
    """ Evaluate OCR performance on the synthetic dataset """
    results = []
    for img, gt_text in tqdm(zip(images, ground_truths), total=len(images), desc="Evaluating OCR..."):
        pred_text = perform_ocr(img)
        cer = calculate_cer(gt_text, pred_text)
        wer = calculate_wer(gt_text, pred_text)

        results.append({
            "Image": img,
            "Ground Truth": gt_text,
            "OCR Output": pred_text,
            "CER": cer,
            "WER": wer
        })

    return pd.DataFrame(results)

def simple_evaluator():
    """ Main function to run OCR evaluation, save results, and visualize errors """
    images, ground_truths = load_synthetic_data(IMAGE_PATH, LABEL_PATH)
    if images and ground_truths:
        evaluation_results = evaluate_ocr(images, ground_truths)

        # Save results to CSV
        evaluation_results.to_csv(OUTPUT_CSV, index=False)
        print(f"Evaluation complete! Results saved to {OUTPUT_CSV}")

        # Display results
        print(evaluation_results.head())

        # Plot Character Error Rate (CER) and Word Error Rate (WER)
        plt.figure(figsize=(12, 5))

        # Plot CER
        plt.subplot(1, 2, 1)
        plt.hist(evaluation_results["CER"], bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Character Error Rate (CER)")
        plt.ylabel("Frequency")
        plt.title("Distribution of CER across OCR Results")

        # Plot WER
        plt.subplot(1, 2, 2)
        plt.hist(evaluation_results["WER"], bins=20, edgecolor="black", alpha=0.7, color="orange")
        plt.xlabel("Word Error Rate (WER)")
        plt.ylabel("Frequency")
        plt.title("Distribution of WER across OCR Results")

        # Show plots
        plt.tight_layout()
        plt.show()
    else:
        print("No valid images and labels found for evaluation.")

