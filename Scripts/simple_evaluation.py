import os
import pytesseract
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps
from difflib import SequenceMatcher

class OCREvaluator:
    def __init__(self, image_path, label_path, output_csv):
        """Initializes paths for images, labels, and output CSV."""
        self.image_path = image_path
        self.label_path = label_path
        self.output_csv = output_csv

    def load_synthetic_data(self):
        """Loads images and ground truth text from dataset."""
        images, ground_truths = [], []

        if not os.path.exists(self.image_path) or not os.path.exists(self.label_path):
            print("Error: Dataset paths do not exist.")
            return images, ground_truths

        for label_file in os.listdir(self.label_path):
            if label_file.endswith(".json"):
                label_filepath = os.path.join(self.label_path, label_file)

                with open(label_filepath, "r", encoding="utf-8") as f:
                    label_data = json.load(f)

                img_filepath = os.path.join(self.image_path, label_data["image_path"])
                if os.path.exists(img_filepath):
                    images.append(img_filepath)
                    ground_truths.append(" ".join([entry["text"] for entry in label_data["texts"]]))

        return images, ground_truths

    def perform_ocr(self, image_path):
        """Runs OCR on an image and extracts text."""
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = ImageOps.autocontrast(image)  # Enhance contrast
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip()

    def calculate_cer(self, gt_text, pred_text):
        """Computes Character Error Rate (CER)."""
        return 1 - SequenceMatcher(None, gt_text, pred_text).ratio()

    def calculate_wer(self, gt_text, pred_text):
        """Computes Word Error Rate (WER)."""
        gt_words, pred_words = gt_text.split(), pred_text.split()
        return 1 - SequenceMatcher(None, gt_words, pred_words).ratio()

    def evaluate_ocr(self, images, ground_truths):
        """Evaluates OCR performance across images."""
        results = []
        for img, gt_text in tqdm(zip(images, ground_truths), total=len(images), desc="Evaluating OCR..."):
            pred_text = self.perform_ocr(img)
            results.append({
                "Image": img,
                "Ground Truth": gt_text,
                "OCR Output": pred_text,
                "CER": self.calculate_cer(gt_text, pred_text),
                "WER": self.calculate_wer(gt_text, pred_text)
            })

        return pd.DataFrame(results)

    def run_simple_evaluation(self):
        """Runs OCR evaluation, saves results, and visualizes errors."""
        images, ground_truths = self.load_synthetic_data()
        if images and ground_truths:
            evaluation_results = self.evaluate_ocr(images, ground_truths)
            evaluation_results.to_csv(self.output_csv, index=False)
            print(f"Evaluation complete! Results saved to {self.output_csv}")
            print(evaluation_results.head())

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(evaluation_results["CER"], bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Character Error Rate (CER)")
            plt.ylabel("Frequency")
            plt.title("CER Distribution")
            plt.subplot(1, 2, 2)
            plt.hist(evaluation_results["WER"], bins=20, edgecolor="black", alpha=0.7, color="orange")
            plt.xlabel("Word Error Rate (WER)")
            plt.ylabel("Frequency")
            plt.title("WER Distribution")

            plt.tight_layout()
            plt.show()
        else:
            print("No valid images and labels found for evaluation.")
