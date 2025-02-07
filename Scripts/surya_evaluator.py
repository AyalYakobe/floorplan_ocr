import json
import os
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SuryaEvaluator:
    def __init__(self, image_dir, label_dir, output_coco_path, baseline_results, fine_tuned_results):
        """Initializes paths and result storage for OCR evaluation."""
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_coco_path = output_coco_path
        self.ground_truth = None
        self.baseline_results = baseline_results
        self.fine_tuned_results = fine_tuned_results

    def convert_to_coco(self):
        """Converts dataset annotations to COCO format."""
        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "text"}]}
        annotation_id = 1

        for label_file in os.listdir(self.label_dir):
            if label_file.endswith(".json"):
                with open(os.path.join(self.label_dir, label_file), 'r') as f:
                    data = json.load(f)

                image_path = os.path.basename(data['image_path'])
                image = Image.open(os.path.join(self.image_dir, image_path))
                width, height = image.size

                image_id = len(coco["images"]) + 1
                coco["images"].append({
                    "id": image_id,
                    "file_name": image_path,
                    "width": width,
                    "height": height
                })

                for text_entry in data['texts']:
                    x, y = text_entry['position']
                    text = text_entry['text']
                    bbox = [x, y, 50, 20]
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "bbox": bbox,
                        "text": text,
                        "category_id": 1
                    })
                    annotation_id += 1

        with open(self.output_coco_path, 'w') as f:
            json.dump(coco, f, indent=4)

    def split_images(self):
        """Splits dataset into baseline and fine-tuned sets."""
        images = [f for f in os.listdir(self.image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        midpoint = len(images) // 2
        return images[:midpoint], images[midpoint:]

    def evaluate_images(self, model, images):
        """Runs OCR on images and stores extracted text."""
        detection_predictor = DetectionPredictor()
        results = []

        for image_name in images:
            image_path = os.path.join(self.image_dir, image_name)
            image = Image.open(image_path)

            ocr_results = model([image], [None], detection_predictor)

            text_lines = [{"text": line.text, "bbox": line.bbox, "confidence": line.confidence}
                          for ocr_result in ocr_results for line in ocr_result.text_lines]

            results.append({"image": image_name, "text_lines": text_lines})

        return results

    def load_ground_truth(self):
        """Loads ground truth annotations from COCO format."""
        with open(self.output_coco_path, "r") as f:
            coco_data = json.load(f)

        self.ground_truth = {img["file_name"]: [] for img in coco_data["images"]}
        for annotation in coco_data["annotations"]:
            image_name = next(img["file_name"] for img in coco_data["images"] if img["id"] == annotation["image_id"])
            self.ground_truth[image_name].append(annotation["text"])

    def evaluate_results(self, results):
        """Computes accuracy, precision, recall, and F1 score."""
        y_true, y_pred = [], []

        for result in results:
            image_name = result["image"]
            predicted_texts = [line["text"] for line in result["text_lines"]]

            if image_name in self.ground_truth:
                gt_texts = self.ground_truth[image_name]

                if len(gt_texts) > len(predicted_texts):
                    predicted_texts.extend([""] * (len(gt_texts) - len(predicted_texts)))
                elif len(predicted_texts) > len(gt_texts):
                    gt_texts.extend([""] * (len(predicted_texts) - len(gt_texts)))

                y_true.extend(gt_texts)
                y_pred.extend(predicted_texts)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

        return accuracy, precision, recall, f1

    def plot_comparison(self, baseline_metrics, fine_tuned_metrics):
        """Plots a bar graph comparing baseline and fine-tuned models."""
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width / 2, baseline_metrics, width, label="Baseline")
        ax.bar(x + width / 2, fine_tuned_metrics, width, label="Fine-Tuned")
        ax.set_xlabel("Metrics")
        ax.set_title("Baseline vs Fine-Tuned OCR Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.show()

    def run_surya_evaluation(self):
        """Runs full evaluation process for OCR models."""
        start_time = time.time()

        self.convert_to_coco()
        self.load_ground_truth()

        baseline_images, fine_tuned_images = self.split_images()

        baseline_recognition = RecognitionPredictor()
        fine_tuned_recognition = RecognitionPredictor()

        print("\nEvaluating Baseline OCR...")
        self.baseline_results = self.evaluate_images(baseline_recognition, baseline_images)
        with open("Synthetic_Dataset/surya_data/baseline_results.json", "w") as f:
            json.dump(self.baseline_results, f, indent=4)

        print("\nEvaluating Fine-Tuned OCR...")
        self.fine_tuned_results = self.evaluate_images(fine_tuned_recognition, fine_tuned_images)
        with open("Synthetic_Dataset/surya_data/fine_tuned_results.json", "w") as f:
            json.dump(self.fine_tuned_results, f, indent=4)

        print("\nResults saved!")

        baseline_metrics = self.evaluate_results(self.baseline_results)
        fine_tuned_metrics = self.evaluate_results(self.fine_tuned_results)

        print("\nBaseline Model Performance:")
        print(f"Accuracy: {baseline_metrics[0]:.4f}, Precision: {baseline_metrics[1]:.4f}, "
              f"Recall: {baseline_metrics[2]:.4f}, F1 Score: {baseline_metrics[3]:.4f}")

        print("\nFine-Tuned Model Performance:")
        print(f"Accuracy: {fine_tuned_metrics[0]:.4f}, Precision: {fine_tuned_metrics[1]:.4f}, "
              f"Recall: {fine_tuned_metrics[2]:.4f}, F1 Score: {fine_tuned_metrics[3]:.4f}")

        self.plot_comparison(baseline_metrics, fine_tuned_metrics)

        elapsed_time = time.time() - start_time
        num_samples = len(baseline_images) + len(fine_tuned_images)
        print(f"\nTotal Evaluation Time: {elapsed_time:.2f} seconds for {num_samples} samples")

        if num_samples > 0:
            estimated_time_500k = (elapsed_time / num_samples) * 500000
            estimated_hours = estimated_time_500k / 3600
            print(f"Estimated Time for 500K samples: {estimated_time_500k:.2f} seconds (~{estimated_hours:.2f} hours)")
