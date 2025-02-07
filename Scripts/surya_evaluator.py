import json
import os
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import Scripts.surya_evaluator as evaluator

print(dir(evaluator))


def convert_to_coco(image_dir, label_dir, output_path):
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "text"}]}
    annotation_id = 1

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".json"):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                data = json.load(f)

            image_path = os.path.basename(data['image_path'])
            image = Image.open(os.path.join(image_dir, image_path))
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

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)


def split_images(directory_path):
    images = [f for f in os.listdir(directory_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    midpoint = len(images) // 2
    return images[:midpoint], images[midpoint:]  # Baseline, Fine-Tuned


def evaluate_images(model, images, directory_path):
    detection_predictor = DetectionPredictor()
    results = []

    for image_name in images:
        image_path = os.path.join(directory_path, image_name)
        image = Image.open(image_path)

        # `RecognitionPredictor` returns a list of OCR results
        ocr_results = model([image], [None], detection_predictor)

        # Extract text lines for each OCRResult
        for ocr_result in ocr_results:  # Iterate through the list of results
            text_lines = [
                {
                    "text": line.text,
                    "bbox": line.bbox,
                    "confidence": line.confidence,
                }
                for line in ocr_result.text_lines  # Access `text_lines` from the result
            ]

            results.append({
                "image": image_name,
                "text_lines": text_lines
            })

    return results

def evaluate_image(image_path):
    recognition_predictor = RecognitionPredictor()
    detection_predictor = DetectionPredictor()

    image = Image.open(image_path)
    results = recognition_predictor([image], [None], detection_predictor)
    return results[0]["text_lines"]

# Load Ground Truth from COCO JSON
def load_ground_truth(coco_file):
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    ground_truth = {}

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        image_name = next((img["file_name"] for img in coco_data["images"] if img["id"] == image_id), None)

        if image_name:
            if image_name not in ground_truth:
                ground_truth[image_name] = []
            ground_truth[image_name].append(annotation["text"])

    return ground_truth

def evaluate_results(results, ground_truth):
    y_true = []
    y_pred = []

    for result in results:
        image_name = result["image"]
        predicted_texts = [line["text"] for line in result["text_lines"]]

        if image_name in ground_truth:
            gt_texts = ground_truth[image_name]

            # Ensure the lengths match
            if len(gt_texts) > len(predicted_texts):
                predicted_texts.extend([""] * (len(gt_texts) - len(predicted_texts)))  # Pad missing predictions
            elif len(predicted_texts) > len(gt_texts):
                gt_texts.extend([""] * (len(predicted_texts) - len(gt_texts)))  # Pad missing ground truth

            y_true.extend(gt_texts)
            y_pred.extend(predicted_texts)

    # Final check: If lengths still don't match, print debug info
    if len(y_true) != len(y_pred):
        print(f"Error: y_true has {len(y_true)} samples but y_pred has {len(y_pred)} samples!")
        print("Check your ground truth JSON file for inconsistencies.")

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    return accuracy, precision, recall, f1

# Plot the comparison results
def plot_comparison(baseline_metrics, fine_tuned_metrics):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    baseline_values = baseline_metrics
    fine_tuned_values = fine_tuned_metrics

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, baseline_values, width, label="Baseline")
    ax.bar(x + width/2, fine_tuned_values, width, label="Fine-Tuned")

    ax.set_xlabel("Metrics")
    ax.set_title("Comparison of Baseline and Fine-Tuned OCR Models")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.show()

def evaluate_ocr_custom():

    import Scripts.surya_evaluator as evaluator
    print(dir(evaluator))

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
