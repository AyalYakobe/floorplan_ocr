import fitz
from PIL import Image
import io
import os
import json
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

class SuryaPDFProcessor:
    def __init__(self, input_path, output_path, corner_size=(400, 400), confidence_threshold=0.70):
        self.input_path = input_path
        self.output_path = output_path
        self.corner_size = corner_size
        self.confidence_threshold = confidence_threshold
        self.output_image_dir = os.path.join(self.output_path, "extracted_images")
        self.output_json_dir = os.path.join(self.output_path, "json_labels")

        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

        # Create output directories if they don't exist
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_json_dir, exist_ok=True)

    def get_pdf_files(self):
        return [os.path.join(self.input_path, f) for f in os.listdir(self.input_path) if f.endswith(".pdf")]

    def extract_images(self):
        extracted_images = []
        pdf_files = self.get_pdf_files()

        for pdf_path in pdf_files:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.basename(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("ppm")))
                img_width, img_height = img.size

                left = max(0, img_width - self.corner_size[0])
                upper = max(0, img_height - self.corner_size[1])
                right = img_width
                lower = img_height
                cropped_img = img.crop((left, upper, right, lower))

                image_filename = f"{pdf_name}_page{page_num+1}.png"
                image_path = os.path.join(self.output_image_dir, image_filename)
                cropped_img.save(image_path)
                extracted_images.append(image_path)

        return extracted_images

    def evaluate_images(self, images):
        results = []
        for image_path in images:
            image = Image.open(image_path)
            ocr_results = self.recognition_predictor([image], [None], self.detection_predictor)

            text_lines = [
                {"text": line.text, "bbox": line.bbox, "confidence": line.confidence}
                for ocr_result in ocr_results
                for line in ocr_result.text_lines
                if line.confidence >= self.confidence_threshold
            ]

            results.append({"file_name": os.path.basename(image_path), "annotations": text_lines})

        return results

    def save_json(self, ocr_results):
        saved_files = []

        for result in ocr_results:
            json_filename = result["file_name"].replace(".png", ".json")
            json_path = os.path.join(self.output_json_dir, json_filename)

            with open(json_path, "w") as json_file:
                json.dump(result, json_file, indent=4)

            saved_files.append(json_path)

        return saved_files

    def run(self):
        extracted_images = self.extract_images()
        ocr_results = self.evaluate_images(extracted_images)
        saved_json_files = self.save_json(ocr_results)

        return saved_json_files


