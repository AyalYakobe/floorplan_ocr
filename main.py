from Scripts.combine_text_w_image import TextImageProcessor
from Scripts.sidebar_processor import SuryaPDFProcessor
from Scripts.simple_evaluation import OCREvaluator
from Scripts.initial_image_processing import ImageProcessor
from Scripts.text_generation import SyntheticTextGenerator
NUM_SAMPLES = 20


if __name__ == '__main__':
    is_sidebar_details = False

    if is_sidebar_details:
        # ------ Only Step -------
        processor = SuryaPDFProcessor(
            'Synthetic_Dataset/sidebar_input',
            'Synthetic_Dataset/sidebar_output',
            corner_size=(250, 300),
            confidence_threshold=0.75
        )
        json_results = processor.run()
    else:
        # ------ Step .1 -------
        processor = ImageProcessor(
            input_dir="Clippings/Raw",
            output_dir="Clippings/Sliced",
            grid_size=2,
            target_size=(256, 256)
        )
        processor.slice_images()
        processor.resize_png_images()

        # ------ Step .2 -------
        generator = SyntheticTextGenerator(
            font_dir="/System/Library/Fonts/Supplemental/",
            acceptable_fonts_file="Synthetic_Dataset/acceptable_fonts.txt",
            output_dir="Synthetic_Dataset",
            num_samples=NUM_SAMPLES
        )
        generator.generate_text_images()

        # ------ Step .3 -------
        processor = TextImageProcessor(
            text_images_dir="Synthetic_Dataset/images",
            background_images_dir="Clippings/Sliced",
            output_dir="Synthetic_Dataset/final_images"
        )
        processor.estimate_execution_time(NUM_SAMPLES)

        # ------ Step .4 -------
        simple_evaluator = OCREvaluator(
            image_path="Synthetic_Dataset/final_images",
            label_path="Synthetic_Dataset/labels",
            output_csv="Results/ocr_evaluation_results.csv"
        )
        simple_evaluator.run_simple_evaluation()
