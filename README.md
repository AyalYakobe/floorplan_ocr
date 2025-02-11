# Floorplan OCR Testing Generator

## Overview
The **Floorplan OCR Testing Generator** is a comprehensive system designed to process images, generate textual content from those images, and evaluate the accuracy of the generated text. It consists of multiple modular components that handle image preprocessing, text extraction, content generation, and quality evaluation. 

The system is built to support batch processing of images extracted from PDFs, ensuring that OCR-generated text is well-integrated with the images and properly evaluated for quality assurance. 

## Features
- **Image Processing**: Prepares images for OCR processing by enhancing quality and extracting key features.
- **Text Generation**: Uses OCR and additional processing techniques to generate textual descriptions from processed images.
- **Evaluation**: Assesses the accuracy and quality of generated text.
- **Combining Text with Images**: Integrates the extracted text with images to create meaningful and usable outputs.
- **PDF Processing**: Extracts images from PDF documents and applies OCR analysis to generate annotated JSON outputs.
- **Modular Design**: The system is structured into distinct modules, making it easy to update, customize, and maintain.

## System Workflow
The system can operate in two modes:
1. **Step-by-Step Processing**: Each step in the workflow can be executed separately to allow for more control and debugging.
2. **One-Step Execution**: A single command runs all five steps in sequence for an automated pipeline.
3. **Sidebar Training Mode**: Modify the main script to use sidebar training data instead of the default five-step training data.

### Processing Steps
1. **Initial Image Processing**: Enhances and prepares images for text extraction.
2. **Text Generation**: Extracts text from images using OCR and text generation algorithms.
3. **Text-Image Integration**: Combines extracted text with images to form structured outputs.
4. **Evaluation**: Assesses the quality of the generated content.
5. **PDF Processing (if applicable)**: Extracts images from PDFs and processes them accordingly.

## File Descriptions
### Core Modules
- **`main.py`**: The central script that orchestrates the execution of different modules. It allows users to either run all five steps in sequence, use sidebar training data, or execute an individual step as needed.
- **`initial_image_processing.py`**: Handles preprocessing tasks such as resizing, noise reduction, and feature extraction to prepare images for OCR analysis.
- **`text_generation.py`**: Implements text extraction and generation algorithms to interpret and process content from images.
- **`combine_text_w_image.py`**: Merges the extracted text with processed images to generate meaningful annotated outputs.
- **`simple_evaluation.py`**: Provides a basic method to evaluate the extracted text’s quality and accuracy.
- **`sidebar_processor.py`**: Handles any special sidebar-related text extraction and processing for enhanced OCR results. Extracts images from PDF files, applies OCR, and stores the output in JSON format for further processing.

## Usage
Run the main script to process images:
```bash
python main.py
```

To modify whether to use sidebar training data or the five-step training data, adjust the relevant settings in `main.py`.

## Output
- Extracted and processed images.
- Generated text output in JSON format.
- Combined text-image outputs.

## Future Enhancements
- Support for additional OCR models.
- Improved AI-driven text enhancement.
- Interactive UI for processing and evaluating outputs.

This project was developed using PyCharm as the primary IDE.

