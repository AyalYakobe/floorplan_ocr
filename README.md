# Floorplan OCR Testing Generator

## Overview
This project is a comprehensive system for processing images, generating text, and evaluating outputs. It consists of several modules that work together to handle image processing, text generation, and evaluation of generated content.

## Features
- **Image Processing**: Handles initial image processing tasks.
- **Text Generation**: Generates textual content based on processed images.
- **Evaluation**: Evaluates generated text for accuracy and quality.
- **Combining Text with Images**: Integrates generated text with images to create cohesive outputs.
- **Main Module**: Coordinates different components to ensure seamless execution.
- **PDF Processing**: Extracts images from PDFs and performs OCR analysis.

## File Descriptions
```
- main.py: The central script that orchestrates different modules.
- initial_image_processing.py: Handles image preprocessing and feature extraction.
- text_generation.py: Implements text generation algorithms.
- combine_text_w_image.py: Merges generated text with processed images.
- simple_evaluation.py: Provides a basic evaluation mechanism for generated content.
- surya_evaluator.py: Advanced evaluation mechanism for text and image alignment.
- surya_pdf_processor.py: Extracts images from PDF files, performs OCR analysis, and saves annotations in JSON format.
```

## Requirements
This project was developed using PyCharm.

## Usage
```
python main.py
```
