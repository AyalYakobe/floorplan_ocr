import os
import random
import json
import string
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

class SyntheticTextGenerator:
    def __init__(self, font_dir, acceptable_fonts_file, output_dir, num_samples=100):
        """
        Initialize the text generator with font directory, accepted fonts file, and output paths.
        """
        self.font_dir = font_dir
        self.acceptable_fonts_file = acceptable_fonts_file
        self.output_dir = output_dir
        self.num_samples = num_samples

        self.image_dir = os.path.join(self.output_dir, "images")
        self.label_dir = os.path.join(self.output_dir, "labels")

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        self.available_fonts = self.load_fonts()

    def load_fonts(self):
        """Loads acceptable fonts from the given file and verifies their existence."""
        with open(self.acceptable_fonts_file, "r") as f:
            acceptable_fonts = {line.strip() for line in f.readlines()}

        available_fonts = [
            os.path.join(self.font_dir, font) for font in acceptable_fonts if font.endswith(('.ttf', '.otf'))
        ]

        if not available_fonts:
            raise ValueError("No acceptable fonts found. Check your acceptable_fonts.txt file.")

        return available_fonts

    def generate_random_text(self):
        """Generates a random alphanumeric string with optional hyphens."""
        length = random.randint(1, 10)
        chars = string.ascii_uppercase + string.digits
        text = ''.join(random.choices(chars, k=length))

        # Add hyphens with a maximum of 2 in random positions
        if length > 2 and random.choice([True, False]):
            hyphen_positions = sorted(random.sample(range(1, length), k=random.randint(1, 2)))
            for pos in reversed(hyphen_positions):
                text = text[:pos] + '-' + text[pos:]

        return text

    def generate_text_images(self):
        """Generates synthetic text images and labels."""
        for i in tqdm(range(self.num_samples), desc="Generating Text Samples"):
            stacked_text = random.random() < 0.30  # 30% chance of stacked text
            num_lines = random.randint(2, 4) if stacked_text else 1

            texts = [self.generate_random_text() for _ in range(num_lines)]
            font_size = random.randint(60, 80)
            font_path = random.choice(self.available_fonts)

            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                continue

            # Use getbbox() instead of the deprecated getsize()
            text_sizes = [font.getbbox(text) for text in texts]
            text_sizes = [(x_max - x_min, y_max - y_min) for (x_min, y_min, x_max, y_max) in text_sizes]

            max_text_width = max(w for w, h in text_sizes)
            total_text_height = sum(h for w, h in text_sizes) + (10 * (len(texts) - 1))

            padding = 10
            img_size = (max_text_width + 2 * padding, total_text_height + 2 * padding)

            alternate_white = i % 2 == 0

            if alternate_white:
                image = Image.new("RGBA", img_size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                white_box_coords = (padding, padding, max_text_width + padding, total_text_height + padding)
                draw.rectangle(white_box_coords, fill=(255, 255, 255, 255))
            else:
                image = Image.new("RGBA", img_size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)

            text_y = padding
            label_data = []

            for text in texts:
                text_width, text_height = font.getbbox(text)[2:]  # Extract width and height
                text_x = padding
                draw.text((text_x, text_y), text, font=font, fill="black")

                underline = random.choice([True, False])
                if underline:
                    line_y = text_y + text_height + 2
                    draw.line([(text_x, line_y), (text_x + text_width, line_y)], fill="black", width=2)

                label_data.append({
                    "text": text,
                    "position": (text_x, text_y),
                    "underlined": underline
                })

                text_y += text_height + 10

            angle = random.choice([0, 90])
            image = image.rotate(angle, expand=True)

            image_filename = f"synthetic_{i:06d}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            image.save(image_path)

            final_label_data = {
                "texts": label_data,
                "font": os.path.basename(font_path),
                "rotation": angle,
                "image_path": image_filename,
                "white_background": alternate_white
            }
            label_filename = f"synthetic_{i:06d}.json"
            label_path = os.path.join(self.label_dir, label_filename)

            with open(label_path, "w") as label_file:
                json.dump(final_label_data, label_file, indent=4)

        print("Text generation completed successfully.")


