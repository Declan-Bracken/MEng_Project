from PIL import Image, ImageOps
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

"""
Class which takes in a directory of input images, resizes them to 640x640 default pixels, applies
auto-orient to them, and then creates sub-folders in an output directory to parse the images. The output
directory should be in COCO format (train, valid, test subfolders) and will be used at the next step to
label images using labelImg.
"""

class ImagePreprocessor:
    def __init__(self, source_folder, destination_folder, image_size=(640, 640)):
        self.source_folder = Path(source_folder)
        self.destination_folder = Path(destination_folder)
        self.image_size = image_size
        self.original_sizes = {}  # To record original sizes
        self._create_split_dirs()

    def _create_split_dirs(self):
        # Create train/val/test directories
        for split in ['train', 'valid', 'test']:
            split_path = self.destination_folder / split 
            split_path.mkdir(parents=True, exist_ok=True)

    # Add a method to save the original sizes to a JSON file
    def save_original_sizes(self, filename="original_sizes.json"):
        sizes_file = self.destination_folder / filename
        with open(sizes_file, 'w') as f:
            json.dump(self.original_sizes, f, indent=4)

    def _resize_and_orient_image(self, image_path):
        with Image.open(image_path) as img:
            # Record the original size
            self.original_sizes[image_path.name] = img.size

            # Auto-orient and resize
            img = ImageOps.exif_transpose(img)
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            return img
    
    def _process_and_move(self, image_path, split):
        img = self._resize_and_orient_image(image_path)
        dest_path = self.destination_folder / split / image_path.name
        img.save(dest_path)

    def distribute_images(self, train_size=0.7, val_size=0.2):
        images = list(self.source_folder.glob('*'))
        # Calculate number of images for each split
        num_train = int(len(images) * train_size)
        num_val = int(len(images) * val_size)

        # Split the images
        train_images, rest_images = train_test_split(images, train_size=num_train, shuffle=True)
        val_images, test_images = train_test_split(rest_images, train_size=num_val, shuffle=True)

        # Process and move images
        for image_path in train_images:
            self._process_and_move(image_path, 'train')
        for image_path in val_images:
            self._process_and_move(image_path, 'valid')
        for image_path in test_images:
            self._process_and_move(image_path, 'test')

        # After distributing images, save their original sizes
        self.save_original_sizes()

# Usage:
# preprocessor = ImagePreprocessor('/path/to/source/images', '/path/to/destination')
# preprocessor.distribute_images()

if __name__ == "__main__":
    source_folder = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Real Transcripts'
    destination_folder = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Preprocessed Real Transcripts'

    preprocessor = ImagePreprocessor(source_folder, destination_folder)
    preprocessor.distribute_images()
