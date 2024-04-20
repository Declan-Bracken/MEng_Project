
# # Example usage:
# input_folder_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Sample Transcripts/Actual Sample Transcripts'
# output_folder_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/transcript-table-detector-6'

import json
import shutil
from pathlib import Path
from shutil import copy2

class COCOIntegrator:
    def __init__(self, base_dataset_path, new_dataset_path, combined_dataset_path):
        self.base_dataset_path = Path(base_dataset_path)
        self.new_dataset_path = Path(new_dataset_path)
        self.combined_dataset_path = Path(combined_dataset_path)
        self.splits = ['train', 'valid', 'test']

        # Create the combined dataset structure
        self._create_combined_dataset_structure()

    def _create_combined_dataset_structure(self):
        for split in self.splits:
            (self.combined_dataset_path / split).mkdir(parents=True, exist_ok=True)

    def integrate(self):
        for split in self.splits:
            base_split_annotations_path = self.base_dataset_path / split / '_annotations.coco.json'
            new_split_annotations_path = self.new_dataset_path / split / 'coco_annotations.json'
            combined_split_annotations_path = self.combined_dataset_path / split / '_annotations.coco.json'
            
            # Assert files exist
            assert base_split_annotations_path.exists(), f"Base annotations path does not exist for {split}"
            assert new_split_annotations_path.exists(), f"New annotations path does not exist for {split}"
            
            # Load and combine annotations
            with open(base_split_annotations_path) as f:
                base_annotations = json.load(f)
            with open(new_split_annotations_path) as f:
                new_annotations = json.load(f)
            
            # Integrate and write combined annotations to the new directory
            combined_annotations = self._integrate_split(base_annotations, new_annotations)
            print(combined_annotations['images'])
            with open(combined_split_annotations_path, 'w') as f:
                json.dump(combined_annotations, f, indent=2)

            # Copy base dataset images to the combined dataset directory
            base_images_path = self.base_dataset_path / split 
            combined_images_path = self.combined_dataset_path / split
            for image_file in base_images_path.iterdir():
                if not image_file.suffix.lower() in ['.json']:  # Skip json files
                    shutil.copy(image_file, combined_images_path)

            # Copy new dataset images to the combined dataset directory
            new_images_path = self.new_dataset_path / split
            for image_file in new_images_path.iterdir():
                # Avoid overwriting images from the base dataset
                if not image_file.suffix.lower() in ['.json'] and not (combined_images_path / image_file.name).exists():
                    shutil.copy(image_file, combined_images_path)
    
    def _integrate_split(self, base_annotations, new_annotations):
        # Assuming image IDs and annotation IDs in new_annotations start from 1 and are sequential.
        # If not, you should adjust the ID generation logic accordingly.

        # Find the highest IDs in the base annotations to avoid conflicts
        max_image_id = self._find_max_id(base_annotations['images'], 'id')
        max_annotation_id = self._find_max_id(base_annotations['annotations'], 'id')

        # Prepare a mapping for new image IDs to their updated IDs
        id_mapping = {}

        # Integrate new images
        for image in new_annotations['images']:
            max_image_id += 1  # Increment the max ID for each new image
            id_mapping[image['id']] = max_image_id  # Map old ID to new ID
            image['id'] = max_image_id  # Assign the new ID
            base_annotations['images'].append(image)  # Append the updated image to the base annotations

        # Integrate new annotations
        for annotation in new_annotations['annotations']:
            max_annotation_id += 1  # Increment the max ID for each new annotation
            annotation['id'] = max_annotation_id  # Assign the new ID
            # Remap the image_id in the annotation to the updated image ID
            annotation['image_id'] = id_mapping.get(annotation['image_id'], annotation['image_id'])
            base_annotations['annotations'].append(annotation)  # Append the updated annotation

        return base_annotations
    
    @staticmethod
    def _find_max_id(items, key):
        return max((item[key] for item in items), default=0, key=int)


# Example usage
if __name__ == "__main__":
    base_dataset_path = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/transcript-table-detector-6"
    new_dataset_path = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Preprocessed Real Transcripts"
    combined_dataset_path = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/Combined_Dataset"
    integrator = COCOIntegrator(base_dataset_path, new_dataset_path, combined_dataset_path)
    integrator.integrate()

