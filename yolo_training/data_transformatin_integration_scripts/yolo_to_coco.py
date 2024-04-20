import json
from pathlib import Path

class YOLOToCOCOConverter:
    def __init__(self, yolo_annotations_path, images_dir_path, image_size=(640, 640)):
        self.yolo_annotations_path = Path(yolo_annotations_path)
        self.images_dir_path = Path(images_dir_path)
        self.image_width, self.image_height = image_size
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []  # You need to fill this with your actual category data
        }
        self.annotation_id = 1

    def add_category(self, category_id, category_name):
        self.coco_data["categories"].append({
            "id": category_id,
            "name": category_name
        })

    def convert(self):
        image_id = 1
        for annotation_file in self.yolo_annotations_path.glob('*.txt'):
            # Assuming image files have the same stem as annotation files
            image_file = self._find_matching_image_file(annotation_file.stem)
            if not image_file:
                print(f"No image file found for annotation {annotation_file.stem}")
                continue
                
            self.coco_data["images"].append({
                "id": image_id,
                "width": self.image_width,
                "height": self.image_height,
                "file_name": image_file.name,
            })

            with open(annotation_file) as f:
                for line in f:
                    self._process_annotation_line(line, image_id)
            image_id += 1

    def _find_matching_image_file(self, stem):
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_image_file = self.images_dir_path / (stem + ext)
            if potential_image_file.exists():
                return potential_image_file
        return None

    def _process_annotation_line(self, line, image_id):
        class_id, x_center, y_center, width, height = map(float, line.split())
        x_min = (x_center - width / 2) * self.image_width
        y_min = (y_center - height / 2) * self.image_height
        bbox_width = width * self.image_width
        bbox_height = height * self.image_height

        self.coco_data["annotations"].append({
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": int(class_id) + 1,
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0,
        })
        self.annotation_id += 1

    def save(self, output_path):
        with open(Path(output_path), 'w') as f:
            json.dump(self.coco_data, f, indent=2)

# Example usage
if __name__ == "__main__":
    yolo_path = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/makesense_labelsv1"
    coco_path = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Preprocessed Real Transcripts/valid"
    converter = YOLOToCOCOConverter(yolo_path, coco_path)
    # Add categories as needed
    converter.add_category(1, "table")
    converter.convert()
    converter.save(Path(coco_path + '/coco_annotations.json'))
