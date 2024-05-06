from pytesseract import image_to_data, Output
import numpy as np

class TextClassifier:
    def __init__(self, cropped_images, classes, boxes):
        self.cropped_images = cropped_images
        self.classes = classes
        self.boxes = boxes
        self.headers = []
        self.tables = []
        self.single_row = []

    def classify_text(self):
        for idx, box in enumerate(self.boxes):
            class_type = self.classes[idx]
            text = image_to_data(self.cropped_images[idx], output_type=Output.DICT, config='--psm 6')
            if class_type == 0:
                self.headers.append({'box': box, 'text': text})
            elif class_type == 1:
                self.tables.append({'box': box, 'text': text})
            else:
                self.single_row.append({'box': box, 'text': text})
        
        self.extract_horizontal_positions()

   # Only processes multirow tables currently 
    def extract_horizontal_positions(self):
        self.all_tables_data = []
        
        # Loop through tables
        for table in self.tables:
            positions = []
            texts = []
            y_positions = []
            line_numbers = []
            
            for i, text in enumerate(table['text']['text']):
                if text.strip():  # Ensure the text is not empty
                    # Get Bounding Box Positions
                    x1 = table['text']['left'][i]
                    width = table['text']['width'][i]
                    y1 = table['text']['top'][i]
                    x2 = x1 + width
                    height = table['text']['height'][i]
                    line_num = table['text']['line_num'][i]

                    positions.append([x1, x2])
                    texts.append(text)
                    y_positions.append((y1, height))
                    line_numbers.append(line_num)

            # Store data for the current table in a dictionary
            table_data = {
                'positions': np.array(positions),
                'texts': texts,
                'y_positions': y_positions,
                'line_numbers': line_numbers
            }
            self.all_tables_data.append(table_data)

        return self.all_tables_data

    def get_tables(self):
        return self.all_tables_data
