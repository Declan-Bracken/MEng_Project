import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import Textarea, Button, VBox, Output, HBox
from IPython.display import display, clear_output
import folium
from folium import plugins
from IPython.display import display

class DatasetInteractor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
        self.current_index = 0  # Initialize the current index
        self.load_dataset()

    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)

    def save_dataset(self):
        with open(self.dataset_path, 'w') as f:
            json.dump(self.dataset, f, indent=4)
        print(f"Dataset saved to {self.dataset_path}")

    def get_absolute_image_path(self, relative_path):
        return os.path.join(self.dataset_dir, relative_path)

    def display_image(self, index):
        if index < 0 or index >= len(self.dataset):
            print(f"Index {index} out of bounds")
            return
        
        entry = self.dataset[index]
        image_path = entry['image']
        if not os.path.exists(image_path):
            print(f"Image not found at path: {image_path}")
            return

        # Create a map
        m = folium.Map(location=[0.5, 0.5], zoom_start=10, max_zoom=20, tiles=None)

        # Add the image overlay
        folium.raster_layers.ImageOverlay(
            name='Image Overlay',
            image=image_path,
            bounds=[[0, 0], [1, 1]],
            opacity=1,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(m)

        # Add zoom control and layer control
        folium.LayerControl().add_to(m)
        plugins.MousePosition().add_to(m)

        # Display the map
        display(m)

    def print_prompt_and_response(self, index):
        if index < 0 or index >= len(self.dataset):
            print(f"Index {index} out of bounds")
            return
        
        entry = self.dataset[index]
        conversations = entry['conversations']
        for conversation in conversations:
            role = conversation['role']
            content = conversation['content']
            print(f"{role}: {content}")

    def interactive_edit_response(self, index):
        if index < 0 or index >= len(self.dataset):
            print(f"Index {index} out of bounds")
            return

        entry = self.dataset[index]

        # Get Human Prompt:
        prompt = None
        for conversation in entry['conversations']:
            if conversation['role'] == 'user':
                prompt = conversation
                break
        
        if not prompt:
            print(f"No prompt found for entry index: {index}")

        # Create interactive widgets
        prompt_textarea = Textarea(
            value=prompt['content'],
            placeholder='Edit the prompt here...',
            description='Prompt:',
            disabled=False,
            layout={'width': '100%', 'height': '50px'}
        )

        # Find the assistant's response in the conversation
        assistant_response = None
        for conversation in entry['conversations']:
            if conversation['role'] == 'assistant':
                assistant_response = conversation
                break

        if not assistant_response:
            print(f"No assistant response found for entry index: {index}")
            return

        # Create interactive widgets
        response_textarea = Textarea(
            value=assistant_response['content'],
            placeholder='Edit the response here...',
            description='Response:',
            disabled=False,
            layout={'width': '100%', 'height': '200px'}
        )

        save_button = Button(description="Save Response", button_style='success')
        next_button = Button(description="Next", button_style='primary')
        prev_button = Button(description="Previous", button_style='primary')
        output = Output()

        def save_response(b):
            assistant_response['content'] = response_textarea.value
            self.save_dataset()
            with output:
                clear_output()
                print(f"Response updated and saved for entry index: {index}")

        def next_entry(b):
            self.current_index = (self.current_index + 1) % len(self.dataset)
            clear_output()
            self.display_image(self.current_index)
            # self.print_prompt_and_response(self.current_index)
            self.interactive_edit_response(self.current_index)

        def prev_entry(b):
            self.current_index = (self.current_index - 1) % len(self.dataset)
            clear_output()
            self.display_image(self.current_index)
            # self.print_prompt_and_response(self.current_index)
            self.interactive_edit_response(self.current_index)

        save_button.on_click(save_response)
        next_button.on_click(next_entry)
        prev_button.on_click(prev_entry)

        display(VBox([prompt_textarea, response_textarea, HBox([prev_button, next_button]), save_button, output]))

    def get_entry_by_index(self, index):
        if index < 0 or index >= len(self.dataset):
            return None
        return self.dataset[index]
    
    def alter_response(self, entry_id, new_response):
        entry = self.get_entry_by_id(entry_id)
        if not entry:
            print(f"No entry found with id: {entry_id}")
            return
        
        # Find the assistant's response in the conversation
        for conversation in entry['conversations']:
            if conversation['role'] == 'assistant':
                conversation['content'] = new_response
                break
        
        self.save_dataset()
        print(f"Response updated for entry id: {entry_id}")

if __name__ == "__main__":
    dataset_path = "Synthetic_Image_Annotation/Cleaned_JSON/cleaned_transcripts.json"
    interactor = DatasetInteractor(dataset_path)

    # Example usage
    example_id = "request-1"

    print("Displaying image:")
    interactor.display_image(example_id)

    print("\nPrinting prompt and response:")
    interactor.print_prompt_and_response(example_id)

    print("\nInteractive edit response:")
    interactor.interactive_edit_response(example_id)
