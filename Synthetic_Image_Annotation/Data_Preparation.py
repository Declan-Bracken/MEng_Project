import json
import os
from Download_imgs import download_images
""" Delete This
"""
def extract_image_url(data):
    try:
        # Navigate through the nested structure to get the image URL
        messages = data.get("body", {}).get("messages", [])[0].get("content", [])[1].get("image_url", {}).get("url", "")
        return messages
    except Exception as e:
        print(f"Error extracting image URL: {e}")
    return None

def generate_dataset_json(original_files, response_files, images_output_path, output_file):
    # Create requestID to URL Dictionary
    ID_to_URL_and_prompt = {}

    for original_file in original_files:
        with open(original_file, 'r') as of:
            for line in of:
                original_data = json.loads(line)
                # Key is request ID, prompt is first entry, Image url is second entry
                ID_to_URL_and_prompt[original_data.get("custom_id")] = [original_data.get("body", {}).get("messages", [])[0].get("content", [])[0].get("text", ""), extract_image_url(original_data)]

    cleaned_data = []

    for input_file in response_files:
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                response_content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                # Filter out responses indicating "not a transcript"
                if response_content.lower().strip() == "not a transcript":
                    continue

                custom_ID = data.get("custom_id")
                list = ID_to_URL_and_prompt[custom_ID] # Get prompt and url from dictionary
                prompt = list[0]
                image_url = list[1]

                # Download Image to folder
                local_path = download_images([image_url], images_output_path)
                print(f"Downloaded Image at {local_path}.")

                # Extract the relevant data
                cleaned_data.append({
                    "custom_id": custom_ID,
                    "prompt": prompt,
                    "content": response_content,
                    "image_path": local_path[0]
                })

    # Save the cleaned data to a new JSON file
    with open(output_file, 'w') as out_f:
        json.dump(cleaned_data, out_f, indent=4)

    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":

    # Original File (used for grabbing image url)
    # original_files = [
    #     "Synthetic_Image_Annotation/Input_JSONL/batchinput.jsonl"
    # ]
    original_files = [
        "Synthetic_Image_Annotation/Test_Data/Input_JSONL/input.jsonl"
    ]

    image_folder_path = "Synthetic_Image_Annotation/Test_Data/images/"

    # List of chatgpt response files
    # response_files = [
    #     "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_1.jsonl",
        # "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_2.jsonl",
        # "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_3.jsonl",
        # "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_4.jsonl",
    # ]
    response_files = ["Synthetic_Image_Annotation/Test_Data/Output_JSONL/batch_mNvrFzYMUglQNiBy9OYZoMpY_output.jsonl"]
    # Output file for cleaned data
    output_file = "Synthetic_Image_Annotation/Test_Data/cleaned_transcripts.json"

    generate_dataset_json(original_files, response_files, image_folder_path, output_file)
