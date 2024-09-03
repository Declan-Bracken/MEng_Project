import json
import os
import requests
from pathlib import Path
from Synthetic_Image_Annotation.Generate_Data.Download_imgs import download_images

# Function to extract the image URL from the request data
def extract_image_url(data):
    try:
        messages = data.get("body", {}).get("messages", [])[0].get("content", [])[1].get("image_url", {}).get("url", "")
        return messages
    except Exception as e:
        print(f"Error extracting image URL: {e}")
    return None

# Function to generate the dataset JSON
def generate_dataset_json(original_files, response_files, output_file, download_dir):
    # Create requestID to URL Dictionary
    ID_to_URL_and_prompt = {}

    for original_file in original_files:
        with open(original_file, 'r') as of:
            for line in of:
                original_data = json.loads(line)
                custom_id = original_data.get("custom_id")
                prompt = original_data.get("body", {}).get("messages", [])[0].get("content", [])[0].get("text", "")
                image_url = extract_image_url(original_data)
                ID_to_URL_and_prompt[custom_id] = {"prompt": prompt, "image_url": image_url}

    dataset = []

    for input_file in response_files:
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                response_content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                if response_content.lower().strip() == "not a transcript":
                    continue

                custom_id = data.get("custom_id")
                prompt_info = ID_to_URL_and_prompt.get(custom_id)
                if not prompt_info:
                    continue

                prompt = prompt_info["prompt"]
                image_url = prompt_info["image_url"]

                local_path = download_images([image_url], download_dir)[0]
                print(f"Downloaded image from {local_path}")
                if not local_path:
                    continue

                dataset.append({
                    "id": custom_id,
                    "image": local_path,
                    "conversations": [
                        {"role": "user", "content": f"<image>\n{prompt}"},
                        {"role": "assistant", "content": response_content}
                    ]
                })

    with open(output_file, 'w') as out_f:
        json.dump(dataset, out_f, indent=4)

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    # original_files = [
    #     "Synthetic_Image_Annotation/Input_JSONL/batchinput.jsonl"
    # ]

    # response_files = [
    #     "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_1.jsonl",
    #     "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_2.jsonl",
    #     "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_3.jsonl",
    #     "Synthetic_Image_Annotation/Output_JSONL/batchoutput_part_4.jsonl",
    # ]

    # output_file = "Synthetic_Image_Annotation/Cleaned_JSONL/cleaned_transcripts.json"
    # download_dir = "Synthetic_Image_Annotation/images/"

    # generate_dataset_json(original_files, response_files, output_file, download_dir)

    original_files = [
        "Synthetic_Image_Annotation/Test_Data/Input_JSONL/input.jsonl"
    ]

    response_files = ["Synthetic_Image_Annotation/Test_Data/Output_JSONL/batch_mNvrFzYMUglQNiBy9OYZoMpY_output.jsonl"]
    # Output file for cleaned data
    output_file = "Synthetic_Image_Annotation/Test_Data/cleaned_transcripts.json"
    download_dir = "Synthetic_Image_Annotation/Test_Data/images/"

    generate_dataset_json(original_files, response_files, output_file, download_dir)
