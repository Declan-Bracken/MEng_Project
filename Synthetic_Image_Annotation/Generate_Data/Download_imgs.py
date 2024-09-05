import os
import json
import requests
from pathlib import Path

def download_images(image_urls, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    local_paths = []
    for url in image_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Extract image filename from the URL
            image_name = url.split('id=')[1] + ".jpg"  # Modify this if URL format is different
            local_path = os.path.join(download_dir, image_name)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            local_paths.append(local_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            local_paths.append(None)
    return local_paths