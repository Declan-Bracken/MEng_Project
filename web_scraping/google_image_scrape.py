import requests
from image_finder import ImageFinder # Custom Class
from dynamic_image_finder import SlideImageFinder # Custom Class
from tqdm import tqdm
from save_to_drive import GoogleDriveManager # Custom Class
import io
import requests
from googleapiclient.http import MediaIoBaseUpload

def scrape_images(search_query, cx, api_key, imgSize, filter, total_results_required, start_index = 1):
    slide_image_finder = SlideImageFinder()
    image_finder = ImageFinder()

    # remember original starting index
    start_index0 = start_index

    image_urls = []
    image_site_urls = []

    while start_index <= total_results_required + start_index0:
        if (total_results_required + start_index0 - start_index) < 10:
            search_num = (total_results_required - start_index)
        else:
            search_num = 10
        
        print(f"-----------------------------------------------------------------------\n \
              PROCESSING NEW IMAGE SET... index = {start_index - start_index0} \
              \n-----------------------------------------------------------------------")

        url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&cx={cx}&searchType=image&key={api_key}&imgSize={imgSize}&filter={filter}&num={search_num}&start={start_index}"
        response = requests.get(url)

        if response.status_code == 200:
            search_results = response.json()
            items = search_results.get("items", [])

            for image in tqdm(items):
                # Get the website url which posted the image
                original_image_url = image.get('image', {}).get('contextLink')
                if original_image_url:
                    # Record original site url
                    image_site_urls.append(original_image_url)
                    print("Context Link:", original_image_url)

                    # Check for slideshow:
                    if slide_image_finder.has_slides(original_image_url):
                        # If slideshow is present, get image urls for each slide
                        slide_image_urls = slide_image_finder.find_all_slides(original_image_url)
                        image_urls.extend(slide_image_urls)
                        print(f"Number of Images: {len(slide_image_urls)}")
                        # print(f"Slide Image Urls: {slide_image_urls}")
                    else:
                        # Get primary image
                        primary_image_url = image_finder.find_primary_image(original_image_url)
                        image_urls.append(primary_image_url)
                        print(f"Number of Images: {len(image_urls)}")
                        # print(f"Image Url: {primary_image_url}")

            start_index += len(items)  # Increment the start_index by the number of items returned
            if not items:
                break  # No more results were returned; exit the loop
        else:
            print(f"Failed to retrieve data: {response.status_code}, {response.text}")
            break

    slide_image_finder.exit_driver()
    return image_urls, image_site_urls

def upload_image_to_drive(gdrive_manager, image_url, folder_id, image_num):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # Prepare the file for upload as a byte stream
            file_metadata = {
                'name': f'image_{image_num}.jpg',
                'parents': [folder_id]
            }
            
            # Convert the streamed content to a byte stream
            fh = io.BytesIO(response.content)
            
            # Upload the byte stream to Google Drive
            media = MediaIoBaseUpload(fh, mimetype='image/jpeg', resumable=True)
            file = gdrive_manager.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Image {image_num} uploaded: {image_url} with file ID: {file.get('id')}")
            return file.get('id')
        else:
            print(f"Failed to download image {image_num}: {image_url} with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"RequestException while downloading image {image_num}: {image_url}. Error: {e}")

if __name__ == "__main__":
    search_query = "university undergraduate transcript site:slideshare.net"
    api_key = "AIzaSyAHKp3QovX78D8-RjPyiBZ3HM-4nvcflRM"
    cx = "868233c9494a2462a"
    imgSize = "imgSizeUndefined"
    filter = '1'
    total_results_required = 50
    starting_index = 11

    scraped_image_urls, scraped_image_site_urls = scrape_images(search_query, cx, api_key, imgSize, 
                                                                filter, total_results_required, 
                                                                start_index = starting_index)

    # Instantiate GoogleDriveManager with the path to your credentials
    gdrive_manager = GoogleDriveManager()
    folder_key = "14zyq0BXTYrYj81bGlKtYpEG-KL59oNnM" # key to gdrive folder

    # Download each image and upload to Google Drive
    for idx, image_url in enumerate(scraped_image_urls):
        upload_image_to_drive(gdrive_manager, image_url, folder_key, idx)

# For Saving Locally:
    
# Function to download an image from a URL
# def download_image(image_url, save_path, image_num):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         # Open a local file with wb (write binary) permission.
#         with open(os.path.join(save_path, f"image_{image_num}.jpg"), 'wb') as f:
#             f.write(response.content)
#         print(f"Image {image_num} downloaded: {image_url}")
#     else:
#         print(f"Failed to download image {image_num}: {image_url}")

# # Specify the directory to save the images
# save_directory = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts_v2"
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)

# # Download each image
# for idx, url in enumerate(image_urls):
#     download_image(url, save_directory, idx)