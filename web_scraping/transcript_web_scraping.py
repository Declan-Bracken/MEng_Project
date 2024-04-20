import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import cv2
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime as dt
from selenium.webdriver.common.keys import Keys

import time

class WebImageExtractor:
    def __init__(self, search_query, num_images, chromedriver_path):
        self.search_query = search_query
        self.num_images = num_images
        self.chromedriver_path = chromedriver_path
        self.image_urls = []
        print("---WEB EXTRACTOR INSTANTIATED---")

    def fetch_image_urls(self):
        search_url = f"https://www.google.com/search?q={self.search_query}&tbm=isch"
        driver = webdriver.Chrome()  # You need to have chromedriver installed. You can also use Firefox or other browsers.
        driver.get(search_url)
        time.sleep(5)  # Allow time for the page to load

        # Scroll to the bottom of the page to load more images (you can adjust the number of scrolls if needed)
        for _ in range(5):
            driver.find_element('xpath', '//body').send_keys(Keys.END)
            time.sleep(2)  # Wait for more images to load

        # Find image elements and extract URLs
        img_elements = driver.find_elements('xpath', '//img[@class="t0fcAb"]')

        for img_element in img_elements:
            img_url = img_element.get_attribute('src')
            if img_url:
                self.image_urls.append(img_url)

        driver.quit()  # Close the browser after scraping

    def download_images(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, url in enumerate(self.image_urls):
            response = requests.get(url)
            with open(f'{output_folder}/image_{i}.jpg', 'wb') as f:
                f.write(response.content)

    def display_images(self, dataset_folder):
        images = os.listdir(dataset_folder)
        for i, image_name in enumerate(images):
            image_path = os.path.join(dataset_folder, image_name)
            image = cv2.imread(image_path)
            cv2.imshow(f'Image {i+1} (Press y to keep, any other key to remove)', image)
            key = cv2.waitKey(0)
            if key == ord('y'):
                print(f'Image {i+1} kept.')
            else:
                os.remove(image_path)
                print(f'Image {i+1} removed.')
            cv2.destroyAllWindows()

if __name__ == "__main__":
    PATH = "/Applications/chromedriver-mac-arm64/chromedriver"
    search_query = input("Enter search query: ")
    num_images = int(input("Enter the number of images to extract: "))

    extractor = WebImageExtractor(search_query, num_images, PATH)
    extractor.fetch_image_urls()
    print("Image URLs:")
    print(extractor.image_urls)