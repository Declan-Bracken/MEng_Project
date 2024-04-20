from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import geckodriver_autoinstaller
import time
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class SlideImageFinder:
    def __init__(self):
        # Automatically install geckodriver if not present
        geckodriver_autoinstaller.install()
        
        # Set up Firefox options
        firefox_options = Options()
        firefox_options.add_argument("--headless")  # Runs Firefox in headless mode.
        
        # Initialize the Firefox WebDriver
        print(" ***STARTING WEBDRIVER...***")
        self.driver = webdriver.Firefox(options=firefox_options)

    def has_slides(self, url):
        
        if 'slideshare.net' not in url:
            return False
        
        self.driver.get(url)
        time.sleep(0.5)  # Give some time for the page and its JavaScript to load
        # Attempt to find a unique element that identifies a slideshow
        # Replace 'slideshow_container' with the actual CSS selector for the slideshow container or element
        return len(self.driver.find_elements(By.CSS_SELECTOR, '#new-player')) > 0

    def find_all_slides(self, url):
        slides = []
        num_loaded = 0

        # Instantiate driver
        try:
            # Try to navigate to the page
            self.driver.get(url)
        except TimeoutException:
            # If a timeout occurs, log it and return the slides that have been loaded so far
            print(f"Timeout occurred when loading the page: {url}")
            return slides

        for n in range(20): # Use 20 max scrolls
            # Scroll down by one window height
            self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
            time.sleep(0.5)  # Give time for images to load

            # Find all images that match the pattern
            slide_elements = self.driver.find_elements(By.CSS_SELECTOR, "[id^='slide-image-']")

            if num_loaded == len(slide_elements):
                break

            for slide_element in slide_elements[num_loaded:]:  # Only process slides which have not loaded in properly on past iterations
                srcset = slide_element.get_attribute('srcset')
                # Split into list based on commas, then select the last URL (assuming it's the highest resolution)
                if srcset:
                    image_urls = [url.strip().split(' ')[0] for url in srcset.split(',')]
                    high_res_image_url = image_urls[-1]  # Take the last image URL, assuming it's the largest
                    if high_res_image_url.startswith('http'):
                        # append loaded url to slide list
                        slides.append(high_res_image_url)
                        # increment the number loaded
                        num_loaded += 1
        
        if n == 19:
            print("Max number of scrolls reached, continuing...")
        # return high quality slide URLs
        return slides
    
    # for exiting the driver
    def exit_driver(self):
        print(" ***CLOSING WEBDRIVER...***")
        self.driver.quit()

if __name__ == "__main__":
    # Example usage
    finder = SlideImageFinder()
    urls = finder.find_all_slides("https://www.slideshare.net/JeremyChapman6/undergraduate-transcript-63145056")
    finder.exit_driver()

    for url in urls:
        print(url)
