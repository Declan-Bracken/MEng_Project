import requests
from bs4 import BeautifulSoup

class ImageFinder:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    def find_primary_image(self, url):
        """Find the primary image of a webpage by checking Open Graph tags and then evaluating image sizes."""
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the image using Open Graph meta tag
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image['content']:
            return og_image['content']

        # If no Open Graph image, fallback to searching for the largest image
        images = soup.find_all('img')
        if not images:
            return None

        # Filter out icons or very small images
        images = [img for img in images if 'icon' not in img.get('src', '').lower() and 'logo' not in img.get('src', '').lower()]

        max_size = 0
        primary_image = None
        for img in images:
            try:
                image_url = self.resolve_image_url(img['src'], url)
                img_response = requests.get(image_url, headers=self.headers)
                size = len(img_response.content)
                if size > max_size:
                    max_size = size
                    primary_image = image_url
            except Exception as e:
                print(f"Error downloading image {img['src']}: {e}")

        return primary_image

    def resolve_image_url(self, src, base_url):
        """Resolve the complete image URL."""
        if src.startswith('//'):
            return 'http:' + src
        elif src.startswith('/'):
            return f"{base_url}{src}"
        return src

if __name__ == "__main__":
    # Create an instance of the ImageFinder class
    image_finder = ImageFinder()

    # URL of the webpage to scrape
    url = "https://www.binghamton.edu/registrar/student/transcripts/transcript-key.html"

    # Find the primary image
    primary_image_url = image_finder.find_primary_image(url)
    print("Primary Image URL:", primary_image_url)