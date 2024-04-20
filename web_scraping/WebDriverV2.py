import requests
import os
import io
import cv2
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from datetime import datetime as dt
import time

PATH = "/Applications/chromedriver-mac-arm64/chromedriver"

wd = webdriver.Chrome(PATH)

# Get and list images from google

def get_images_from_google(wd, delay, max_images, url):
    def scroll_down(wd):
        wd.execute_script('window.scrollTo(0,document.body.scrollHeight);')
        time.sleep(delay)

    url = url
    wd.get(url)

    # create empty set
    image_urls = set()
    skips = 0
    classy = "MXKnqf"
    while len(image_urls) + skips < max_images:
        scroll_down(wd)
        thumbnails = wd.find_elements(By.CLASS_NAME, classy)
    
        for img in thumbnails[len(image_urls) + skips:max_images]:
            try:
                #try to click on image to access path
                img.click()
                time.sleep(delay)
            except:
                continue
            link_add = 'https://www.google.com/imgres?imgurl=https%3A%2F%2Fregistrar.buffalo.edu%2Fhub%2Fimages%2FunoffTran.png&tbnid=ZUPWJ7fGWXVc1M&vet=12ahUKEwi6luLAopCFAxVowRQJHXKjCbsQMygBegQIARBS..i&imgrefurl=https%3A%2F%2Fregistrar.buffalo.edu%2Fhub%2FviewUnoffTranscript.php&docid=KFp8qeBermcbyM&w=675&h=523&q=undergraduate%20unofficial%20transcript&ved=2ahUKEwi6luLAopCFAxVowRQJHXKjCbsQMygBegQIARBS'
            images = wd.find_elements(By.CLASS_NAME, '')
            