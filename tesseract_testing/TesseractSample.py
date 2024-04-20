#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:35:28 2024

@author: declanbracken
"""
from PIL import Image
import pytesseract
import numpy as np

filename = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/TeddyRooseveltTranscript.png'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

# import fitz  # PyMuPDF

# # Open the PDF file
# pdf_document = '/Users/declanbracken/Development/UofT_Projects/Meng Project/UAlbertaTranscript_DB.pdf'
# pdf = fitz.open(pdf_document)

# # Read each page's text
# for page_num in range(len(pdf)):
#     page = pdf.load_page(page_num)
#     text = page.get_text()
#     print(f"Page {page_num + 1}:\n{text}")

# # Close the PDF after processing
# pdf.close()