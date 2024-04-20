import os
import cv2
from pytesseract import Output
import pytesseract
import imutils

class DetectOrientation:
    def __init__(self, save_directory):
        # construct the argument parser
        self.save_directory = save_directory
        self.print_orientation = False

    def load_image(self, image_path):
        try:
            # Load the image
            image = cv2.imread(image_path)

            # Check if image loaded successfully and has valid dimensions
            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                raise ValueError("Invalid image dimensions")

            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
    def find_orientation(self, image_path):

        # load image from path
        image = self.load_image(image_path)

        # Check image condition
        if image is None:
            return "Unable to load image"
        
        try:
            # Convert the image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use pytesseract to determine orientation
            results = pytesseract.image_to_osd(rgb_image, output_type=Output.DICT)

        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error determining orientation"
        print(self.print_orientation)
        if self.print_orientation:
            # display the orientation information
            print("[INFO] detected orientation: {}".format(
                results["orientation"]))
            print("[INFO] rotate by {} degrees to correct".format(
                results["rotate"]))
            print("[INFO] detected script: {}".format(results["script"]))
        return image, results
    
    # save image
    def save_image(self, file_name, rotated_image):
        cv2.imwrite(self.save_directory + file_name, rotated_image)

    
    def auto_orient(self, image_path, print_orientation = False, save_image = False):

        # update print statement conditional
        self.print_orientation = print_orientation

        # find image orientation
        image, results = self.find_orientation(image_path)

        # rotate the image to correct the orientation
        rotated = imutils.rotate_bound(image, angle=results["rotate"])

        # show the original image and output image after orientation
        # correction
        if self.print_orientation:
            cv2.imshow("Original", image)
            cv2.imshow("Output", rotated)
            cv2.waitKey(0)

        if save_image:
            # Extract the filename from the image_path
            filename = os.path.basename(image_path)
            # Split the filename and extension
            name, extension = os.path.splitext(filename)
            output_filename = name + '_corrected' + extension
            # Construct the output path by joining output folder and output filename
            output_path = os.path.join(self.save_directory, output_filename)  

            # save image
            self.save_image(output_path, rotated)
        return
    

if __name__ == '__main__':

    # Directory containing images
    # input_dir = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Sample Transcripts/Actual Sample Transcripts'
    input_dir = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Sample Transcripts/Actual Sample Transcripts/1.JPG'
    output_dir = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Sample Transcripts/Preprocessed_Images'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create class instance
    detect_orientation = DetectOrientation(output_dir)

    detect_orientation.auto_orient(input_dir, print_orientation = True, save_image = False)

    # Iterate over images in the input directory
    # for filename in os.listdir(input_dir):
    #     if filename.endswith('.jpg') or filename.endswith('.png'):
            


