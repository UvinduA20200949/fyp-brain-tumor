from datetime import datetime
import os
import cv2

def convert_to_black_and_white(input_image_path):
    filename = os.path.basename(input_image_path)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    output_image_path = f"backend/output/{filename}-bnw-{time}.jpg"
    # Read the image in color
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary using a threshold
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Save the black and white image
    cv2.imwrite(output_image_path, black_and_white_image)

    print(f"Black and white image is saved as {output_image_path}")

# # Example usage
# input_image_path = 'backend/input/PXL_20230615_045153562.jpg'
# output_image_path = 'backend/output/PXL_20230615_045153562-bnw.jpg'
# convert_to_black_and_white(input_image_path, output_image_path)
