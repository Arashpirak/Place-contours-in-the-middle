import os
import cv2 as cv
import numpy as np

# Define some color constants in BGR format
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# This code takes images from the specified directory, finds contours in the images,
# and then saves two versions of each image: one with contours drawn on it and one without,
# both centered on a black background. It was written by Arash Pirak as a practice assignment
# for Mr. Mesbah's class on 07.Agu.2024 and is an initial version that needs improvements.

def find_external_contours(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def find_contours(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def find_center(contour):
    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

def thresh(img, threshold):
    gray_img = gray(img)
    ret, thresh = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY_INV)
    return thresh

def gaussian_blur(img, size, sigma):
    return cv.GaussianBlur(img, (size, size), sigma)

def gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def draw_contour(img, contour):
    return cv.drawContours(img, [contour], 0, RED, 2)

def draw_all_contours(img, contours):
    return cv.drawContours(img, contours, -1, RED, 2)

def find_bounding_rect(contours):
    x, y, w, h = cv.boundingRect(np.concatenate(contours))
    return x, y, w, h

def place_in_center_of_black_background(image, x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    
    # Create a black background image
    background = np.zeros_like(image)
    
    # Calculate the start positions to place the image in the center
    start_x = background.shape[1] // 2 - w // 2
    start_y = background.shape[0] // 2 - h // 2
    
    # Place the image on the black background
    background[start_y:start_y + h, start_x:start_x + w] = image[y:y + h, x:x + w]
    
    return background

# Specify the input image path and output directory
image_path = r'C:python/images'
output_dir = r'C:python/images/resized_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Walk through the directory and process each image file
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(root, file)
            image = cv.imread(file_path)
            if image is None:
                print(f"Failed to load image: {file_path}")
                continue
            
            # Make a copy of the original image
            original_image = image.copy()
            
            gray_image = gray(image)
            contours = find_external_contours(gray_image)
            
            if contours:
                # Draw contours on the copy
                image_with_contours = original_image.copy()
                draw_all_contours(image_with_contours, contours)
                x, y, w, h = find_bounding_rect(contours)
                cv.rectangle(image_with_contours, (x, y), (x + w, y + h), BLUE, 2)
                centered_image = place_in_center_of_black_background(image_with_contours, x, y, w, h)
                
                # Save the image with contours
                output_file_with_contours = f"{os.path.splitext(file)[0]}_with_contours{os.path.splitext(file)[1]}"
                output_path_with_contours = os.path.join(output_dir, output_file_with_contours)
                cv.imwrite(output_path_with_contours, centered_image)
                
                # Save the original image without contours
                centered_image_no_contours = place_in_center_of_black_background(original_image, x, y, w, h)
                output_file_no_contours = f"{os.path.splitext(file)[0]}_center{os.path.splitext(file)[1]}"
                output_path_no_contours = os.path.join(output_dir, output_file_no_contours)
                cv.imwrite(output_path_no_contours, centered_image_no_contours)
                
                # Display the image for visual inspection
                cv.imshow('Image', centered_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
