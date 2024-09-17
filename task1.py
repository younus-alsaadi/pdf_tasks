import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PyPDF2 import PdfReader

# Path to Tesseract binary
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  #for mac-os


#  preprocess the image for better contrast and edge detection accuracy
def preprocess_image(image):
    """
    Preprocesses the image for better line detection accuracy by converting to grayscale,
    applying bilateral filtering, resizing, and adaptive thresholding.

    Args:
        image: The input image (as a numpy array).

    Returns:
        Processed image suitable for edge detection and Hough Transform.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filtering for noise reduction while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Resize the image to make the text larger and clearer (scaling up by 1.5 times)
    resized = cv2.resize(filtered, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Apply adaptive thresholding for better binarization
    binary_image = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    return binary_image


# detect rotation using Hough Line Transform
def detect_rotation_using_hough(image):
    """
    Detect the rotation angle of the page using the Hough Line Transform.

    Args:
        image: the page image.

    Returns:
        An integer representing the rotation angle in degrees (normalized [0, 359]).
    """
    # Detect edges in the image using Canny Edge Detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta)  # Convert theta from radians to degrees
            if angle > 45:  # Normalize angle to [-45, 45] degrees
                angle -= 90
            angles.append(angle)

    # If lines are detected, calculate the median angle of the lines
    if len(angles) > 0:
        detected_angle = int(np.median(angles))  # Return the median angle from detected lines
    else:
        detected_angle = 0  # Assume upright if no lines are detected

    # Normalize the detected angle to [0, 359] for clockwise rotation correction
    rotation_angle = (360 - detected_angle) % 360

    return rotation_angle


# Fallback function: Detect rotation using Tesseract OSD (Orientation and Script Detection)
def detect_rotation_using_osd(image):
    """
    Detect the rotation of the page using Tesseract's Orientation and Script Detection (OSD).

    Args:
        image: A numpy array representing the page image.

    Returns:
        An integer representing the rotation angle in degrees (normalized to [0, 359]).
    """
    try:
        osd_result = pytesseract.image_to_osd(image)
        angle = int(osd_result.split("\n")[2].split(":")[1].strip())  # Extract the rotation angle
    except pytesseract.TesseractError as e:
        #print(f"OSD Detection Error: {e}")
        angle = 0  # If OCR OSD fails, assume the page is upright

    return angle


# Function to manually rotate the image by 90, 180, and 270 degrees, then apply Hough Transform
def manual_rotation_and_check(image):
    """
    Manually rotate the image by 90, 180, and 270 degrees and reapply Hough Line Transform to detect rotation.

    Args:
        image: A numpy array representing the page image.

    Returns:
        An integer representing the rotation angle in degrees (normalized to [0, 359]).
    """
    rotation_angles = [0, 90, 180, 270]  # Degrees to rotate and check

    for angle in rotation_angles:
        # Rotate the image
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated_image = image  # No rotation for 0 degrees

        #  try detecting the rotation using Hough Line Transform again
        processed_rotated_image = preprocess_image(rotated_image)
        detected_angle = detect_rotation_using_hough(processed_rotated_image)

        if detected_angle != 0:
            #print(f"Manual Rotation worked at {angle} degrees.")
            return angle  # Return the rotation angle where detection worked

    # If no valid rotation was found, assume upright
    return 0


# Main function to detect rotation angles for non-machine-readable pages
def detect_rotation_all_pages(input_pdf: str):
    """
    Analyze all pages in the input PDF and detect the rotation for non-machine-readable pages.

    Args:
        input_pdf: Path to the input PDF.

    Returns:
        List[int]: A list of rotation angles (in degrees) for each page.

    """
    reader = PdfReader(input_pdf)
    pages = convert_from_path(input_pdf, dpi=400)  # Increased DPI to 400 for better resolution

    rotation_angles = []

    for page_number in range(len(reader.pages)):
        current_page = reader.pages[page_number]
        #print(f"Processing Page {page_number + 1}:")

        #Check if the page is machine-readable
        extracted_text = current_page.extract_text()
        if extracted_text and len(extracted_text.strip()) > 31:
            #print(f"Page {page_number + 1} is machine-readable, no rotation detection needed.")
            rotation_angles.append(0)  # No rotation needed for machine-readable pages
        else:
            #print(f"Page {page_number + 1} is not machine-readable, applying Hough Line Transform.")

            # Convert the page image to numpy array for OpenCV
            page_image = np.array(pages[page_number])
            page_image_np = page_image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV

            # Preprocess the image for rotation detection
            processed_image = preprocess_image(page_image_np)

            # Detect rotation using Hough Line Transform
            rotation_angle = detect_rotation_using_hough(processed_image)

            # If Hough Line Transform fails (returns 0), fallback to OCR OSD
            if rotation_angle == 0:
                #print(f"Hough Line Transform failed for page {page_number + 1}, trying OCR OSD.")
                rotation_angle = detect_rotation_using_osd(page_image_np)

            # If both Hough Line Transform and OCR OSD fail, apply manual rotation and retry
            if rotation_angle == 0:
                #print(f"OSD also failed for page {page_number + 1}. Trying manual rotations (90, 180, 270 degrees).")
                rotation_angle = manual_rotation_and_check(page_image_np)

            # Normalize the rotation angle to be in range [0, 359]
            rotation_angle = (rotation_angle + 360) % 360
            #print(f"Detected rotation angle for page {page_number + 1}: {rotation_angle} degrees.")

            rotation_angles.append(rotation_angle)

    return rotation_angles


# Example usage
input_pdf = 'grouped_documents.pdf'  # Path to your PDF file
rotation_angles = detect_rotation_all_pages(input_pdf)
print(f"Rotation angles for each page: {rotation_angles}")
