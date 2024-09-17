import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PyPDF2 import PdfReader
import re
from collections import defaultdict

# Set path to Tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # for mac-os

# preprocess the image for better OCR and visual feature detection
def preprocess_image(image):
    """
    Preprocess the image for visual analysis by converting to grayscale and applying thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary_image

# Detect visual features like colored borders, watermarks, or headers/footers
def detect_visual_features(image):
    """
    Detects colored borders, headers/footers, watermarks, and layout changes.

    Args:
        image: The input image.

    Returns:
        dict: A dictionary of detected visual features.
    """
    features = {
        'colored_border': False,
        'watermark': False,
        'header_footer': False,
    }

    # Check for colored borders (by checking color distribution in borders)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    border_thickness = 50  # For detecting colored borders
    top_border = mask[:border_thickness, :]
    bottom_border = mask[-border_thickness:, :]

    if np.count_nonzero(top_border) > 100 or np.count_nonzero(bottom_border) > 100:
        features['colored_border'] = True

    # Check for headers/footers by analyzing the top and bottom regions
    height, width, _ = image.shape
    top_region = image[:int(height * 0.1), :]
    bottom_region = image[-int(height * 0.1):, :]

    top_text = cv2.countNonZero(preprocess_image(top_region))
    bottom_text = cv2.countNonZero(preprocess_image(bottom_region))

    if top_text > 100 or bottom_text > 100:
        features['header_footer'] = True

    # Optional: Check for watermarks using Fourier Transform
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    if np.median(magnitude_spectrum) < 100:  # A rough heuristic to detect patterns like watermarks
        features['watermark'] = True

    return features

# Extract text from image using OCR (Tesseract)
def extract_text_with_ocr(image):
    """
    Use OCR to extract text from an image.
    """
    text = pytesseract.image_to_string(image)
    return text.strip()

# Main function to partition the PDF based on text and visual features, with OCR fallback
def partition_the_pdf_document(input_pdf: str):
    reader = PdfReader(input_pdf)
    pages = convert_from_path(input_pdf, dpi=300)  # Convert PDF to images for analysis

    document_groups = defaultdict(list)
    current_document = None
    document_counter = 1

    # extract document metadata (e.g., 'Document X')
    document_pattern = re.compile(r"(Document \d+)")

    for page_number in range(len(reader.pages)):
        current_page = reader.pages[page_number]
        page_image = np.array(pages[page_number])[:, :, ::-1].copy()  # Convert to BGR

        # First  extract text from the PDF page
        extracted_text = current_page.extract_text()

        # If text extraction fails, try OCR
        if not extracted_text or len(extracted_text.strip()) == 0:
            extracted_text = extract_text_with_ocr(page_image)

        # Check for identifiable text-based document metadata
        if extracted_text:
            match = document_pattern.search(extracted_text)
            if match:
                current_document = match.group(1)
            else:
                current_document = f"Document {document_counter}"
                document_counter += 1
        else:
            # If no text is found, fallback to visual analysis to detect transitions
            visual_features = detect_visual_features(page_image)
            if visual_features['colored_border'] or visual_features['header_footer'] or visual_features['watermark']:
                current_document = f"Document {document_counter}"
                document_counter += 1

        # Group pages under the same document
        document_groups[current_document].append(page_number + 1)

    return dict(document_groups)

# Example usage
input_pdf = "grouped_documents.pdf"  # Input PDF file
partitions = partition_the_pdf_document(input_pdf)
print(f"Document partitions: {partitions}")
