import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
import tempfile
from typing import List

# Path to Tesseract binary - update based on your installation
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # For macOS, adjust if necessary


# Function to preprocess the image for OCR
def preprocess_image_for_ocr(image):
    """
    Preprocess the image for OCR by converting to grayscale and applying thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better text recognition
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image


# Function to save a single PDF page to a temporary file and return its path
def save_single_page_as_pdf(page, page_number):
    """
    Some pages need to be processed separately. The function takes one PdfReader page object,
    saves it as a temporary PDF file, and returns the path of that file.
    We will do that so we treat every page separately during the OCR process.

    """
    writer = PdfWriter()
    writer.add_page(page)

    # Create a temporary PDF file
    temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_page_{page_number}.pdf')
    with open(temp_pdf_file.name, 'wb') as f:
        writer.write(f)

    return temp_pdf_file.name


# classify a PDF page into one of three categories
def classify_page(page, page_number) -> int:
    """
    Classify a PDF page into one of three categories:

    0: Machine-readable / searchable (if extracted text length > 31 characters)
    1: Non-machine-readable but OCR-able (if extracted text length <= 31 characters)
    2: Non-machine-readable and not OCR-able (if both PyPDF2 and OCR fail)
    """
    # Attempt to extract text directly using PyPDF2
    extracted_text = page.extract_text()

    # for Debug
    #print(f"Extracted text from page {page_number + 1} (length {len(extracted_text.strip()) if extracted_text else 0}): {repr(extracted_text)}")

    # If extracted text length is greater than 31 characters, classify as machine-readable, if less could be metedata
    if extracted_text and len(extracted_text.strip()) > 31:
        return 0  # Machine-readable

    # If text is short or not meaningful, fallback to OCR
    try:
        # Save the single page as a temporary PDF
        temp_pdf_path = save_single_page_as_pdf(page, page_number)

        # Convert the single page PDF to an image
        page_image = convert_from_path(temp_pdf_path, dpi=300)[0]
        page_image_np = np.array(page_image)  # Convert PIL image to NumPy array

        # Preprocess image for OCR
        processed_image = preprocess_image_for_ocr(page_image_np)

        # Apply OCR to the preprocessed image
        ocr_text = pytesseract.image_to_string(processed_image)

        # for Debug
        #print(f"OCR extracted text from page {page_number + 1} (length {len(ocr_text.strip())}): {repr(ocr_text)}")

        # If OCR successfully extracts text, classify as OCR-able
        if len(ocr_text.strip()) > 0:
            return 1  # OCR-able but not machine-readable
    except Exception as e:
        print(f"Error processing page {page_number + 1} for OCR: {e}")

    # If no text is found even with OCR, classify as not OCR-able
    return 2  # Non-machine-readable and not OCR-able


# Main function to classify all pages of a PDF
def classify_all_pages(input_pdf: str) -> List[int]:
    """
    Analyze all pages in the input PDF and classify each page.
    """
    reader = PdfReader(input_pdf)
    classes = []

    for page_number in range(len(reader.pages)):
        current_page = reader.pages[page_number]
        page_class = classify_page(current_page, page_number)
        classes.append(page_class)

    return classes


# Example usage
input_pdf = 'grouped_documents.pdf'
page_classes = classify_all_pages(input_pdf)
print(f"Classes for each page: {page_classes}")
