Description:

You are contributing to the development of a pipeline to parse medical documents within an archive. All documents of a single patient are appended in one file in a random order. The documents can have the following origins: scans of printed documents (e.g. doctor letters, medication plans, printed lab results), scans of medical images (e.g. sonography-images or electrocardiograms [ECG]), programatically generated PDF-documents (e.g. letters, reports, lab reports, ECGs, etc.) which were sent to the archive. It is possible that documents appear as single files (one document, single file) but it can also appear that a collection of paper documents (e..g 3 letters, lab results, medication plans) were scanned into one file and uploaded into a single file (multiple documents, single file).



task1 :A function which takes a PDF file path as inputs and returns a dictionary which page has to be rotated by which angle to be upright (OCR-parsable). Documents can be generated PDFs (with embedded text) but are especially scanned pages. The documents may be landscape or portrait oriented. The function is performant also for documents up to 200 pages.

tasks 2:A function which classifies pages of a 200 page document into 3 categories:
machine-readable PDF / searchable PDF (e.g. internally generated discharge letter)
Image-based PDF which may be OCR’d (e.g. discharge letters brought by the patient from an external hospital and then scanned)
Image-based PDF which may not be OCR`d (e.g. ECG)
Cherry-on-the-cake task:

Provide a function which takes a filepath to a 200 page long PDF-file with mostly scanned and a few generated PDF documents. The file contains no index or table of contents. Your function has to return a list of extracted PDF pages which belong to one document, e.g. pages 1-3 extracted ⇒ doctor letter, page 4 ⇒ medication plan, pages 5-13 again a doctor’s letter, pages 13-21 another doctor’s letter.
