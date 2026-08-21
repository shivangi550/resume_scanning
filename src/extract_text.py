import pymupdf
import os


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF resume.
    """

    document = pymupdf.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text

if __name__ == "__main__":

    pdf_path = "data/resumes/resume1.pdf"

    text = extract_text_from_pdf(pdf_path)

    print(text)