import pdfplumber
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)

def extract_text(uploaded_file):
    """
    Extract all text from a PDF. Returns empty string if no text found.
    Works with Streamlit UploadedFile objects directly.
    """
    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

    return text.strip()