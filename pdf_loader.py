from pdfminer.high_level import extract_text

def load_pdf_text(path):
    return extract_text(path)