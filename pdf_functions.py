from pdfminer.high_level import extract_text

def extract_text_from_pdf(uploaded_file):
    pdf_text = extract_text(uploaded_file)
    return pdf_text
