import os
from pypdf import PdfReader


def extract_text_from_pdfs(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text
