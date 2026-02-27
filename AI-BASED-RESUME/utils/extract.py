import pdfplumber
import docx

def extract_text(path):
    """
    Extract text from PDF or DOCX file safely.
    """
    text = ""
    try:
        if path.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        elif path.lower().endswith(".docx"):
            doc = docx.Document(path)
            text = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
    return text.strip()
