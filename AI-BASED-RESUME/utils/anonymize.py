import spacy
import re

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def anonymize(text):
    """
    Anonymize sensitive information in text:
    - PERSON, GPE, ORG
    - Emails
    - Phone numbers (10+ digits)
    """
    if not text:
        return ""

    doc = nlp(text)
    result = text

    # Replace named entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            result = re.sub(re.escape(ent.text), "[REDACTED]", result)

    # Replace emails
    result = re.sub(r'\b\S+@\S+\b', "[REDACTED]", result)
    # Replace phone numbers (10+ digits)
    result = re.sub(r'\b\d{10,}\b', "[REDACTED]", result)

    return result
