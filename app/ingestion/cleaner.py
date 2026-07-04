import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    # keep readable utf text only
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # remove strange symbols
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove tiny junk words
    words = [w for w in text.split() if len(w) > 1]

    return " ".join(words)