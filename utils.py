import os
import re
import requests
from datasets import load_dataset, Dataset
import nltk

# NLTK für Satzsegmentierung sicherstellen
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# URLs der Shakespeare-Werke
shakespeare_urls = {
    "Hamlet": "https://www.gutenberg.org/files/1524/1524-0.txt",
    "Macbeth": "https://www.gutenberg.org/files/1533/1533-0.txt",
    "Romeo and Juliet": "https://www.gutenberg.org/files/1112/1112-0.txt",
    "Othello": "https://www.gutenberg.org/files/1531/1531-0.txt"
}

def download_and_prepare_datasets(bookcorpus_limit=1000):
    """
    Lädt und bereitet den BookCorpus und die Shakespeare-Texte vor.
    Wenn die Daten bereits vorhanden sind, werden sie nicht erneut heruntergeladen.
    """

    # 1️⃣ BookCorpus herunterladen oder laden
    bookcorpus_file = "Bookcorpus_dataset.txt"
    if not os.path.exists(bookcorpus_file):
        print("Lade BookCorpus-Daten herunter...")
        dataset = load_dataset("bookcorpus", trust_remote_code=True)
        sentences = dataset["train"]["text"][:bookcorpus_limit]
        with open(bookcorpus_file, "w", encoding="utf-8") as file:
            for sentence in sentences:
                file.write(sentence + "\n")
        print(f"BookCorpus-Daten wurden erfolgreich gespeichert in '{bookcorpus_file}'.")
    else:
        print(f"BookCorpus-Daten bereits vorhanden in '{bookcorpus_file}'.")
        with open(bookcorpus_file, "r", encoding="utf-8") as file:
            sentences = [line.strip() for line in file.readlines()]

    # 2️⃣ Shakespeare-Texte herunterladen oder laden
    shakespeare_folder = "Shakespeare_Texts"
    if not os.path.exists(shakespeare_folder):
        os.makedirs(shakespeare_folder)
        for title, url in shakespeare_urls.items():
            response = requests.get(url)
            response.raise_for_status()
            file_path = os.path.join(shakespeare_folder, f"{title}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"{title} wurde erfolgreich heruntergeladen und gespeichert.")
    else:
        print(f"Shakespeare-Texte bereits vorhanden im Ordner '{shakespeare_folder}'.")

    # 3️⃣ Shakespeare-Texte bereinigen und in Sätze aufteilen
    shakespeare_sentences = []
    for filename in os.listdir(shakespeare_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(shakespeare_folder, filename)
            cleaned_text = clean_gutenberg_text(file_path)
            sentences_split = sent_tokenize(cleaned_text)
            sentences_split = [s.strip() for s in sentences_split if len(s.split()) > 2]
            shakespeare_sentences.extend(sentences_split)

    print(f"Anzahl der Shakespeare-Sätze nach Segmentierung: {len(shakespeare_sentences)}")

    # 4️⃣ Kombiniere beide Datensätze
    combined_texts = sentences + shakespeare_sentences
    print(f"Gesamtkombinierte Datensätze: {len(combined_texts)}")

    # 5️⃣ Umwandeln in Hugging Face Dataset
    dataset = Dataset.from_dict({"text": combined_texts})
    return dataset


def clean_gutenberg_text(file_path):
    """
    Bereinigt den Text von Project Gutenberg (Entfernung von Header/Footer und Formatierungen).
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Entferne Header/Footer
    start_match = re.search(r'\*\*\* START OF.*?\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF.*?\*\*\*', text)
    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    elif start_match:
        text = text[start_match.end():]
    elif end_match:
        text = text[:end_match.start()]

    # Entferne leere Zeilen, Szenenangaben und andere Metadaten
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(ACT [IVXLC]+)|(SCENE [IVXLC]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z\s]{5,}\n', '', text, flags=re.MULTILINE)
    return text.strip()

# Funktion aufrufen
dataset = download_and_prepare_datasets(bookcorpus_limit=1000)

# Beispielausgabe
print(f"Erste 5 Sätze im kombinierten Datensatz:\n{dataset['text'][:5]}")
