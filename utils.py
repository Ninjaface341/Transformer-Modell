import os
import re
import requests
from datasets import load_dataset, Dataset
import nltk


# === Globale Konfigurationsvariablen ===
BOOKCORPUS_LIMIT = 1000  # Anzahl der Sätze aus dem BookCorpus
SHAKESPEARE_LIMIT = None  # Anzahl der Sätze aus den Shakespeare-Texten - if None use complete Data


# NLTK für die Satzsegmentierung herunterladen
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# URLs der Shakespeare-Werke von Project Gutenberg
SHAKESPEARE_URLS = {
    "Hamlet": "https://www.gutenberg.org/files/1524/1524-0.txt",
    "Macbeth": "https://www.gutenberg.org/files/1533/1533-0.txt",
    "Romeo and Juliet": "https://www.gutenberg.org/files/1112/1112-0.txt",
    "Othello": "https://www.gutenberg.org/files/1531/1531-0.txt"
}

# Funktion zum einmaligen Herunterladen des BookCorpus
def download_bookcorpus():
    bookcorpus_file = "Bookcorpus_dataset.txt"

    if not os.path.exists(bookcorpus_file):
        print("Lade BookCorpus-Daten herunter...")
        dataset = load_dataset("bookcorpus", trust_remote_code=True)
        sentences = dataset["train"]["text"]
        
        with open(bookcorpus_file, "w", encoding="utf-8") as file:
            for sentence in sentences:
                file.write(sentence + "\n")
        print(f"BookCorpus-Daten erfolgreich in '{bookcorpus_file}' gespeichert.")
    else:
        print(f"BookCorpus-Daten bereits vorhanden in '{bookcorpus_file}'.")

# Funktion zum Laden der BookCorpus-Daten mit Limit
def load_bookcorpus(limit=BOOKCORPUS_LIMIT):  # eigentlich None -- dauert aber zu lange 
    bookcorpus_file = "Bookcorpus_dataset.txt"
    
    with open(bookcorpus_file, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.readlines()]
    
    if limit is not None:
        sentences = sentences[:limit]
    
    print(f"Lade {len(sentences)} Sätze aus dem BookCorpus.")
    return sentences

# Funktion zum einmaligen Herunterladen der Shakespeare-Texte
def download_shakespeare():
    shakespeare_folder = "Shakespeare_Texts"

    if not os.path.exists(shakespeare_folder):
        os.makedirs(shakespeare_folder)
        print("Lade Shakespeare-Texte herunter...")
        
        for title, url in SHAKESPEARE_URLS.items():
            response = requests.get(url)
            response.raise_for_status()
            
            file_path = os.path.join(shakespeare_folder, f"{title}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"{title} erfolgreich heruntergeladen und gespeichert.")
    else:
        print(f"Shakespeare-Texte bereits im Ordner '{shakespeare_folder}' vorhanden.")

# Funktion zum Laden und Bereinigen der Shakespeare-Texte mit Limit
def load_and_clean_shakespeare(limit=SHAKESPEARE_LIMIT):
    shakespeare_folder = "Shakespeare_Texts"
    cleaned_sentences = []
    
    for filename in os.listdir(shakespeare_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(shakespeare_folder, filename)
            cleaned_text = clean_gutenberg_text(file_path)
            sentences = sent_tokenize(cleaned_text)
            sentences = [s.strip() for s in sentences if len(s.split()) > 2]
            cleaned_sentences.extend(sentences)
    
    if limit is not None:
        cleaned_sentences = cleaned_sentences[:limit]
    
    print(f"Lade {len(cleaned_sentences)} bereinigte Shakespeare-Sätze.")
    return cleaned_sentences

# Funktion zur Bereinigung von Project Gutenberg-Texten
def clean_gutenberg_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    start_match = re.search(r'\*\*\* START OF.*?\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF.*?\*\*\*', text)
    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    elif start_match:
        text = text[start_match.end():]
    elif end_match:
        text = text[:end_match.start()]

    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(ACT [IVXLC]+)|(SCENE [IVXLC]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z\s]{5,}\n', '', text, flags=re.MULTILINE)

    return text.strip()

# Funktion zum Kombinieren der Datensätze mit variablen Limits
def prepare_combined_dataset(bookcorpus_limit=None, shakespeare_limit=None):
    download_bookcorpus()
    download_shakespeare()

    bookcorpus_sentences = load_bookcorpus(limit=bookcorpus_limit)
    shakespeare_sentences = load_and_clean_shakespeare(limit=shakespeare_limit)

    combined_texts = bookcorpus_sentences + shakespeare_sentences
    print(f"Gesamtanzahl der Sätze im kombinierten Datensatz: {len(combined_texts)}")

    dataset = Dataset.from_dict({"text": combined_texts})
    return dataset


def combine_datasets():

    bookcorpus_sentences = load_bookcorpus()  # Nimmt BOOKCORPUS_LIMIT automatisch
    shakespeare_sentences = load_and_clean_shakespeare()  # Nimmt SHAKESPEARE_LIMIT automatisch

    combined_texts = bookcorpus_sentences + shakespeare_sentences
    print(f"Kombinierte Datensätze mit {len(combined_texts)} Sätzen.")
    
    return combined_texts


# Test: Kombiniertes Dataset erstellen
if __name__ == "__main__":


    dataset = prepare_combined_dataset(bookcorpus_limit=BOOKCORPUS_LIMIT, shakespeare_limit=SHAKESPEARE_LIMIT)
    print(f"Erste 5 Sätze im kombinierten Datensatz:\n{dataset['text'][:5]}")
