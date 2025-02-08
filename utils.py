"""
from datasets import load_dataset
# BookCorpus-Datensatz laden für Training
def load_bookcorpus():
    dataset = load_dataset("bookcorpus", trust_remote_code=True)  # Benutzerdefinierten Code erlauben
    return dataset["train"].select(range(10_000))  # Nur den Trainingssplit verwenden
"""
###Komplettes Dataset herunterladen um durchzuschauen wie es aufgebaut ist
"""
from datasets import load_dataset

def download_and_save_bookcorpus():
    # BookCorpus-Datensatz laden
    dataset = load_dataset("bookcorpus", trust_remote_code=True)

    # Nur den Trainingssplit verwenden und die ersten 1000 Sätze auswählen
    sentences = dataset["train"]["text"]
    # Die Textdaten in eine Datei schreiben
    with open("Bookcorpus_dataset.txt", "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")

    print("Datensatz wurde erfolgreich gespeichert in 'Bookcorpus_dataset.txt'.")



# Funktion aufrufen
download_and_save_bookcorpus()
"""
# ----------------------------------------------------------------


##Shakesspearewerke herunterladen und kürzen
"""
from datasets import load_dataset
import requests
import os
import re

# URLs der Shakespeare-Werke von Project Gutenberg
shakespeare_urls = {
    "Hamlet": "https://www.gutenberg.org/files/1524/1524-0.txt",
    "Macbeth": "https://www.gutenberg.org/files/1533/1533-0.txt",
    "Romeo and Juliet": "https://www.gutenberg.org/files/1112/1112-0.txt",
    "Othello": "https://www.gutenberg.org/files/1531/1531-0.txt"
}

def download_and_save_shakespeare():
    if not os.path.exists("Shakespeare_Texts"):
        os.makedirs("Shakespeare_Texts")

    for title, url in shakespeare_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()

            file_path = os.path.join("Shakespeare_Texts", f"{title}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)

            print(f"{title} wurde erfolgreich heruntergeladen und gespeichert.")
        except Exception as e:
            print(f"Fehler beim Herunterladen von {title}: {e}")

# Funktion ausführen
download_and_save_shakespeare()

"""

from datasets import load_dataset, Dataset
import os
import re
import nltk

# Stelle sicher, dass die Punktuationserkennung heruntergeladen ist
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Lade den BookCorpus-Datensatz
def load_bookcorpus():
    dataset = load_dataset("bookcorpus", trust_remote_code=True)
    bookcorpus_sentences = dataset["train"]["text"][:50_000]  # Nur 50.000 Sätze
    return bookcorpus_sentences


# Bereinige Shakespeare-Texte aus dem lokalen Ordner
def clean_gutenberg_text(file_path):
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

    # Entferne leere Zeilen und Szenenangaben
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(ACT [IVXLC]+)|(SCENE [IVXLC]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z\s]{5,}\n', '', text, flags=re.MULTILINE)

    return text.strip()


# Bereinige Shakespeare-Texte aus dem lokalen Ordner und segmentiere in Sätze
def load_and_clean_shakespeare(folder_path="Shakespeare_Texts"):
    cleaned_sentences = []  # Hier speichern wir die Sätze

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            #  Text bereinigen
            cleaned_text = clean_gutenberg_text(file_path)
            #  In Sätze aufteilen
            sentences = sent_tokenize(cleaned_text)
            #  Entferne zu kurze Sätze (z.B. weniger als 3 Wörter)
            sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 2]
            cleaned_sentences.extend(sentences)  # Füge die Sätze zur Liste hinzu

    print(f"Anzahl der Shakespeare-Sätze nach Segmentierung: {len(cleaned_sentences)}")
    return cleaned_sentences


def combine_datasets():
    # Lade BookCorpus und Shakespeare-Texte
    bookcorpus_sentences = load_bookcorpus()
    shakespeare_texts = load_and_clean_shakespeare()

    # Kombiniere die Sätze in einer Liste
    combined_texts = list(bookcorpus_sentences) + shakespeare_texts

    print(f"Anzahl der Sätze in BookCorpus: {len(bookcorpus_sentences)}")
    print(f"Anzahl der Shakespeare-Texte: {len(shakespeare_texts)}")
    print(f"Gesamtkombinierte Datensätze: {len(combined_texts)}")

    return combined_texts


def prepare_dataset_for_training(combined_texts):
    # In Hugging Face Dataset-Format umwandeln
    dataset = Dataset.from_dict({"text": combined_texts})
    return dataset
