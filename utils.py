import os
import re

import nltk
import requests
from datasets import load_dataset, Dataset

# === Configuration variables for loading data ===
BOOKCORPUS_LIMIT = 12000  # Maximum number of sentences from BookCorpus. If None, all data is used.
SHAKESPEARE_LIMIT = None  # Maximum number of sentences from Shakespeare texts. If None, all texts are used.

# Download NLTK library for sentence tokenization
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# URLs of Shakespeare's works from Project Gutenberg
SHAKESPEARE_URLS = {
    "Hamlet": "https://www.gutenberg.org/files/1524/1524-0.txt",
    "Macbeth": "https://www.gutenberg.org/files/1533/1533-0.txt",
    "Romeo and Juliet": "https://www.gutenberg.org/files/1112/1112-0.txt",
    "Othello": "https://www.gutenberg.org/files/1531/1531-0.txt"
}


# Function to download the BookCorpus dataset once
def download_bookcorpus():
    bookcorpus_file = "Bookcorpus_dataset.txt"

    # Check if the data has already been downloaded
    if not os.path.exists(bookcorpus_file):
        print("Downloading BookCorpus data...")
        dataset = load_dataset("bookcorpus", trust_remote_code=True)  # Load dataset from Hugging Face
        sentences = dataset["train"]["text"]  # Extract text data

        # Save the sentences to a file
        with open(bookcorpus_file, "w", encoding="utf-8") as file:
            for sentence in sentences:
                file.write(sentence + "\n")
        print(f"BookCorpus data successfully saved to '{bookcorpus_file}'.")
    else:
        print(f"BookCorpus data already exists in '{bookcorpus_file}'.")


# Function to load BookCorpus data with an optional limit
def load_bookcorpus(limit=BOOKCORPUS_LIMIT):
    bookcorpus_file = "Bookcorpus_dataset.txt"

    # Read the file containing BookCorpus sentences
    with open(bookcorpus_file, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.readlines()]

    # Limit the number of loaded sentences if a limit is set
    if limit is not None:
        sentences = sentences[:limit]

    print(f"Loading {len(sentences)} sentences from BookCorpus.")
    return sentences


# Function to download Shakespeare texts from Project Gutenberg once
def download_shakespeare():
    shakespeare_folder = "Shakespeare_Texts"

    # Check if the texts already exist
    if not os.path.exists(shakespeare_folder):
        os.makedirs(shakespeare_folder)
        print("Downloading Shakespeare texts...")

        # Iterate over the works and download the texts
        for title, url in SHAKESPEARE_URLS.items():
            response = requests.get(url)
            response.raise_for_status()

            # Save the downloaded text to a file
            file_path = os.path.join(shakespeare_folder, f"{title}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"{title} successfully downloaded and saved.")
    else:
        print(f"Shakespeare texts already exist in the folder '{shakespeare_folder}'.")


# Function to load and clean Shakespeare texts with an optional limit
def load_and_clean_shakespeare(limit=SHAKESPEARE_LIMIT):
    shakespeare_folder = "Shakespeare_Texts"
    cleaned_sentences = []

    # Iterate over the saved files in the Shakespeare folder
    for filename in os.listdir(shakespeare_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(shakespeare_folder, filename)
            cleaned_text = clean_gutenberg_text(file_path)  # Clean the text
            sentences = sent_tokenize(cleaned_text)  # Segment into sentences
            sentences = [s.strip() for s in sentences if len(s.split()) > 2]  # Remove very short sentences
            cleaned_sentences.extend(sentences)

    # Limit the number of sentences if specified
    if limit is not None:
        cleaned_sentences = cleaned_sentences[:limit]

    print(f"Loading {len(cleaned_sentences)} cleaned Shakespeare sentences.")
    return cleaned_sentences


# Function to clean Project Gutenberg texts (remove meta-information)
def clean_gutenberg_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Remove Project Gutenberg metadata
    start_match = re.search(r'\*\*\* START OF.*?\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF.*?\*\*\*', text)
    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    elif start_match:
        text = text[start_match.end():]
    elif end_match:
        text = text[:end_match.start()]

    # Remove unnecessary line breaks and scene descriptions
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(ACT [IVXLC]+)|(SCENE [IVXLC]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z\s]{5,}\n', '', text, flags=re.MULTILINE)

    return text.strip()


# Function to combine datasets with optional limits
def prepare_combined_dataset(bookcorpus_limit=None, shakespeare_limit=None):
    download_bookcorpus()  # Download BookCorpus data
    download_shakespeare()  # Download Shakespeare texts

    # Load datasets with the given limits
    bookcorpus_sentences = load_bookcorpus(limit=bookcorpus_limit)
    shakespeare_sentences = load_and_clean_shakespeare(limit=shakespeare_limit)

    # Combine both datasets
    combined_texts = bookcorpus_sentences + shakespeare_sentences
    print(f"Total number of sentences in the combined dataset: {len(combined_texts)}")

    # Create a Hugging Face dataset object
    dataset = Dataset.from_dict({"text": combined_texts})
    return dataset


# Function to combine datasets without specifying limits
def combine_datasets():
    bookcorpus_sentences = load_bookcorpus()  # Default limit from configuration variable
    shakespeare_sentences = load_and_clean_shakespeare()  # Default limit from configuration variable

    combined_texts = bookcorpus_sentences + shakespeare_sentences
    print(f"Combined datasets with {len(combined_texts)} sentences.")

    return combined_texts


# Create combined dataset when the script is run directly
if __name__ == "__main__":
    dataset = prepare_combined_dataset(bookcorpus_limit=BOOKCORPUS_LIMIT, shakespeare_limit=SHAKESPEARE_LIMIT)
    print(f"First 5 sentences in the combined dataset:\n{dataset['text'][:5]}")
