"""
from transformers import BertTokenizer, BertForMaskedLM
import torch
from datasets import Dataset
from utils import load_bookcorpus

# Tokenizer für MLM
tokenizer_mlm = BertTokenizer.from_pretrained('bert-base-cased')
model_mlm = BertForMaskedLM.from_pretrained('bert-base-cased')


# MLM-Daten vorbereiten

# Maskierungsprozess für BERT cased
def prepare_data_mlm():
    dataset = load_bookcorpus()
    sentences = dataset["text"]
    dataset = Dataset.from_dict({"text": sentences})

    # Tokenisierung
    tokenized_dataset = dataset.map(
        lambda e: tokenizer_mlm(e["text"], truncation=True, padding="max_length", max_length=128), batched=True,
        batch_size=1000  # Batch-Größe anpassen
    )
    # Ausgabe für Datenvorbereitung
    print(f"Anzahl der vorbereiteten Sätze für MLM: {len(sentences)}")

    # Dataset in train und validation aufteilen
    train_size = int(0.8 * len(tokenized_dataset))  # 80% für Training
    valid_size = len(tokenized_dataset) - train_size  # Der Rest für Validation
    train_dataset, valid_dataset = tokenized_dataset.train_test_split(test_size=valid_size).values()

    # Ausgabe von Beispiel-Sätzen aus der Quelle
    print(f"Beispiel-Sätze aus der Quelle: {sentences[:3]}")  # Ersten 3 Sätze anzeigen

    return {"train": train_dataset, "validation": valid_dataset}


""""""
# DataPreparation.py
from transformers import AutoTokenizer, AutoModelForMaskedLM , DataCollatorForLanguageModeling
from datasets import Dataset
from utils import load_bookcorpus
import logging

# Tokenizer für MLM
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-cased')
logging.getLogger("transformers").setLevel(logging.ERROR)


data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)
# Maskierungsprozess für BERT cased
def mask_tokens(batch, tokenz, mlm_probability=0.15, max_length=512):
    # Tokenisierung des Textes (Paddung und Truncation sicherstellen)
    inputs = tokenz(batch["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    labels = inputs.input_ids.clone()

    # Maskierungswahrscheinlichkeitsmatrix erstellen
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        1 if token in tokenz.all_special_ids else 0 for token in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Maskierte Tokens im Input
    inputs.input_ids[masked_indices] = tokenz.convert_tokens_to_ids(tokenz.mask_token)

    # Die nicht maskierten Labels auf -100 setzen, damit sie im Training ignoriert werden
    labels[~masked_indices] = -100

    # Sicherstellen, dass sowohl input_ids als auch labels die gleiche Länge haben
    assert inputs.input_ids.shape == labels.shape, f"Shape mismatch: {inputs.input_ids.shape} != {labels.shape}"

    return {
        "input_ids": inputs.input_ids.tolist(),
        "attention_mask": inputs.attention_mask.tolist(),
        "labels": labels.tolist(),
    }

# MLM-Daten vorbereiten
def prepare_data_mlm():
    dataset = load_bookcorpus()
    sentences = dataset["text"]
    dataset = Dataset.from_dict({"text": sentences})

    # Tokenisierung
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        batch_size=32  # Reduziere die Batch-Größe
    )

    # Ausgabe für Datenvorbereitung
    print(f"Anzahl der vorbereiteten Sätze für MLM: {len(sentences)}")
    print(f"Durchschnittliche Länge: {sum([len(text.split()) for text in sentences]) / len(sentences)}")
    print(f"Maximale Länge: {max([len(text.split()) for text in sentences])}")

    # Dataset in train und validation aufteilen
    train_size = int(0.8 * len(tokenized_dataset))  # 80% für Training
    valid_size = len(tokenized_dataset) - train_size  # Der Rest für Validation
    train_dataset, valid_dataset = tokenized_dataset.train_test_split(test_size=valid_size).values()

    # Ausgabe von Beispiel-Sätzen aus der Quelle
    print(f"Beispiel-Sätze aus der Quelle: {sentences[:3]}")  # Ersten 3 Sätze anzeigen

    return {"train": train_dataset, "validation": valid_dataset}
"""

###Funktioniert.
"""
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
from utils import load_bookcorpus
import logging
import torch

logging.basicConfig(level=logging.INFO)

# Leichteres Modell und Tokenizer verwenden
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


# Funktion zur Maskierung der Daten

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()

    # Erstellen einer Maske mit mlm_probability
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # Ignoriere nicht-maskierte Tokens

    # 80% der Zeit: [MASK] Token einfügen
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% der Zeit: zufälliges Token einfügen
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 10% der Zeit: Original-Token beibehalten
    return inputs, labels


# MLM-Daten vorbereiten
def prepare_data_mlm():
    dataset = load_bookcorpus()
    sentences = dataset["text"]
    dataset = Dataset.from_dict({"text": sentences})

    # Tokenisierung mit explizitem Padding und Truncation
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        batch_size=8,
        remove_columns=["text"]
    )

    # Maskierung der Daten
    def apply_masking(examples):
        input_ids = torch.tensor(examples["input_ids"])
        masked_inputs, labels = mask_tokens(input_ids, tokenizer)
        return {"input_ids": masked_inputs.tolist(), "labels": labels.tolist(),
                "attention_mask": examples["attention_mask"]}

    masked_dataset = tokenized_dataset.map(apply_masking, batched=True)

    # Ausgabe für Datenvorbereitung
    print(f"Anzahl der vorbereiteten Sätze für MLM: {len(sentences)}")
    print(f"Durchschnittliche Länge: {sum([len(text.split()) for text in sentences]) / len(sentences)}")
    print(f"Maximale Länge: {max([len(text.split()) for text in sentences])}")

    # Dataset in train und validation aufteilen
    train_size = int(0.8 * len(masked_dataset))
    valid_size = len(masked_dataset) - train_size
    train_dataset, valid_dataset = masked_dataset.train_test_split(test_size=valid_size).values()

    print(f"Trainingsdaten: {len(train_dataset)} Beispiele")
    print(f"Validierungsdaten: {len(valid_dataset)} Beispiele")
    print(f"Beispiel-Sätze aus der Quelle: {sentences[:3]}")

    return {"train": train_dataset, "validation": valid_dataset}
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
from utils import load_bookcorpus, load_and_clean_shakespeare  # Shakespeare-Ladefunktion hinzufügen
import logging
import torch

logging.basicConfig(level=logging.INFO)

# Leichteres Modell und Tokenizer verwenden
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


# Funktion zur Maskierung der Daten
def mask_tokens(inputs, tokenizer, mlm_probability=0.14):
    labels = inputs.clone()

    # Erstellen einer Maske mit mlm_probability
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # Ignoriere nicht-maskierte Tokens

    # 80% der Zeit: [MASK] Token einfügen
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% der Zeit: zufälliges Token einfügen
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 10% der Zeit: Original-Token beibehalten
    return inputs, labels


# Kombinierte MLM-Daten vorbereiten
def prepare_data_mlm():
    # Lade BookCorpus und Shakespeare-Daten
    bookcorpus_dataset = load_bookcorpus()  # Fix: Keine Indizierung mit ["text"]
    shakespeare_texts = load_and_clean_shakespeare()

    # Kombiniere beide Datensätze
    combined_sentences = bookcorpus_dataset + shakespeare_texts
    dataset = Dataset.from_dict({"text": combined_sentences})

    #   Tokenisierung mit Padding und Truncation
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        batch_size=8,
        remove_columns=["text"]
    )

    #  Maskierung der Daten
    def apply_masking(examples):
        input_ids = torch.tensor(examples["input_ids"])
        masked_inputs, labels = mask_tokens(input_ids, tokenizer)
        return {"input_ids": masked_inputs.tolist(), "labels": labels.tolist(),
                "attention_mask": examples["attention_mask"]}

    masked_dataset = tokenized_dataset.map(apply_masking, batched=True)

    # Ausgabe für Datenvorbereitung
    print(f"Anzahl der vorbereiteten Sätze für MLM: {len(combined_sentences)}")
    print(f"Beispiel-Sätze aus BookCorpus: {bookcorpus_dataset[:2]}")
    print(f"Beispiel-Sätze aus Shakespeare: {shakespeare_texts[:2]}")

    # Dataset in train und validation aufteilen
    train_size = int(0.8 * len(masked_dataset))
    valid_size = len(masked_dataset) - train_size
    train_dataset, valid_dataset = masked_dataset.train_test_split(test_size=valid_size).values()

    print(f"Trainingsdaten: {len(train_dataset)} Beispiele")
    print(f"Validierungsdaten: {len(valid_dataset)} Beispiele")

    return {"train": train_dataset, "validation": valid_dataset}