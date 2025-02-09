from transformers import AutoTokenizer, BertForNextSentencePrediction
from datasets import Dataset
from utils import combine_datasets  # Gemeinsames Dataset laden
import random
import logging

logging.basicConfig(level=logging.INFO)

# Tokenizer und Modell für NSP laden
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = BertForNextSentencePrediction.from_pretrained(model_checkpoint)


# Funktion zur Erstellung des NSP-Datensatzes
def create_nsp_dataset(sentences, num_negative_examples=1):
    data = []
    for i in range(len(sentences) - 1):
        # Positives Beispiel (tatsächliche Folge)
        data.append({"sentence1": sentences[i], "sentence2": sentences[i + 1], "label": 0})

        # Negatives Beispiel (zufällige Sätze)
        for _ in range(num_negative_examples):
            random_index = random.randint(0, len(sentences) - 1)
            if random_index != i + 1:
                data.append({"sentence1": sentences[i], "sentence2": sentences[random_index], "label": 1})

    random.shuffle(data)
    logging.info(f"NSP-Daten erstellt mit {len(data)} Beispielen.")
    return data


# Funktion zur Vorbereitung der NSP-Daten
def prepare_data_nsp():
    combined_texts = combine_datasets()  # Kombiniertes Dataset aus BookCorpus & Shakespeare
    nsp_data = create_nsp_dataset(combined_texts)
    dataset = Dataset.from_list(nsp_data)

    # Tokenisierung der NSP-Daten
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["sentence1"], e["sentence2"], truncation=True, padding="max_length", max_length=128),
        batched=True,
        batch_size=32
    )

    # Ausgabe für NSP-Datenvorbereitung
    logging.info(f"Anzahl der vorbereiteten Paare für NSP: {len(nsp_data)}")

    # Aufteilung in Trainings- und Validierungsdaten
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    valid_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    logging.info(f"Trainingsdaten: {len(train_dataset)} Beispiele")
    logging.info(f"Validierungsdaten: {len(valid_dataset)} Beispiele")

    return {"train": train_dataset, "validation": valid_dataset}
