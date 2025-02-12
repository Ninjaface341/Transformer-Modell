import random

from datasets import Dataset
from transformers import AutoTokenizer

from utils import combine_datasets  # Gemeinsames Dataset laden


# === Funktion zur Erstellung des kombinierten NSP-Datensatzes ===
def create_nsp_dataset(sentences, num_negative_examples=1):
    data = []

    # Iteration über aufeinanderfolgende Satzpaare
    for i in range(len(sentences) - 1):
        # --- Positives Beispiel ---
        # Ein tatsächlich aufeinanderfolgendes Satzpaar wird als positives Beispiel markiert (Label = 0).
        data.append({"sentence1": sentences[i], "sentence2": sentences[i + 1], "next_sentence_label": 0})

        # --- Negatives Beispiel ---
        # Zufällige Satzpaare werden erstellt, die nicht zusammengehören (Label = 1).
        for _ in range(num_negative_examples):
            random_index = random.randint(0, len(sentences) - 1)
            if random_index != i + 1:
                data.append({"sentence1": sentences[i], "sentence2": sentences[random_index], "next_sentence_label": 1})

    # Zufälliges Mischen der Daten zur Vermeidung von Reihenfolgeeffekten
    random.shuffle(data)

    return data


# === Funktion zur Tokenisierung und Vorbereitung der kombinierten Daten ===
def prepare_combined_data():
    # --- Kombiniertes Dataset laden ---
    # Das Dataset besteht aus einer Kombination von BookCorpus- und Shakespeare-Sätzen.
    combined_texts = combine_datasets()

    # --- NSP-Daten erstellen ---
    # Satzpaare werden für die NSP-Aufgabe generiert.
    nsp_data = create_nsp_dataset(combined_texts)
    dataset = Dataset.from_list(nsp_data)

    # --- Tokenizer laden ---
    # Der Tokenizer zerlegt Text in Tokens, die vom Modell verarbeitet werden können.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- Tokenisierung der NSP-Daten ---
    # Die Satzpaare werden tokenisiert, um sie für das Modell vorzubereiten.
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["sentence1"], e["sentence2"], truncation=True, padding="max_length", max_length=128),
        batched=True
    )

    # --- Formatierung der tokenisierten Daten ---
    # Das Dataset wird in ein Torch-Format konvertiert, das vom Modell direkt verarbeitet werden kann.
    tokenized_dataset.set_format(type='torch',
                                 columns=['input_ids', 'attention_mask', 'token_type_ids', 'next_sentence_label'])

    # --- Aufteilung in Trainings- und Validierungsdaten ---
    # 80% der Daten werden für das Training und 20% für die Validierung verwendet.
    split = tokenized_dataset.train_test_split(test_size=0.2)

    return split, tokenizer


# === Beispiel zur Nutzung ===
# Wenn das Skript direkt ausgeführt wird, werden die kombinierten NSP-Daten vorbereitet und ein Beispiel angezeigt.
if __name__ == "__main__":
    datasets, tokenizer = prepare_combined_data()
    print(f"Trainingsbeispiel: {datasets['train'][0]}")
