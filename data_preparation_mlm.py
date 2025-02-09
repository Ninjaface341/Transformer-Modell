from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
from utils import load_bookcorpus, load_and_clean_shakespeare
import logging
import torch

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer und Modell laden
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Funktion zur Maskierung der Daten
def mask_tokens(inputs, tokenizer, mlm_probability=0.14):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=inputs.device)
    
    special_tokens_mask = [
        torch.tensor(tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True), 
                     dtype=torch.bool, device=inputs.device)
        for val in labels.tolist()
    ]
    
    probability_matrix.masked_fill_(torch.stack(special_tokens_mask), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Nicht-maskierte Tokens ignorieren
    
    # 80% der Zeit: [MASK] Token einsetzen
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10% der Zeit: zufälliges Token einsetzen
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1, device=inputs.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
    inputs[indices_random] = random_words[indices_random]
    
    return inputs, labels

# Kombinierte MLM-Daten vorbereiten
def prepare_data_mlm():
    bookcorpus_dataset = load_bookcorpus()
    shakespeare_texts = load_and_clean_shakespeare()

    combined_sentences = bookcorpus_dataset + shakespeare_texts
    dataset = Dataset.from_dict({"text": combined_sentences})

    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        batch_size=8,
        remove_columns=["text"]
    )

    # Maskierung der Daten
    def apply_masking(examples):
        input_ids = torch.tensor(examples["input_ids"]).to(device)
        attention_mask = torch.tensor(examples["attention_mask"]).to(device)
        masked_inputs, labels = mask_tokens(input_ids, tokenizer)
        
        return {
            "input_ids": masked_inputs,
            "labels": labels,
            "attention_mask": attention_mask
        }

    masked_dataset = tokenized_dataset.map(apply_masking, batched=True)

    # Ausgabe zur Kontrolle
    print(f"Anzahl der vorbereiteten Sätze für MLM: {len(combined_sentences)}")
    print(f"Beispiel-Sätze aus BookCorpus: {bookcorpus_dataset[:2]}")
    print(f"Beispiel-Sätze aus Shakespeare: {shakespeare_texts[:2]}")

    # Aufteilen in Trainings- und Validierungsdaten
    train_size = int(0.8 * len(masked_dataset))
    valid_size = len(masked_dataset) - train_size
    train_dataset, valid_dataset = masked_dataset.train_test_split(test_size=valid_size).values()

    print(f"Trainingsdaten: {len(train_dataset)} Beispiele")
    print(f"Validierungsdaten: {len(valid_dataset)} Beispiele")

    return {"train": train_dataset, "validation": valid_dataset}
