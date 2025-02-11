# === data_preparation_combined.py ===
import random
from datasets import Dataset
from transformers import AutoTokenizer
from utils import combine_datasets  # Kombinierte BookCorpus & Shakespeare-Daten laden

# NSP-Daten vorbereiten
def create_nsp_dataset(sentences, num_negative_examples=1):
    data = []
    for i in range(len(sentences) - 1):
        data.append({"sentence1": sentences[i], "sentence2": sentences[i + 1], "next_sentence_label": 0})
        for _ in range(num_negative_examples):
            random_index = random.randint(0, len(sentences) - 1)
            if random_index != i + 1:
                data.append({"sentence1": sentences[i], "sentence2": sentences[random_index], "next_sentence_label": 1})
    random.shuffle(data)
    return data

# Tokenisierung der kombinierten Daten
def prepare_combined_data():
    combined_texts = combine_datasets()
    nsp_data = create_nsp_dataset(combined_texts)
    dataset = Dataset.from_list(nsp_data)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["sentence1"], e["sentence2"], truncation=True, padding="max_length", max_length=128),
        batched=True
    )

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'next_sentence_label'])
    
    split = tokenized_dataset.train_test_split(test_size=0.2)
    return split, tokenizer
