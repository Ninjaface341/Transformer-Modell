import logging
import random

from datasets import Dataset
from transformers import AutoTokenizer, BertForNextSentencePrediction

from utils import combine_datasets  # Load combined dataset

# === Configure logging ===
# The logging module is used to display progress and important information during execution.
logging.basicConfig(level=logging.INFO)

# === Load tokenizer and model for Next Sentence Prediction (NSP) ===
# We are using the 'bert-base-uncased' model, which is suitable for tasks like NSP.
model_checkpoint = "bert-base-uncased"

# The tokenizer breaks text into tokens that can be processed by the model.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# The BERT model for Next Sentence Prediction is loaded.
model = BertForNextSentencePrediction.from_pretrained(model_checkpoint)


# === Function to create the NSP dataset ===
def create_nsp_dataset(sentences, num_negative_examples=1):
    data = []

    # Iterate over consecutive sentence pairs
    for i in range(len(sentences) - 1):
        # --- Positive example ---
        # An actual consecutive sentence pair is marked as a positive example (Label = 0).
        data.append({"sentence1": sentences[i], "sentence2": sentences[i + 1], "label": 0})

        # --- Negative example ---
        # Random sentence pairs are created that do not belong together (Label = 1).
        for _ in range(num_negative_examples):
            random_index = random.randint(0, len(sentences) - 1)
            if random_index != i + 1:
                data.append({"sentence1": sentences[i], "sentence2": sentences[random_index], "label": 1})

    # Shuffle the data randomly to avoid order effects
    random.shuffle(data)

    logging.info(f"NSP data created with {len(data)} examples.")
    return data


# === Function to prepare NSP data ===
def prepare_data_nsp():
    # --- Load combined dataset ---
    # The dataset consists of a combination of BookCorpus and Shakespeare sentences.
    combined_texts = combine_datasets()

    # --- Create NSP data ---
    # Sentence pairs are generated for the NSP task.
    nsp_data = create_nsp_dataset(combined_texts)
    dataset = Dataset.from_list(nsp_data)

    # --- Tokenize NSP data ---
    # The sentence pairs are tokenized to prepare them for the model.
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["sentence1"], e["sentence2"], truncation=True, padding="max_length", max_length=128),
        batched=True,
        batch_size=32
    )

    # --- Log NSP data ---
    logging.info(f"Number of prepared pairs for NSP: {len(nsp_data)}")

    # --- Split into training and validation data ---
    # 80% of the data is used for training and 20% for validation.
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    valid_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    logging.info(f"Training data: {len(train_dataset)} examples")
    logging.info(f"Validation data: {len(valid_dataset)} examples")

    # --- Return prepared datasets ---
    return {"train": train_dataset, "validation": valid_dataset}


# === Example usage ===
# When the script is run directly, the NSP data is prepared and an example is displayed.
if __name__ == "__main__":
    datasets = prepare_data_nsp()
    print(f"Training example: {datasets['train'][0]}")
