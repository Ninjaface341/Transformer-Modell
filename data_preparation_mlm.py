import logging
import random
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

from utils import combine_datasets, load_bookcorpus, load_and_clean_shakespeare  # Load combined dataset and additional data sources

# === Configure logging ===
# The logging module is used to display progress and important information during execution.
logging.basicConfig(level=logging.INFO)

# === Set device for computations ===
# The model will run on GPU if available, otherwise on CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer and model ===
# Using 'distilbert-base-cased', a lighter version of the BERT model.
model_checkpoint = "distilbert-base-cased"

# The tokenizer breaks text into tokens that can be processed by the model.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# The Masked Language Model (MLM) is loaded and moved to the chosen device.
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint).to(device)

# === Function to prepare data for Masked Language Modeling (MLM) ===
def prepare_data_mlm():
    # --- Load data ---
    # Load BookCorpus data and cleaned Shakespeare texts.
    bookcorpus_dataset = load_bookcorpus()
    shakespeare_texts = load_and_clean_shakespeare()

    # --- Combine datasets ---
    # The loaded sentences are combined into one dataset.
    combined_sentences = bookcorpus_dataset + shakespeare_texts
    dataset = Dataset.from_dict({"text": combined_sentences})

    # --- Tokenization ---
    # The combined dataset is tokenized. Texts are truncated to a maximum length of 512 tokens.
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        remove_columns=["text"]  # The original text is removed as only tokens are needed.
    )

    # --- Data Collator for masking ---
    # The DataCollator randomly masks 15% of the tokens, required for training the MLM.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 15% of tokens are masked for Masked Language Modeling.
    )

    # --- Split dataset into training and validation ---
    # The dataset is split into 80% training and 20% validation data.
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    valid_dataset = split["test"]

    # --- Log dataset sizes ---
    logging.info(f"Number of training samples: {len(train_dataset)}")
    logging.info(f"Number of validation samples: {len(valid_dataset)}")

    # --- Return prepared dataset ---
    # The prepared training and validation datasets along with the DataCollator are returned.
    return {"train": train_dataset, "validation": valid_dataset, "collator": data_collator}

# === Example usage ===
# When the script is run directly, the MLM data is prepared and an example is displayed.
if __name__ == "__main__":
    datasets = prepare_data_mlm()
    print(f"MLM Training example: {datasets['train'][0]}")
