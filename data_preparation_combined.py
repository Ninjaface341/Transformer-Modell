import random

from datasets import Dataset
from transformers import AutoTokenizer

from utils import combine_datasets  # Load combined dataset


# === Function to create the combined NSP dataset ===
def create_nsp_dataset(sentences, num_negative_examples=1):
    data = []

    # Iterate over consecutive sentence pairs
    for i in range(len(sentences) - 1):
        # --- Positive example ---
        # An actual consecutive sentence pair is marked as a positive example (Label = 0).
        data.append({"sentence1": sentences[i], "sentence2": sentences[i + 1], "next_sentence_label": 0})

        # --- Negative example ---
        # Random sentence pairs are created that do not belong together (Label = 1).
        for _ in range(num_negative_examples):
            random_index = random.randint(0, len(sentences) - 1)
            if random_index != i + 1:
                data.append({"sentence1": sentences[i], "sentence2": sentences[random_index], "next_sentence_label": 1})

    # Shuffle the data randomly to avoid order effects
    random.shuffle(data)

    return data


# === Function to tokenize and prepare the combined data ===
def prepare_combined_data():
    # --- Load combined dataset ---
    # The dataset consists of a combination of BookCorpus and Shakespeare sentences.
    combined_texts = combine_datasets()

    # --- Create NSP data ---
    # Sentence pairs are generated for the NSP task.
    nsp_data = create_nsp_dataset(combined_texts)
    dataset = Dataset.from_list(nsp_data)

    # --- Load tokenizer ---
    # The tokenizer breaks text into tokens that can be processed by the model.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- Tokenize NSP data ---
    # The sentence pairs are tokenized to prepare them for the model.
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["sentence1"], e["sentence2"], truncation=True, padding="max_length", max_length=128),
        batched=True
    )

    # --- Format tokenized data ---
    # The dataset is converted to a Torch format that can be directly processed by the model.
    tokenized_dataset.set_format(type='torch',
                                 columns=['input_ids', 'attention_mask', 'token_type_ids', 'next_sentence_label'])

    # --- Split into training and validation data ---
    # 80% of the data is used for training and 20% for validation.
    split = tokenized_dataset.train_test_split(test_size=0.2)

    return split, tokenizer


# === Example usage ===
# When the script is run directly, the combined NSP data is prepared and an example is displayed.
if __name__ == "__main__":
    datasets, tokenizer = prepare_combined_data()
    print(f"Training example: {datasets['train'][0]}")
