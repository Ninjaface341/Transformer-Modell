import logging
import os
import random
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForNextSentencePrediction, BertForMaskedLM

from utils import combine_datasets  # Load combined dataset

# === Configure logging ===
# The logging module is used to display progress and important information during execution.
logging.basicConfig(level=logging.INFO)

# === Path to the MLM model ===
MLM_PATH = "./bookcorpus_mlm_model"
# COMBINED_PATH = "./bert_combined_model"  # --- Change if COMBINED_PATH needs to be checked ---

# === Check if the model exists ===
if not os.path.exists(MLM_PATH):
    raise FileNotFoundError("The saved MLM model was not found.")

# === Load model and tokenizer ===
print("Loading saved MLM model and tokenizer...")
try:
    mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_PATH)
    # mlm_model = BertForMaskedLM.from_pretrained(COMBINED_PATH)  # --- Change if COMBINED_PATH needs to be checked ---

    tokenizer = AutoTokenizer.from_pretrained(MLM_PATH)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# === Set device for computations ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlm_model.to(device)

# === Set model to evaluation mode ===
mlm_model.eval()

# === Function for Masked Language Modeling (MLM) ===
def evaluate_mlm(input_text: str, top_k=5):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if mask_token_index.numel() == 0:
        raise ValueError("No [MASK] token found in the input text.")

    with torch.no_grad():
        outputs = mlm_model(**inputs)
        logits = outputs.logits

    softmax = torch.nn.functional.softmax(logits, dim=-1)

    mlm_predictions = []
    for idx in mask_token_index:
        token_logits = softmax[0, idx, :]
        top_tokens = torch.topk(token_logits, top_k, dim=0).indices.tolist()
        predictions = [(tokenizer.decode([token]), float(token_logits[token].item())) for token in top_tokens]
        mlm_predictions.append(predictions)

    return mlm_predictions

# === Function to calculate MLM accuracy (counting Top-5 hits) ===
def evaluate_mlm_accuracy(examples_with_answers, top_k=5):
    correct_predictions = 0
    total_masks = 0

    for example, correct_words in examples_with_answers:
        if not isinstance(correct_words, list):
            correct_words = [correct_words]  # In case only one word is provided

        try:
            predictions = evaluate_mlm(example, top_k=top_k)
            total_masks += len(correct_words)

            for idx, correct_word in enumerate(correct_words):
                predicted_words = [word.strip().lower() for word, _ in predictions[idx]]
                if correct_word.lower() in predicted_words:
                    correct_predictions += 1
        except ValueError as e:
            print(f"Error for input '{example}': {e}")

    accuracy = correct_predictions / total_masks
    return accuracy

# === Example texts for MLM with correct answers ===
mlm_examples_with_answers = [
    ("To be, or not to be, that is the [MASK]:", "question"),
    ("All the world's a [MASK], and all the men and women merely [MASK].", ["stage", "players"]),
    ("Shall I compare thee to a [MASK]'s day?", "summer"),
    ("If [MASK] be the food of [MASK], play on.", ["music", "love"]),
    ("O Romeo, Romeo! Wherefore art thou [MASK] Romeo?", "Romeo"),
    ("The lady doth protest too [MASK], methinks.", "much"),
    ("A horse! A horse! My [MASK] for a horse!", "kingdom"),
    ("Brevity is the soul of [MASK].", "wit"),

    ("She opened the door and saw the [MASK] shining brightly.", "sun"),
    ("It was a long journey, but finally they reached the [MASK].", "destination"),
    ("The cat jumped onto the [MASK] and knocked over the vase.", "table"),
    ("He couldn't believe his [MASK] when he saw the results.", "eyes"),
    ("They sat around the [MASK], sharing stories from their past.", "fire"),
    ("She always dreamed of visiting the [MASK] during summer.", "beach"),
    ("He picked up the [MASK] and started reading quietly.", "book"),
    ("After the storm, the sky turned a brilliant shade of [MASK].", "blue"),
    ("He whispered the secret into her [MASK], hoping no one else would hear.", "ear"),
    ("She found the hidden [MASK] under the old wooden floor.", "treasure"),
    ("The sound of the waves crashing against the [MASK] was calming.", "shore"),
    ("They packed their bags and left for the [MASK] early in the morning.", "airport"),
]

# === Output MLM results ===
print("\n=== Masked Language Modeling (MLM) Results ===")
for example, correct_words in mlm_examples_with_answers:
    try:
        predictions = evaluate_mlm(example)
        print(f"\nInput: {example}")
        for idx, preds in enumerate(predictions):
            print(f"Predictions for mask {idx + 1}:")
            for word, prob in preds:
                print(f" - {word}: {prob:.4f}")
        print(f" - Correct answer(s): {correct_words}")
    except ValueError as e:
        print(f"Error for input '{example}': {e}")

# === Test MLM accuracy ===
accuracy = evaluate_mlm_accuracy(mlm_examples_with_answers)
print(f"\nMLM Accuracy: {accuracy:.2%}")
