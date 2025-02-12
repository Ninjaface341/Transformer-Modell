import logging
import os

import torch
from transformers import BertForNextSentencePrediction, AutoTokenizer

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Path to the NSP model ===
NSP_PATH = "./bookcorpus_nsp_model"
# COMBINED_PATH = "./bert_combined_model" # --- Change if COMBINED_PATH needs to be checked ---

# === Check if the model exists ===
if not os.path.exists(NSP_PATH):
    raise FileNotFoundError("The saved NSP model was not found.")

# === Load model and tokenizer ===
logger.info("Loading saved NSP model and tokenizer...")
try:
    nsp_model = BertForNextSentencePrediction.from_pretrained(NSP_PATH)  # --- Change if COMBINED_PATH needs to be checked ---
    tokenizer = AutoTokenizer.from_pretrained(NSP_PATH)  # --- Change if COMBINED_PATH needs to be checked ---
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    exit(1)

# === Set model to evaluation mode ===
nsp_model.eval()

# === Function for Next Sentence Prediction (NSP) ===
def evaluate_nsp(sentence1: str, sentence2: str):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = nsp_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    next_sentence_prob = probabilities[0][0].item()
    not_next_sentence_prob = probabilities[0][1].item()

    nsp_prediction = "Next sentence" if next_sentence_prob > not_next_sentence_prob else "Not next sentence"
    return nsp_prediction, next_sentence_prob, not_next_sentence_prob

# === Example texts for Next Sentence Prediction (NSP) with labels ===
nsp_examples = [
    ("To be, or not to be,", "that is the question:", 0),
    ("The sun rises in the east,", "and sets in the west.", 0),
    ("Friends, Romans, countrymen, lend me your ears!", "I come to bury Caesar, not to praise him.", 0),
    ("O Romeo, Romeo!", "Wherefore art thou Romeo?", 0),
    ("In the beginning,", "God created the heavens and the earth.", 0),
    ("The quick brown fox", "jumps over the lazy dog.", 0),
    ("Once upon a time,", "there was a little girl named Red Riding Hood.", 0),
    ("He opened the old book,", "and dust flew into the air.", 0),
    ("The storm was fierce,", "but the sailors held their course.", 0),
    ("She knocked on the door,", "and waited for a response.", 0),

    ("To be, or not to be,", "The cat ran across the street.", 1),
    ("The sun rises in the east,", "Bananas are rich in potassium.", 1),
    ("Friends, Romans, countrymen, lend me your ears!", "It’s going to rain tomorrow.", 1),
    ("O Romeo, Romeo!", "The price of oil has dropped significantly.", 1),
    ("In the beginning,", "The concert was sold out in minutes.", 1),
    ("The quick brown fox", "She loves to paint in her free time.", 1),
    ("Once upon a time,", "The temperature in Antarctica is freezing.", 1),
    ("He opened the old book,", "They decided to buy a new car.", 1),
    ("The storm was fierce,", "Mathematics is a fundamental subject in school.", 1),
    ("She knocked on the door,", "The stock market closed higher today.", 1),
]

# === Function to calculate NSP accuracy with labels ===
def evaluate_nsp_accuracy(examples, model, tokenizer):
    correct_predictions = 0
    total_examples = len(examples)

    for sentence1, sentence2, true_label in examples:
        try:
            inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()

            if predicted_label == true_label:
                correct_predictions += 1
        except Exception as e:
            logger.error(f"Error processing ('{sentence1}', '{sentence2}'): {e}")

    accuracy = correct_predictions / total_examples
    return accuracy

# === Output NSP results ===
logger.info("\n=== Next Sentence Prediction (NSP) Results ===")
for s1, s2, label in nsp_examples:
    try:
        prediction, next_prob, not_next_prob = evaluate_nsp(s1, s2)
        logger.info(f"\nInput:\nSentence 1: {s1}\nSentence 2: {s2}")
        logger.info(f" - Prediction: {prediction}")
        logger.info(f" - Probabilities: Next: {next_prob:.4f}, Not Next: {not_next_prob:.4f}")
    except Exception as e:
        logger.error(f"Error for input ('{s1}', '{s2}'): {e}")

# === Test NSP accuracy ===
accuracy = evaluate_nsp_accuracy(nsp_examples, nsp_model, tokenizer)
logger.info(f"\nNSP Accuracy: {accuracy:.2%}")
