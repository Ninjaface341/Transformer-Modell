from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import random

# Pfad zum NSP-Modell
NSP_PATH = "./bookcorpus_nsp_model"

# Überprüfung, ob das Modell existiert
if not os.path.exists(NSP_PATH):
    raise FileNotFoundError("Das gespeicherte NSP-Modell wurde nicht gefunden.")

# Modell und Tokenizer laden
print("Lade gespeichertes NSP-Modell und Tokenizer...")
try:
    nsp_model = AutoModelForSequenceClassification.from_pretrained(NSP_PATH)
    tokenizer = AutoTokenizer.from_pretrained(NSP_PATH)
    print("Modell und Tokenizer erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit(1)

# Setze Modell in den Evaluationsmodus
nsp_model.eval()

# Funktion für Next Sentence Prediction (NSP)
def evaluate_nsp(sentence1: str, sentence2: str):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt")
    with torch.no_grad():
        outputs = nsp_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    next_sentence_prob = probabilities[0][1].item()
    not_next_sentence_prob = probabilities[0][0].item()

    nsp_prediction = "Next sentence" if next_sentence_prob > not_next_sentence_prob else "Not next sentence"
    return nsp_prediction, next_sentence_prob, not_next_sentence_prob

# Funktion zur Berechnung der NSP-Genauigkeit
def evaluate_nsp_accuracy(texts, model, mytokenizer, num_samples=100):
    correct = 0

    for _ in range(num_samples):
        sent1 = random.choice(texts)
        sent2 = random.choice(texts)

        is_next = int(sent2 in sent1)  # 1 = korrekt, 0 = zufällig

        encoding = mytokenizer(sent1, sent2, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        encoding = {k: v.to(model.device) for k, v in encoding.items() if k != "token_type_ids"}  # Entferne 'token_type_ids'

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits[:, 0]  # NSP-Logits
            nsp_prediction = torch.argmax(logits).item()

        if nsp_prediction == is_next:
            correct += 1

    return correct / num_samples

# Beispieltexte für NSP
nsp_examples = [
    ("To be, or not to be,", "that is the question."),
    ("The sun rises in the east,", "and sets in the west."),
    ("Programming is fun", "Thou art more lovely and more temperate."),
    ("To be, or not to be,", "The price of oranges has risen in the market."),
    ("Friends, Romans, countrymen, lend me your ears!", "The temperature in Antarctica is below freezing today."),
    ("All the world’s a stage,", "x."),
    ("The quick brown fox", "jumps over the lazy dog."),
    ("In the beginning", "God created the heavens and the earth."),
]

# NSP-Ergebnisse
print("\n=== Next Sentence Prediction (NSP) Ergebnisse ===")
for s1, s2 in nsp_examples:
    try:
        prediction, next_prob, not_next_prob = evaluate_nsp(s1, s2)
        print(f"\nInput:\nSatz 1: {s1}\nSatz 2: {s2}")
        print(f" - Vorhersage: {prediction}")
        print(f" - Wahrscheinlichkeiten: Next: {next_prob:.4f}, Not Next: {not_next_prob:.4f}")
    except Exception as e:
        print(f"Fehler für Eingabe ('{s1}', '{s2}'): {e}")

# NSP-Genauigkeit testen
accuracy = evaluate_nsp_accuracy(nsp_examples, nsp_model, tokenizer)
print(f"\nNSP Genauigkeit: {accuracy:.2%}")
