from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import os

# Pfad zum MLM-Modell
MLM_PATH = "./bookcorpus_mlm_model"

# Überprüfung, ob das Modell existiert
if not os.path.exists(MLM_PATH):
    raise FileNotFoundError("Das gespeicherte MLM-Modell wurde nicht gefunden.")

# Modell und Tokenizer laden
print("Lade gespeichertes MLM-Modell und Tokenizer...")
try:
    mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MLM_PATH)
    print("Modell und Tokenizer erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit(1)

# Setze Modell in den Evaluationsmodus
mlm_model.eval()

# Funktion für Masked Language Modeling (MLM)
def evaluate_mlm(input_text: str, top_k=5):
    inputs = tokenizer(input_text, return_tensors="pt")
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if mask_token_index.numel() == 0:
        raise ValueError("Kein [MASK] Token im Eingabetext gefunden.")

    with torch.no_grad():
        outputs = mlm_model(**inputs)
        logits = outputs.logits

    softmax = torch.nn.functional.softmax(logits, dim=-1)
    mask_token_logits = softmax[0, mask_token_index, :]

    top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
    mlm_predictions = [
        (tokenizer.decode([token]), float(mask_token_logits[0, token].item()))
        for token in top_tokens
    ]

    return mlm_predictions

# Funktion zur Berechnung der MLM-Genauigkeit (Top-5 Treffer zählen)
def evaluate_mlm_accuracy(examples_with_answers, top_k=5):
    correct_predictions = 0

    for example, correct_word in examples_with_answers:
        try:
            predictions = evaluate_mlm(example, top_k=top_k)
            predicted_words = [word.strip().lower() for word, _ in predictions]

            if correct_word.lower() in predicted_words:
                correct_predictions += 1
        except ValueError as e:
            print(f"Fehler für Eingabe '{example}': {e}")

    accuracy = correct_predictions / len(examples_with_answers)
    return accuracy

# Beispieltexte für MLM mit richtigen Antworten
mlm_examples_with_answers = [
    ("To be, or not to be, that is the [MASK].", "question"),
    ("All the world's a [MASK], and all the men and women merely [MASK].", "stage"),
    ("Shall I compare thee to a [MASK]'s day?", "summer"),
    ("If [MASK] be the food of [MASK], play on.", "music"),
]

# MLM-Ergebnisse
print("\n=== Masked Language Modeling (MLM) Ergebnisse ===")
for example, correct_word in mlm_examples_with_answers:
    try:
        predictions = evaluate_mlm(example)
        print(f"\nInput: {example}")
        for word, prob in predictions:
            print(f" - {word}: {prob:.4f}")
        print(f" - Richtige Antwort: {correct_word}")
    except ValueError as e:
        print(f"Fehler für Eingabe '{example}': {e}")

# MLM-Genauigkeit testen
accuracy = evaluate_mlm_accuracy(mlm_examples_with_answers)
print(f"\nMLM Genauigkeit: {accuracy:.2%}")
