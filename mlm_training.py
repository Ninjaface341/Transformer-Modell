import torch
import os
import logging
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from data_preparation_mlm import prepare_data_mlm

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    try:
        # Daten vorbereiten
        logger.info("Bereite MLM-Datasets vor...")
        mlm_datasets = prepare_data_mlm()

        if mlm_datasets is None:
            raise ValueError("Fehler beim Vorbereiten des MLM-Datensatzes.")

        # Modell und Tokenizer für MLM laden
        checkpoint = "distilbert-base-cased"
        logger.info(f"Lade Modell und Tokenizer von {checkpoint}...")
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        model.to(device)


        # DataCollator mit dynamischer Maskierung
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        # Trainingsparameter
        training_args = TrainingArguments(
            output_dir="./results_mlm",
            evaluation_strategy="epoch",
            save_strategy="epoch",  # Speichern nach jeder Epoche
            learning_rate=5e-5,
            weight_decay=0.01,
            num_train_epochs=5,  # Weniger Epochen für schnellere Tests / Mehr Epochen für genaure Tests
            per_device_train_batch_size=4,  # Kleinere Batch-Größe für weniger Speicherverbrauch
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Erhöhen für stabileres Training
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=50,  # Häufigere Logs zur Fortschrittskontrolle
            fp16=True,
            dataloader_num_workers=2,  # Weniger Worker für stabileres Training

        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=mlm_datasets["train"],
            eval_dataset=mlm_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Training starten
        logger.info("Starte Training des MLM-Modells...")
        trainer.train()

        # Modell speichern
        logger.info("Speichere Modell...")
        trainer.save_model("./bookcorpus_mlm_model")

        # Überprüfung des Speicherns
        if os.path.exists("./bookcorpus_mlm_model"):
            print("Modell erfolgreich gespeichert.")
        else:
            raise FileNotFoundError("WARNUNG: Modell wurde nicht korrekt gespeichert!")

        # Validierung des gespeicherten Modells
        logger.info("Lade gespeichertes Modell und Tokenizer zur Validierung...")
        loaded_model = AutoModelForMaskedLM.from_pretrained("./bookcorpus_mlm_model")
        loaded_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")  # Muss übereinstimmen!

        # Beispiel-Validierung: Test-Tokenisierung und Vorhersage
        test_text = "To be, or not to be, that is the [MASK]."
        inputs = loaded_tokenizer(test_text, return_tensors="pt").to(device)
        outputs = loaded_model(**inputs)
        predicted_index = torch.argmax(outputs.logits, dim=-1)
        predicted_token = loaded_tokenizer.decode(predicted_index[0])
        logger.info(f"Vorhersage abgeschlossen. Vorhergesagtes Wort: {predicted_token}")

    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")


if __name__ == "__main__":
    main()
