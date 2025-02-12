import logging
import os

import torch
from transformers import (
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from data_preparation_combined import prepare_combined_data

# === Logging konfigurieren ===
# Das Logging-Modul wird verwendet, um den Fortschritt und wichtige Informationen während der Ausführung anzuzeigen.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Überprüfen, ob eine GPU verfügbar ist ===
# Das Modell wird auf der GPU ausgeführt, falls verfügbar, andernfalls auf der CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hauptfunktion zum kombinierten Training von MLM und NSP ===
def main():
    try:
        # --- Vorbereitung der kombinierten Datensätze für MLM und NSP ---
        logger.info("Bereite kombiniertes Dataset für MLM und NSP vor...")
        datasets, tokenizer = prepare_combined_data()

        # --- Modell laden ---
        logger.info("Lade Modell BertForPreTraining...")
        model = BertForPreTraining.from_pretrained("./bert_combined_model").to(device)

        # --- DataCollator mit dynamischer Maskierung ---
        # Der DataCollator maskiert zufällig 15% der Tokens während des Trainings.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # --- Trainingsparameter festlegen ---
        training_args = TrainingArguments(
            output_dir="./results_combined",  # Speicherort für die Trainingsergebnisse
            eval_strategy="epoch",  # Evaluation nach jeder Epoche
            save_strategy="epoch",  # Speichern nach jeder Epoche
            save_steps=1000,  # Speichert alle 1000 Schritte
            learning_rate=3e-5,  # Lernrate
            weight_decay=0.01,  # Regularisierung zur Vermeidung von Overfitting
            num_train_epochs=3,  # Anzahl der Trainingsepochen
            per_device_train_batch_size=8,  # Batch-Größe für das Training
            per_device_eval_batch_size=8,  # Batch-Größe für die Validierung
            gradient_accumulation_steps=1,  # Schnellere Updates
            save_total_limit=2,  # Begrenzung der gespeicherten Modelle
            logging_dir="./logs",  # Verzeichnis für Logs
            logging_steps=100,  # Weniger Logging für mehr Performance
            fp16=True,  # Mixed Precision Training zur Beschleunigung
            fp16_opt_level="O1",  # Stabilere Mixed Precision
            dataloader_num_workers=4,  # Mehr Worker für schnellere Datenverarbeitung
            load_best_model_at_end=True  # Bestes Modell am Ende laden
        )

        # --- Trainer initialisieren ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),  # Stoppt das Training bei ausbleibender Verbesserung
            ]
        )

        # --- Training starten ---
        logger.info("Starte Training von MLM und NSP gleichzeitig...")
        trainer.train()

        # --- Modell speichern ---
        logger.info("Speichere das kombinierte Modell...")
        trainer.save_model("./bert_combined_model")

        # --- Überprüfung des Speicherns ---
        if os.path.exists("./bert_combined_model"):
            logger.info("Modell erfolgreich gespeichert.")
        else:
            raise FileNotFoundError("WARNUNG: Modell wurde nicht korrekt gespeichert!")

    except Exception as e:
        logger.error(f"Fehler beim kombinierten Training: {e}")
        torch.cuda.empty_cache()  # Speicherfreigabe bei Fehlern

# === Startpunkt des Skripts ===
if __name__ == "__main__":
    main()
