import torch
import logging
from transformers import Trainer, TrainingArguments, BertForNextSentencePrediction, AutoTokenizer
from data_preparation_nsp import prepare_data_nsp

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    try:
        # Daten vorbereiten
        logger.info("Bereite NSP-Datasets vor...")
        nsp_datasets = prepare_data_nsp()

        if nsp_datasets is None:
            raise ValueError("Fehler beim Vorbereiten des MLM-Datensatzes.")

        # Modell und Tokenizer laden
        checkpoint = "bert-base-uncased"
        logger.info(f"Lade Modell und Tokenizer von {checkpoint}...")
        model = BertForNextSentencePrediction.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model.to(device)

        from transformers import TrainingArguments

        training_args = TrainingArguments(
            output_dir="./results_nsp",
            evaluation_strategy="epoch",  # Evaluation nach jeder Epoche
            save_strategy="epoch",  # Speichern nach jeder Epoche
            learning_rate=3e-5,  # Leicht gesenkt, um Overfitting zu vermeiden
            weight_decay=0.01,
            num_train_epochs=5,  # Erhöhe die Epochen für besseres Lernen
            per_device_train_batch_size=64,  # Moderate Batch-Größe für langsames, stabiles Lernen
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=4,  # Stabilisiert das Training bei kleinen Batch-Sizes
            save_total_limit=2,  # Spart Speicherplatz
            logging_dir="./logs",
            logging_steps=50,
            fp16 = True,
            dataloader_num_workers = 4
        )

        # Trainer initialisieren
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=nsp_datasets["train"],
            eval_dataset=nsp_datasets["validation"],
            tokenizer=tokenizer
        )

        # Training starten
        logger.info("Starte Training des NSP-Modells...")
        trainer.train()

        # Modell speichern
        logger.info("Speichere Modell...")
        trainer.save_model("./bookcorpus_nsp_model")


    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")


if __name__ == "__main__":
    main()
