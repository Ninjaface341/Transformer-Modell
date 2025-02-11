# === combined_training.py ===
import os
import torch
import logging
from transformers import (
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from data_preparation_combined import prepare_combined_data

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU/CPU-Konfiguration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training von MLM und NSP
def main():
    try:
        logger.info("Bereite kombiniertes Dataset f\u00fcr MLM und NSP vor...")
        datasets, tokenizer = prepare_combined_data()

        logger.info("Lade Modell BertForPreTraining...")
        model = BertForPreTraining.from_pretrained("bert-base-uncased").to(device)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir="./results_combined",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=50,
            fp16=True,
            dataloader_num_workers=4,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        logger.info("Starte Training von MLM und NSP gleichzeitig...")
        trainer.train()

        logger.info("Speichere das kombinierte Modell...")
        trainer.save_model("./bert_combined_model")

        if os.path.exists("./bert_combined_model"):
            logger.info("Modell erfolgreich gespeichert.")
        else:
            raise FileNotFoundError("WARNUNG: Modell wurde nicht korrekt gespeichert!")

    except Exception as e:
        logger.error(f"Fehler beim kombinierten Training: {e}")


if __name__ == "__main__":
    main()
