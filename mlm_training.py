import logging
import os

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    AutoTokenizer,
    EarlyStoppingCallback
)

from data_preparation_mlm import prepare_data_mlm

# === Configure logging ===
# The logging module is used to display progress and important information during execution.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Check if a GPU is available ===
# The model will run on the GPU if available, otherwise on the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Main function for training the Masked Language Model (MLM) ===
def main():
    try:
        # --- Prepare data ---
        logger.info("Preparing MLM datasets...")
        mlm_datasets = prepare_data_mlm()

        if mlm_datasets is None:
            raise ValueError("Error preparing the MLM dataset.")

        # --- Load model and tokenizer ---
        checkpoint = './bookcorpus_mlm_model'  # --- Initial training 'distilbert-base-uncased' ---
        logger.info(f"Loading model and tokenizer from {checkpoint}...")
        model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

        # --- DataCollator with dynamic masking ---
        # The DataCollator randomly masks 15% of the tokens during training.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        # --- Set training parameters ---
        training_args = TrainingArguments(
            output_dir="./results_mlm",  # Directory to store training results
            evaluation_strategy="epoch",  # Evaluate after each epoch
            save_strategy="epoch",  # Save the model after each epoch
            learning_rate=5e-5,
            weight_decay=0.01,
            num_train_epochs=5,  # Number of training epochs
            per_device_train_batch_size=4,  # Batch size for training
            per_device_eval_batch_size=4,  # Batch size for evaluation
            gradient_accumulation_steps=2,  # Accumulate gradients
            save_total_limit=2,  # Limit the number of saved models
            logging_dir="./logs",  # Directory for logs
            logging_steps=50,  # Logging interval
            fp16=True,  # Mixed Precision Training for speedup
            dataloader_num_workers=2,  # Number of parallel data loaders
            load_best_model_at_end=True  # Load the best model at the end
        )

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=mlm_datasets["train"],
            eval_dataset=mlm_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop training if no improvement
        )

        # --- Start training ---
        logger.info("Starting training of the MLM model...")
        trainer.train()

        # --- Save model ---
        logger.info("Saving model...")
        trainer.save_model("./bookcorpus_mlm_model")

        # --- Verify saving ---
        if os.path.exists("./bookcorpus_mlm_model"):
            print("Model saved successfully.")
        else:
            raise FileNotFoundError("WARNING: Model was not saved correctly!")

        # --- Model validation ---
        logger.info("Loading saved model and tokenizer for validation...")
        loaded_model = AutoModelForMaskedLM.from_pretrained("./bookcorpus_mlm_model").to(device)
        loaded_tokenizer = AutoTokenizer.from_pretrained("./bookcorpus_mlm_model")

        # --- Example validation: Test tokenization and prediction ---
        test_text = "To be, or not to be, that is the [MASK]."
        inputs = loaded_tokenizer(test_text, return_tensors="pt").to(device)
        outputs = loaded_model(**inputs)
        predicted_index = torch.argmax(outputs.logits, dim=-1)
        predicted_token = loaded_tokenizer.decode(predicted_index[0])
        logger.info(f"Prediction completed. Predicted word: {predicted_token}")

    except Exception as e:
        logger.error(f"Error during training: {e}")

# === Script entry point ===
if __name__ == "__main__":
    main()
