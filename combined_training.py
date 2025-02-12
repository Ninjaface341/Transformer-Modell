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

# === Configure logging ===
# The logging module is used to display progress and important information during execution.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Check if a GPU is available ===
# The model will run on GPU if available, otherwise on CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Main function for combined training of MLM and NSP ===
def main():
    try:
        # --- Prepare combined datasets for MLM and NSP ---
        logger.info("Preparing combined dataset for MLM and NSP...")
        datasets, tokenizer = prepare_combined_data()

        # --- Load model ---
        logger.info("Loading BertForPreTraining model...")
        model = BertForPreTraining.from_pretrained("./bert_combined_model").to(device)

        # --- DataCollator with dynamic masking ---
        # The DataCollator randomly masks 15% of the tokens during training.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # --- Set training parameters ---
        training_args = TrainingArguments(
            output_dir="./results_combined",  # Directory to store training results
            eval_strategy="epoch",  # Evaluate after each epoch
            save_strategy="epoch",  # Save after each epoch
            save_steps=1000,  # Save every 1000 steps
            learning_rate=3e-5,  # Learning rate
            weight_decay=0.01,  # Regularization to avoid overfitting
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=8,  # Batch size for training
            per_device_eval_batch_size=8,  # Batch size for evaluation
            gradient_accumulation_steps=1,  # Faster updates
            save_total_limit=2,  # Limit the number of saved models
            logging_dir="./logs",  # Directory for logs
            logging_steps=100,  # Less logging for better performance
            fp16=True,  # Mixed Precision Training for speedup
            fp16_opt_level="O1",  # More stable mixed precision
            dataloader_num_workers=4,  # More workers for faster data processing
            load_best_model_at_end=True  # Load the best model at the end
        )

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),  # Stop training if no improvement
            ]
        )

        # --- Start training ---
        logger.info("Starting training of MLM and NSP simultaneously...")
        trainer.train()

        # --- Save model ---
        logger.info("Saving the combined model...")
        trainer.save_model("./bert_combined_model")

        # --- Verify model saving ---
        if os.path.exists("./bert_combined_model"):
            logger.info("Model saved successfully.")
        else:
            raise FileNotFoundError("WARNING: Model was not saved correctly!")

    except Exception as e:
        logger.error(f"Error during combined training: {e}")
        torch.cuda.empty_cache()  # Clear memory in case of errors

# === Script entry point ===
if __name__ == "__main__":
    main()
