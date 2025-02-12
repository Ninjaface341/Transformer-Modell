import logging

import torch
from transformers import Trainer, TrainingArguments, BertForNextSentencePrediction, AutoTokenizer

from data_preparation_nsp import prepare_data_nsp

# === Configure logging ===
# The logging module is used to display progress and important information during execution.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Check if a GPU is available ===
# The model will run on the GPU if available, otherwise on the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Main function for training the Next Sentence Prediction (NSP) model ===
def main():
    try:
        # --- Prepare data ---
        logger.info("Preparing NSP datasets...")
        nsp_datasets = prepare_data_nsp()

        if nsp_datasets is None:
            raise ValueError("Error preparing the NSP dataset.")

        # --- Load model and tokenizer ---
        checkpoint = './bookcorpus_nsp_model'  # --- Initial training 'bert-base-uncased' ---
        logger.info(f"Loading model and tokenizer from {checkpoint}...")
        model = BertForNextSentencePrediction.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model.to(device)

        # --- Set training parameters ---
        training_args = TrainingArguments(
            output_dir="./results_nsp",  # Directory to store training results
            evaluation_strategy="epoch",  # Evaluate after each epoch
            save_strategy="epoch",  # Save the model after each epoch
            learning_rate=3e-5,  # Slightly reduced learning rate to avoid overfitting
            weight_decay=0.01,  # Regularization to avoid overfitting
            num_train_epochs=5,  # Number of training epochs
            per_device_train_batch_size=64,  # Batch size for training
            per_device_eval_batch_size=64,  # Batch size for evaluation
            gradient_accumulation_steps=4,  # Accumulate gradients for stabilization
            save_total_limit=2,  # Limit the number of saved models
            logging_dir="./logs",  # Directory for logs
            logging_steps=50,  # Logging interval
            fp16=True,  # Mixed Precision Training for speedup
            dataloader_num_workers=4  # Number of parallel data loaders
        )

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=nsp_datasets["train"],
            eval_dataset=nsp_datasets["validation"],
            tokenizer=tokenizer
        )

        # --- Start training ---
        logger.info("Starting training of the NSP model...")
        trainer.train()

        # --- Save model ---
        logger.info("Saving model...")
        trainer.save_model("./bookcorpus_nsp_model")

    except Exception as e:
        logger.error(f"Error during training: {e}")

# === Script entry point ===
if __name__ == "__main__":
    main()
