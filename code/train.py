# code/train.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
import logging
import wandb, torch
from accelerate import PartialState

from data.dataset_loader import load_train_val_datasets
from data.utils import generate_prompt
from model.load_model import load_model_and_tokenizer
from model.lora_utils import prepare_lora_model
from training.trainer_utils import build_training_args, create_trainer
from training.callbacks import WandbLoggingCallback, PeftSavingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):


    device_string = PartialState().process_index
    logger.info(f"Using device string: {device_string}")

    # Initialize WandB
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args)
    )

    os.environ["WANDB_WATCH"] = "false"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, device_string, args.quantization_bits)
    logger.info(f"Tokenizer max length: {tokenizer.model_max_length}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = load_train_val_datasets(
        dataset_dir=args.dataset_dir
    )

    logger.info(f"Train dataset length: {len(train_dataset)}")
    logger.info(f"Validation dataset length: {len(val_dataset)}")

    # Prepare LoRA
    model, lora_config = prepare_lora_model(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    # Build trainer
    training_args = build_training_args(args)
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        callbacks=[WandbLoggingCallback(), PeftSavingCallback()]
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=False)

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache before evaluation...")
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared.")
    else:
        logger.info("CUDA is not available, skipping memory clearing.")

    # Evaluate
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")

    wandb.finish()

if __name__ == "__main__":
    from config import parse_args
    args = parse_args()
    if args.mode == "train":
        main(args)
    else:
        raise Exception(f"Unknown mode: {args.mode}")
