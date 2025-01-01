# code/training/trainer_utils.py
from trl import SFTConfig
from transformers import DataCollatorForLanguageModeling
from custom_trainer import CustomSFTTrainer

def build_training_args(args):
    return SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False
    )

def create_trainer(model, training_args, train_dataset, val_dataset, tokenizer, callbacks, peft_config):
    return CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        lambda_value=1.0 / 4,               # the ratio of Generation and Verify
        callbacks=callbacks
    )
