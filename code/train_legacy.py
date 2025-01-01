"""
import os
import torch
import wandb
import transformers

from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from accelerate import PartialState
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb

from dataset_loader import verify_dataset, validation_dataset

device_string = PartialState().process_index
print(f"device_string is {device_string}")
wandb.init(
    project="gemma_sft_project_lora",  # W&B project name
    name="gemma_train_run_9b_H100",       # name of this run
    config={
        "model_name": "google/gemma-2-9b",
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-6,
        "weight_decay": 1e-2,
        "max_grad_norm": 1.0,
        "num_train_epochs": 3,
        "warmup_steps": 1000,
        "lr_scheduler_type": "cosine"
    }
)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_id = "google/gemma-2-9b"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map={"":device_string})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.padding_side = 'right'
print(f"Tokenizer max length: {tokenizer.model_max_length}")

os.environ["WANDB_WATCH"]="false"

def generate_prompt(data_point):
    prefix_text = 'Below is an instruction that describes a verify task. Write a response that appropriately completes the request.\\n\\n'
    # Samples with additional context into.
    return text


text_column = [generate_prompt(data_point) for data_point in verify_dataset]
verify_dataset = verify_dataset.add_column("prompts", text_column)
verify_dataset = verify_dataset.shuffle(seed=1234)
verify_dataset = verify_dataset.map(lambda samples: tokenizer(samples["prompts"]), batched=True)

text_column = [generate_prompt(data_point) for data_point in validation_dataset]
val_dataset = validation_dataset.add_column("prompts", text_column)
val_dataset = val_dataset.shuffle(seed=1234)
val_dataset = val_dataset.map(lambda samples: tokenizer(samples["prompts"]), batched=True)

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()
print("Train dataset length:", len(verify_dataset))
print("Validation dataset length:", len(val_dataset))

training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,        # 梯度累积（以达到有效批次大小为 64）
    learning_rate=2e-6,
    weight_decay=1e-2,
    max_grad_norm=0.3,
    num_train_epochs=3,
    warmup_steps=1000,
    save_steps=1000,
    lr_scheduler_type="cosine",
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",
    max_seq_length=2048,
    gradient_checkpointing=True,
    # fp16=True,
    # deepspeed="ds_config.json",
    ddp_find_unused_parameters=False
)

class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)

# Lora configuration
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 定义查找线性层的函数
def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

# 查找并打印线性层名称
modules = find_all_linear_names(model)
print(modules)

# 配置Lora
lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用Lora配置到模型
model = get_peft_model(model, lora_config)
# Print the number of trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=verify_dataset,
    dataset_text_field="prompts",
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # lambda_value=1.0 / 4,               # the ratio of Generation and Verify
    callbacks=[WandbLoggingCallback()]
)
print("trainer loaded!")

trainer.train(resume_from_checkpoint=True)

trainer.save_model("./models")

results = trainer.evaluate()
print(f"Evaluation results: {results}")

wandb.finish()
"""