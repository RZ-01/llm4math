# code/model/lora_utils.py
import bitsandbytes as bnb
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def find_all_linear_names(model):
   cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
   lora_module_names = set()
   for name, module in model.named_modules():
     if isinstance(module, cls):
       names = name.split('.')
       lora_module_names.add(names[0] if len(names) == 1 else names[-1])
     if 'lm_head' in lora_module_names: \
       lora_module_names.remove('lm_head')
   return list(lora_module_names)

def prepare_lora_model(model, lora_r, lora_alpha, lora_dropout):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, lora_config
