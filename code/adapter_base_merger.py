from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "MARIO-Math-Reasoning/AlphaMath-7B"
adapter_model_name = "/mnt/d2/wyin/DPO/AlphaMath-7B_DPO_QLora_Adapter"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained("merged_adapters")
model = model.merge_and_unload().float()
model.save_pretrained("merged_adapters")
