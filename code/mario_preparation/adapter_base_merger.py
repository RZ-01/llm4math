from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


base_model_name = "google/gemma-2-9b-it"
adapter_model_name = "./results/checkpoint-1400"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained("./results/verifier_merged_v2_epochplus")
model = model.merge_and_unload().float()
model.save_pretrained("./results/verifier_merged_v2_epochplus")
