from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

path_with_tilde = "~/.cache/huggingface/hub/gemma-2-9b-it"
expanded_path = os.path.expanduser(path_with_tilde)

base_model_name = expanded_path
adapter_model_name = "./results/final_model"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained("./results/verifier_merged")
model = model.merge_and_unload().float()
model.save_pretrained("./results/verifier_merged")
