# code/model/load_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_model_and_tokenizer(model_id, device_string, quantization_bits=4):
    quantization_config = BitsAndBytesConfig(load_in_4bit=(quantization_bits == 4))
    model = AutoModelForCausalLM.from_pretrained(model_id,  device_map={"": device_string}, torch_dtype=torch.float32, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer