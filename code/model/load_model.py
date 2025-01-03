# code/model/load_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_model_and_tokenizer(model_id, device_string, quantization_bits=4):
    if quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  
            bnb_4bit_quant_type="nf4"        
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device_string},
        torch_dtype=torch.float16, 
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer