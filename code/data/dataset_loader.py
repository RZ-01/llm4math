# code/data/dataset_loader.py
import os,json
from datasets import load_dataset, Dataset
def load_train_val_datasets(dataset_dir, tokenizer, generate_prompt_fn):
    file = "verification_results_MATH_Mistral_L_flattened.jsonl"
    file_path = os.path.join(dataset_dir, file)
    dataset = load_dataset('json', data_files=file_path, split='train')
    # Split into train and eval
    verify_dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    verify_data = verify_dataset_split['train']
    eval_data = verify_dataset_split['test']
    
    # Function to generate prompts and tokenize
    def prepare_dataset(dataset, split_name):
        # Generate prompts
        prompts = [generate_prompt_fn(data_point) for data_point in dataset]
        dataset = dataset.add_column("prompts", prompts)
        
        # Tokenize prompts
        dataset = dataset.map(
            lambda samples: tokenizer(samples["prompts"], padding=True, truncation=True),
            batched=True,
            desc=f"Tokenizing {split_name}"
        )
        return dataset

    verify_data = prepare_dataset(verify_data, "train")
    eval_data = prepare_dataset(eval_data, "validation")

    return verify_data, eval_data

import json

def rename_key(filepath, old_key="model_output", new_key="completion"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):  # 处理 JSON 数组
            for item in data:
                if old_key in item:
                    item[new_key] = item.pop(old_key)
        elif isinstance(data, dict):  # 处理 JSON 对象
            if old_key in data:
                data[new_key] = data.pop(old_key)

        with open("../../../data/verification_results_MATH_Mistral_L_input.jsonl", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 使用示例
filepath = "../../../data/verification_results_MATH_Mistral_L_flattened.jsonl"  
rename_key(filepath)

"""
    # Generation Dataset
    gsm8k_dataset = load_dataset("openai/gsm8k",'main', split="train")
    generate_data = gsm8k_dataset.map(lambda x: {"prompt": x["question"], "completion": x["answer"], "task_type": "generate"})
    eval_data = load_dataset("openai/gsm8k",'main', split="test")
    eval_data = eval_data.map(lambda x: {"prompt": x["question"], "completion": x["answer"], "task_type": "generate"})
"""
    # Mix Verify dataset with Generate dataset
    #mixed_data = {
    #    "prompt": verify_dataset["prompt"] + generate_data["prompt"],
    #    "completion": verify_dataset["completion"] + generate_data["completion"],
    #    "task_type": verify_dataset["task_type"] + generate_data["task_type"]
    #}
    # generate_data = Dataset.from_dict(generate_data).shuffle(seed=42)

    # 验证数据集的验证集
"""
    correct_val_file_path = os.path.join(dataset_dir, "critiques_correct_solutions_batch_16_val.jsonl")
    incorrect_val_file_path = os.path.join(dataset_dir, "critiques_incorrect_solutions_batch_16_val.jsonl")

    correct_val_dataset = load_dataset("json", data_files=correct_val_file_path)
    incorrect_val_dataset = load_dataset("json", data_files=incorrect_val_file_path)
    correct_val_processed = correct_val_dataset.map(lambda x: {"input_text": x["inputs"], "target_text": x["targets"], "task_type": "verify"})
    incorrect_val_processed = incorrect_val_dataset.map(lambda x: {"input_text": x["inputs"], "target_text": x["targets"], "task_type": "verify"})
    val_input_text = correct_val_processed["train"]["input_text"] + incorrect_val_processed["train"]["input_text"]
    val_target_text = correct_val_processed["train"]["target_text"] + incorrect_val_processed["train"]["target_text"]
    val_task_type = correct_val_processed["train"]["task_type"] + incorrect_val_processed["train"]["task_type"]

    validation_data = {
        "prompt": val_input_text,
        "completion": val_target_text,
        "task_type": val_task_type
    }
    # eval_data = Dataset.from_dict(eval_data)
"""
    