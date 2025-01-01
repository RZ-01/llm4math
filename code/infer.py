# code/inference/infer.py
import json
import wandb
import logging
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from inference.utils import extract_final_grade, compute_yes_no_probability
from inference.utils import generate_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    return model, tokenizer


def perform_inference(model, tokenizer, prompt, device, num_votes=1, max_length=350, batch_size=2):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,               
        num_return_sequences=num_votes,
        do_sample=True,              
        top_k=32,                   
        temperature=0.7,              
        #repetition_penalty=1.1,      
        eos_token_id=tokenizer.eos_token_id
    )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # print(generated_texts)
    scores = []
    for text in generated_texts:
        grade = compute_yes_no_probability(text, tokenizer, model)
        scores.append(grade)

    if not scores:
        raise ValueError("All generated texts failed to extract scores.")

    average_score = sum(scores) / len(scores)

    return average_score, generated_texts

def perform_inference_score_only(model, tokenizer, prompt, device, num_votes=7, max_length=350):
    """
    供外部程序调用的函数：
    输入：
      - model_path: 模型路径 (本地或远程)
      - question: 最初的问题
      - solution: 特定步骤的回答
      - num_votes: 越高理论上越好
      - max_length: 一般不高于512
    输出：
      - score only
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_votes,
        do_sample=True,
        top_k=32,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    scores = []
    for text in generated_texts:
        score = compute_yes_no_probability(text, tokenizer, model)
        scores.append(score)

    if not scores:
        raise ValueError("All generated texts failed to extract scores.")
    average_score = sum(scores) / len(scores)
    return average_score
def get_inference_score(model_path, question, solution, num_votes=1, max_length=350):
    """
    供外部程序调用的函数：
    输入：
      - model_path: 模型路径 (本地或远程)
      - question: 问题文本
      - solution: 解答文本
      - num_votes: 生成几条结果，默认 1
      - max_length: 推理最大长度，默认 350
    输出：
      - score only (一个浮点数)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    model.eval()

    prompt = generate_prompt({"question": question, "solution": solution})
    score = perform_inference_score_only(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        num_votes=num_votes,
        max_length=max_length
    )
    return score

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args)
    )
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    model.eval()

    logger.info(f"Loading test data from {args.test_data_path}")
    test_samples = []
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            test_samples.append(sample)

    logger.info(f"Number of test samples: {len(test_samples)}")

    inference_results = []

    for sample in tqdm(test_samples, desc="Inference"):
        question = sample.get("question")
        solution = sample.get("solution")
        prompt = generate_prompt({"question": question, "solution": solution})

        average_score, generated_texts = perform_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            num_votes=args.num_votes,
            batch_size=args.batch_size
        )

        inference_results.append({
            "question": question,
            "solution": solution,
            "average_score": average_score,
            "generated_texts": generated_texts
        })

    logger.info(f"Saving inference results to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for result in inference_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    wandb.finish()


if __name__ == "__main__":
    from config import parse_args
    args = parse_args()
    if args.mode == "inference":
        main(args)
    else:
        raise Exception(f"Unknown mode: {args.mode}")
