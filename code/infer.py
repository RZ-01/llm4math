# code/inference/infer.py
import json
import wandb
import logging
from tqdm import tqdm

import numpy as np
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


def perform_inference(model, tokenizer, prompt, device, num_votes=1, max_length=512, batch_size=2):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    all_outputs = []
    all_generated_texts = []
    all_transition_scores = []

    max_regeneration_attempts = 5

    for _ in range(num_votes):
        regeneration_attempts = 0
        while regeneration_attempts < max_regeneration_attempts:
            try:
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    num_beams=4,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                generated_text = decoded[len(prompt):].strip()

                if generated_text:  
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
                    all_outputs.append(outputs)
                    all_generated_texts.append(generated_text)
                    all_transition_scores.append(transition_scores)
                    break  
                else:
                    regeneration_attempts += 1
                    print(f"Generated empty text, regenerating attempt {regeneration_attempts}/{max_regeneration_attempts}")

            except Exception as e:
                print(f"Error in generation: {str(e)}")
                regeneration_attempts += 1
                if regeneration_attempts == max_regeneration_attempts:
                    all_generated_texts.append("")
                    all_transition_scores.append(None)
                break 

        if regeneration_attempts == max_regeneration_attempts and not all_generated_texts:
            all_generated_texts.append("")
            all_transition_scores.append(None)
        elif regeneration_attempts == max_regeneration_attempts and len(all_generated_texts) <= _: # Handle case where some regenerations failed
            all_generated_texts.append("")
            all_transition_scores.append(None)

    yes_scores = []
    no_scores = []

    for i, text in enumerate(all_generated_texts):
        if all_transition_scores[i] is not None:
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = all_outputs[i].sequences[:, input_length:]
            for tok, score in zip(generated_tokens[0], all_transition_scores[i][0]):
                decoded_token = tokenizer.decode(tok)
                if 'Yes' in decoded_token:
                    yes_scores.append(np.exp(score.cpu().numpy()))
                elif 'No' in decoded_token:
                    no_scores.append(np.exp(score.cpu().numpy()))

    yes_count = len([text for text in all_generated_texts if 'Yes' in text])
    no_count = len([text for text in all_generated_texts if 'No' in text])


    final_grade = 0
    yes_or_no = "Yes"
    if yes_count > no_count:
        if yes_scores:
            final_grade = np.mean(yes_scores)
    elif no_count > yes_count:
        if no_scores:
            final_grade = -np.mean(no_scores)
            yes_or_no = "No"
    else:
        print("Tie in votes, final grade is 0.")
        yes_or_no = "None"

    return yes_or_no, final_grade

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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    all_generated_texts = []
    all_transition_scores = []

    max_regeneration_attempts = 5

    for _ in range(num_votes):
        regeneration_attempts = 0
        while regeneration_attempts < max_regeneration_attempts:
            try:
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    num_beams=4,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                generated_text = decoded[len(prompt):].strip()

                if generated_text:
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
                    all_generated_texts.append(generated_text)
                    all_transition_scores.append(transition_scores)
                    break
                else:
                    regeneration_attempts += 1
                    print(f"Generated empty text, regenerating attempt {regeneration_attempts}/{max_regeneration_attempts}")

            except Exception as e:
                print(f"Error in generation: {str(e)}")
                regeneration_attempts += 1
                if regeneration_attempts == max_regeneration_attempts:
                    all_generated_texts.append("")
                    all_transition_scores.append(None)
                break

        if regeneration_attempts == max_regeneration_attempts and not all_generated_texts:
            all_generated_texts.append("")
            all_transition_scores.append(None)
        elif regeneration_attempts == max_regeneration_attempts and len(all_generated_texts) <= _:
            all_generated_texts.append("")
            all_transition_scores.append(None)

    yes_probabilities = []
    no_probabilities = []

    for i, text in enumerate(all_generated_texts):
        if all_transition_scores[i] is not None:
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = all_transition_scores[i].input_ids[:, input_length:] # 从 transition_scores 中获取 input_ids
            for tok_id, score in zip(generated_tokens[0], all_transition_scores[i].scores[0]):
                decoded_token = tokenizer.decode(tok_id)
                if 'Yes' in text and 'Yes' in decoded_token: # 确保文本中包含 "Yes"
                    yes_probabilities.append(np.exp(score.cpu().numpy()))
                elif 'No' in text and 'No' in decoded_token: # 确保文本中包含 "No"
                    no_probabilities.append(np.exp(score.cpu().numpy()))

    yes_count = len([text for text in all_generated_texts if 'Yes' in text])
    no_count = len([text for text in all_generated_texts if 'No' in text])
    final_grade = 0
    if yes_count > no_count:
        if yes_probabilities:
            final_grade = np.mean(yes_probabilities)
            print(f"Majority is 'Yes', final grade (average 'Yes' probability): {final_grade:.2%}")
    elif no_count > yes_count:
        if no_probabilities:
            final_grade = -np.mean(no_probabilities)
            print(f"Majority is 'No', final grade (average 'No' probability): {final_grade:.2%}")
    else:
        print("Tie in votes, final grade is 0.")

    return final_grade
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
        prompt = generate_prompt(sample)

        result, average_score = perform_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            num_votes=args.num_votes,
            batch_size=args.batch_size
        )

        inference_results.append({
            "question": sample["question"],
            "solution": sample["solution"],
            "context":sample["context"],
            "average_score": average_score,
            "Majority Vote": result
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
