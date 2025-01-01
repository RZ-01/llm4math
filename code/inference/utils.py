# code/inference/utils.py
import re

import torch


def extract_final_grade(text):
    match = re.search(r'Is the answer correct \(Yes/No\)\? (Yes|No)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return None


def generate_prompt(data_point):
    prompt_template = (
        "You are a math teacher.\n Verify the question and answer. If the answer is incorrect, output your expected answer."
        "At the end of your verification, respond in the form \"Verification: Is the answer correct? X\", where X is Yes or No \n"
        "Q: {question}\n"
        "A: {solution}\n"
    )

    # 填充模板中的Few-Shot示例、问题和解决方案
    prompt = prompt_template.format(
        question=data_point.get("question", "No question provided."),
        solution=data_point.get("solution", "No solution provided.")
    )

    return prompt

    import re

def compute_yes_no_probability(text, tokenizer, model):
    """
    计算给定文本中最后一个与 'Verification: Is the answer correct (Yes/No)?' 匹配的 'Yes' 或 'No' 的概率比例。
    最终的分数 = P(Yes) / (P(Yes) + P(No))
    """
    final_text = text.strip()
    matches = re.findall(r"(Yes|No)", final_text)
    if matches:
        final_text = matches[-1]  # 最后一个出现的 Yes 或 No
    else:
        raise ValueError("No 'Yes' or 'No' found in the text.")

    # 对输入进行tokenize
    input_ids = tokenizer(final_text, return_tensors="pt").input_ids.to(model.device)

    yes_ids = tokenizer("Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer("No", add_special_tokens=False).input_ids

    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("Assumption violated: 'Yes' or 'No' is not a single token.")

    yes_id = yes_ids[0]
    no_id = no_ids[0]


    context_input_ids = input_ids[:, :-1]
    with torch.no_grad():
        outputs = model(context_input_ids)

    last_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    yes_prob = probs[yes_id].item()
    no_prob = probs[no_id].item()

    # 若yes_prob + no_prob过小（极不可能），也做下保护
    denom = yes_prob + no_prob
    if denom == 0:
        raise Exception("Too small of yes Token")

    return yes_prob / denom

'''
def compute_yes_no_probability(text, tokenizer, model):
    """
    计算给定文本中最终"Yes"或"No"的概率。
    """
    final_text = text.strip()

    if final_text.endswith("Yes"):
        final_token = "Yes"
    elif final_text.endswith("No"):
        final_token = "No"
    else:
        raise ValueError("Cannot find a final 'Yes' or 'No' at the end of the text.")

    input_ids = tokenizer(final_text, return_tensors="pt").input_ids.to(model.device)

    final_token_ids = tokenizer(final_token, add_special_tokens=False).input_ids
    if len(final_token_ids) != 1:
        raise ValueError(f"Final token '{final_token}' is not a single token.")

    final_token_id = final_token_ids[0]


    # 去掉最后一个token进行模型前向传播
    context_input_ids = input_ids[:, :-1]  # 除去最后一个token
    with torch.no_grad():
        outputs = model(context_input_ids)
    # outputs.logits: [batch_size, seq_length, vocab_size]
    # 最后一个位置的logits对应我们预测最终token的分布
    last_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    final_token_prob = probs[final_token_id].item()

    return final_token_prob
'''