import json
input_file = '/home/llm4math/LLM-for-Math/data/verification_results_MATH_Mistral_L_processed.jsonl'   
output_file = '/home/llm4math/LLM-for-Math/data/verification_results_MATH_Mistral_L_flattened.jsonl' 

extracted_data = []

with open(input_file, 'r', encoding='utf-8') as infile:
    try:
        data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        exit(1)

    if not isinstance(data, list):
        print("输入文件不是一个 JSON 数组。")
        exit(1)

    for line_number, obj in enumerate(data, start=1):
        # 检查是否存在 'steps' 字段
        steps = obj.get('steps', [])
        if not isinstance(steps, list):
            print(f"第 {line_number} 个对象的 'steps' 字段不是列表，跳过该对象。")
            continue

        # 遍历每个步骤，提取 'prompt' 和 'model_output'
        for step_number, step in enumerate(steps, start=1):
            prompt = step.get('prompt', '')
            model_output = step.get('model_output', '')

            if not prompt and not model_output:
                print(f"第 {line_number} 个对象，第 {step_number} 个步骤缺少 'prompt' 和 'model_output' 字段，跳过该步骤。")
                continue

            extracted_data.append({
                'prompt': prompt,
                'model_output': model_output
            })

# 将提取的数据保存为 JSON 文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(extracted_data, outfile, ensure_ascii=False, indent=4)

print(f"提取完成，结果已保存到 {output_file}")


"""
def extract_prompts_and_outputs_from_jsonl(json_file_path, output_file):

    从多行 JSON 文件 (JSON Lines / ndjson) 中逐行解析并提取：
      - question: 当前问题的文本
      - steps: 每个 step 的 prompt、model_output，以及根据自定义逻辑判断的 verification。
      
    自定义逻辑:
      - 如果 model_output (大小写不敏感) 中包含 "correct" 子串, 则视为 Yes
      - 否则视为 No
      - 一旦某个 step 的判断结果是 No，就不再提取该问题剩余的 step

    并将结果以 JSON 格式写到 output_file 中。

    results = []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                question_obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {line_num} 行解析失败：{e}")
                continue
            question_text = question_obj.get("question", "")

            verification_results = question_obj.get("verification_results", [])
            if not isinstance(verification_results, list):
                verification_results = []

            step_data = []
            for step_info in verification_results:
                prompt = step_info.get("prompt", "")
                model_output = step_info.get("model_output", "")

                if "incorrect" in model_output.lower():
                    verification = "No"
                else:
                    verification = "Yes"

                step_data.append({
                    "prompt": prompt,
                    "model_output": model_output,
                    "verification": verification
                })

                if verification == "No":
                    break

            results.append({
                "question": question_text,
                "steps": step_data
            })

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"提取结果已写入 {output_file}")
"""
"""
if __name__ == "__main__":
    json_file = "/home/llm4math/LLM-for-Math/data/verification_results_MATH_selected_Mistral_Large.jsonl"
    output_file = "/home/llm4math/LLM-for-Math/data/verification_results_MATH_Mistral_L_processed.jsonl"
   
    # extract_prompts_and_outputs_from_jsonl(output_file, output_file)
"""
