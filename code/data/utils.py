# code/data/utils.py
def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer
    :param data_point: dict: Data point
    :return: dict: tokenized prompt
    """
    prefix_text = 'Below is an instruction that describes a verify task. Write a response that appropriately completes the request.\\n\\n'
    text = f"""<start_of_turn>user {prefix_text} {data_point["prompt"]} <end_of_turn>\\n<start_of_turn>model{data_point["model_output"]} <end_of_turn>"""
    return text