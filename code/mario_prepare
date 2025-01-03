"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm.outputs import RequestOutput
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from code_infer import infer
from tqdm import tqdm

from mcts_math.nodes.base_node import BaseNode
from mcts_math.constants import (
    NO_VALID_CHILD, 
    TOO_MANY_STEPS, 
    TOO_MANY_CODE_ERRORS, 
    SOLUTION_COLOR, 
    OBSERVATION_COLOR,
)
from .tree import BaseTree, code_execution
from .react import REACT

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(torch.device(device))
    return model, tokenizer

class SBSREACT(REACT):
    """
    Step-level Beam Search
    """

    current_top_num: int = 1
    current_nodes: List[Type[BaseNode]] = []
    final_answer_nodes: List[Type[BaseNode]] = [] 
    candidate_nodes: List[Type[BaseNode]] = [] 
    verify_model_path: str= ''
    verify_device: str=''
    num_votes: int=1
    #verify_model: Type[None]=None
    verify_model: Type[Gemma2ForCausalLM]=None
    verify_tokenizer: Type[GemmaTokenizerFast]=None
    is_verifier: bool=False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.candidate_nodes.append(self.current_node)
        self.current_top_num = self.config.step_beam_width
        self.is_verifier = True
        self.select_next_step()
        self.verify_model_path = '/mnt/d2/wyin/mario/Super_MARIO/code_infer/results/verifier_merged_v2_large_9_epoch_2025_01_02'#self.config.verify_model_path 
        #self.verify_model_path = '/mnt/d2/wyin/mario/Super_MARIO/code_infer/results/final_model'#self.config.verify_model_path 
        Gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        print('verify_device=','cuda:{}'.format(len(Gpus)),'Gpus=', Gpus)
        self.verify_device = 'cuda:{}'.format(len(Gpus))#self.config.verify_device
        #self.verify_model, self.verify_tokenizer = load_model_and_tokenizer(self.verify_model_path, self.verify_devic)
        self.num_votes = 5#self.config.num_votes
        if self.is_verifier:
            try:
                if SBSREACT.verify_model==None:
                    SBSREACT.verify_model, SBSREACT.verify_tokenizer = load_model_and_tokenizer(self.verify_model_path, self.verify_device)
            except:
                SBSREACT.verify_model, SBSREACT.verify_tokenizer = load_model_and_tokenizer(self.verify_model_path, self.verify_device)
        else:
            SBSREACT.verify_model=None

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "sbs":
            raise ValueError(f"Wrong value for config mode, must be react")
        if not cfg.n_generate_sample >= 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be greater than 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg
    
    def create_llm(self) -> Callable[[...], List[str]]:
        # we only implement the batch inference
        pass

    def is_ignored_node(self, node: Type[BaseNode]) -> bool:
        return node.is_terminal or node.depth > self.config.max_depth

    def should_generate_next(self) -> bool:
        need_generate = False
        for step_node in self.current_nodes:
            if not self.is_ignored_node(step_node):
                need_generate = True
                break
        return need_generate

    def create_prompt(
        self,
        is_value_only: bool = False,
    ) -> str:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            partial_solution = self.collect_partial_solution(current_node)
            prompt = self.prompt_wrap(
                self.question, 
                partial_solution,
                self.config,
            )
            prompts.append(prompt)
        return prompts

    @staticmethod
    def is_valid_final_answer_node(node: Type[BaseNode]) -> bool:
        # by default, final_anwer = ""
        if node.is_terminal and node.state["final_answer"] and \
           node.state["final_answer"] not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            return True
        return False

    def collect_partial_solution_with_context(self, node: Type[BaseNode]) -> Dict[str, str]:
        """
        Collects the current node's text and the context from its parent nodes.
        Returns:
            Dict[str, str]: A dictionary containing 'Current Step' and 'Context'.
        """
        if not node:
            return {"Current Step": "", "Context": ""}
        current_step = node.state['text']
        context_trajectory = []
        parent = node.parent
        while parent:
            parent_text = parent.state.get('text', "")
            if parent_text:
                context_trajectory.append(parent_text)
            parent = parent.parent
        context = self.config.step_delim.join(reversed(context_trajectory))
        return {
            "Current Step": current_step,
            "Context": context
        }

    def select_next_step(self, outputs: Optional[List[RequestOutput]] = None) -> None:
        """process output from vllm
        e.g.,
        prompts = tree.create_prompt(is_value_only=True)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in tqdm(zip(self.candidate_nodes, outputs), desc="verify candidate_nodes"):
                # assert self.question in output.prompt
                #print('score\n',output.value_estimate)
                candidate_node.value = output.value_estimate if output.value_estimate is not None else -100
                #print(self.collect_partial_solution_with_context(candidate_node))
                if self.is_verifier:
                    if self.is_ignored_node(candidate_node):
                        continue
                    partial_solution = self.collect_partial_solution_with_context(candidate_node)
                    score = infer.get_inference_score(
                    model = SBSREACT.verify_model,
                    device = self.verify_device,
                    tokenizer = SBSREACT.verify_tokenizer,
                    question=self.question,
                    solution=partial_solution,
                    num_votes=self.num_votes,
                    max_length=4096)
                    candidate_node.value = 0.9*candidate_node.value+0.1*score
                    print(candidate_node.value)
            
        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.current_nodes = self.candidate_nodes[:self.current_top_num]

        for current_node in self.current_nodes[:]:  # must shallow copy because of the remove in the loop 
            if self.__class__.is_valid_final_answer_node(current_node):
                self.final_answer_nodes.append(current_node)
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
            elif current_node.is_terminal or current_node.depth > self.config.max_depth:
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            self.current_node = current_node
            current_output_texts = [otp.text.strip() for otp in output.outputs]
            if self.config.remove_duplicate:
                current_output_texts = set(current_output_texts)
            for idx, cur_output_text in enumerate(current_output_texts):
                step_result, parser_result = self.step_unwrap(cur_output_text)
                self._update_current_node(step_result, parser_result, idx)
            self.candidate_nodes.extend(current_node.children)

    def get_steps(self):
        final_answer_states = []
        for cur_node in self.final_answer_nodes:
            states = {
                "question": self.question,
                "ground_truth": self.ground_truth,
                "value": cur_node.value,
                "final_answer": cur_node.state["final_answer"],
                "solution": self.collect_partial_solution(cur_node),
                "tag": cur_node.tag,
            }
            final_answer_states.append(states)

        solutions = sorted(final_answer_states, key=lambda x: x['value'], reverse=True)
        return solutions

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            if node.has_children():
                candidates.extend(node.children)
        states["solutions"] = self.get_steps()
        return states
