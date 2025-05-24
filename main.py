from model import qwen, LLM, llama

from vlmeval.dataset import build_dataset
from mctstask import MCTS_Task
from node import treeNode
from mcts import selectNode, get_next_steps_expand, expand

import ast
import torch
import os
import os.path as osp
import pandas as pd
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from easydict import EasyDict
from vllm import LLM
# from model import llm
import torch

prm = load_prm(config)
model, tokenizer = LLM('qwen')
#model = LLM(model="meta-llama/Llama-3.2-1B-Instruct",dtype="float16")
def run(args):
   
    print('-'*30, 'Begin testing', '-'*30, '\n')
    try:
        if args.dataset == 'math':
            df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)
            data_len = len(df)
        elif args.dataset == 'aime':
            df = pd.read_parquet("hf://datasets/HuggingFaceH4/aime_2024/data/train-00000-of-00001.parquet")              
            data_len = len(df)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    
    

    output_list = []
    correct_count = 0

    for row in range(len(df)):
        if args.dataset == 'math':
          question = df['problem'][row] 
        elif args.dataset == 'aime':  
          question = df['problem'][row]  
        args.iteration_limit = 4
        args.roll_forward_step = 2

        Task = MCTS_Task(question, model, tokenizer, prm, args.propose_method, args.value_method, args.branch, args.end_gate,
                            args.roll_policy, args.roll_branch, args.roll_forward_steps, args.time_limit,
                            args.iteration_limit, args.exploration_constant, args.alpha, args.inf,
                            args.temperature, use_case_prompt=args.use_case_prompt, use_reflection=args.use_reflection,
                            low=args.low, high=args.high, evaluate=args.evaluate)
        output, root = Task.run()
        output_list.append(output['solution'])
    if args.evaluate:
        pd.DataFrame(output_list, columns=['solution'])
        #output            
        #dataset.data = dataset.data.drop(columns=['image'])  
        #output_path = '/workspace/last/dataset.xlsx'
        #dataset.data.to_excel(output_path, index=False)
    return output
    
#  MllamaForConditionalGeneration

def get_args():
    args = EasyDict({
        'task_name': 'scibench',
        'file': 'thermo_standardized',
        'propose_method': 'qwen',  # choices: ['qwen', 'gpt', 'llamaV_o1', 'llama3', 'llava']
        'value_method': 'qwen',  # choices: ['gpt', 'glm', 'local']
        'mode': 'mcts',  # choices: ['cot', 'tot', 'mcts']
        'temperature': 0.7,
        'time_limit': None,
        'iteration_limit': 6,
        'roll_forward_steps': 2, #2단계 simulation
        'roll_branch': 3, #다음 step에서 몇개의 step을 생성할지
        'roll_policy': 'greedy',  # choices: ['random', 'greedy']
        'exploration_constant': 0.4,
        'end_gate': 9.0,  # End threshold
        'branch': 3,
        'inf': 0.8,
        'evaluate': 'mathvista',  # Empty string means no evaluation
        'alpha': 0.5,
        'visualize': False,  # visualization
        'use_case_prompt': False,  # Use sample prompts
        'use_reflection': 'simple',  # choices: ['simple', 'common']
        'low': 0.0,
        'high': 1.0,
        'algorithm': 'dfs',  # choices: ['dfs', 'bfs']
        'select_branch': 2,
        'max_depth': 8,
        'select_method': 'greedy',  # choices: ['greedy', 'sample']
        'consistency': True,
        'model': '',
        'dataset': 'MathVista_MINI',
        'judge_args': None,
        'llamaV_o1': None,
        'Qwen2_5': "Qwen/Qwen2.5-VL-3B-Instruct",
        'llama3_vision_11b_model_path': None,
        'llava_next_8b_model_path': None,
        'openai_api_key': None,
        'openai_base_url': 'https://api.openai.com/v1',
        'dataset' :  'aime',
    

    })
    return args

# Example usage
args = get_args()
print(args.task_name)  # 'scibench'

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    run(args)
  
