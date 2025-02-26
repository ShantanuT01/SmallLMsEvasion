from transformers import Pipeline
from tqdm import tqdm
import pandas as pd

from tinylm.constants import *

def prompt_slm_zero_shot(pipe: Pipeline, prompts: list, max_new_tokens: int ,batch_size: int):
    def prompt_generator():
        for prompt in prompts:
            yield prompt
    
    results = list()
    for result in tqdm(pipe(prompt_generator(), max_new_tokens=max_new_tokens, batch_size=batch_size), total=len(prompts)):
        results.append(result[0]["generated_text"])
    
    ret = pd.DataFrame()
    ret[PROMPT] = prompts
    ret[TEXT] = results
    ret[LABEL] = 1
    ret[MODEL] = pipe.model.config._name_or_path
    return ret


def prompt_slm_k_shot(pipe: Pipeline, prompt: str, examples: list, max_new_tokens: int, batch_size: int):
    formatted_prompts = [f"{prompt}\n\n" + "\n\n".join([f"Example: {example}" for example in example_set]) for example_set in examples]
    def prompt_generator():
        for formatted_prompt in formatted_prompts:
            yield formatted_prompt
    
    results = list()
    for result in tqdm(pipe(prompt_generator(), max_new_tokens=max_new_tokens, batch_size=batch_size), total=len(examples)):
        results.append(result)

    ret = pd.DataFrame()
    ret[PROMPT] = formatted_prompts
    ret[TEXT] = results
    ret[LABEL] = 1
    ret[MODEL] = pipe.model.config._name_or_path
    return ret



    
    