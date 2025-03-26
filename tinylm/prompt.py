from transformers import Pipeline
from tqdm import tqdm
import pandas as pd
from nltk.tokenize import sent_tokenize
from tinylm.constants import *

def prompt_slm_continuation(pipe: Pipeline, prompts: list, max_new_tokens: int ,batch_size: int):
    new_prompts = list()
    for prompt in prompts:
        sentences = sent_tokenize(prompt)
        continuation = " ".join(sentences[0:min(2, len(sentences))])
        new_prompts.append(continuation)
    

    def prompt_generator():
        for prompt in new_prompts:
            yield prompt
    
    results = list()
    for result in tqdm(pipe(prompt_generator(), max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=True), total=len(prompts)):
        results.append(result[0]["generated_text"])
    
    ret = pd.DataFrame()
    ret[PROMPT] = new_prompts
    ret[TEXT] = results
    ret[LABEL] = 1
    ret[MODEL] = pipe.model.config._name_or_path
    return ret


def prompt_slm_zero_shot(pipe: Pipeline, prompts: list, max_new_tokens: int):

    results = list()
    for prompt in tqdm(prompts):
        messages = [
            {"role": "system","content":prompt}
        ]
        results.append(pipe(messages, max_new_tokens=max_new_tokens,do_sample=True)[0]["generated_text"][-1]["content"])

    ret = pd.DataFrame()
    ret[PROMPT] = prompts
    ret[TEXT] = results
    ret[LABEL] = 1
    ret[MODEL] = pipe.model.config._name_or_path
    return ret




def prompt_slm_k_shot(pipe: Pipeline, prompt: str, examples: list, max_new_tokens: int,role: str = "system"):
    formatted_prompts = [f"{prompt}\n\n" + "\n\n".join([f"Example: {example}" for example in example_set]) for example_set in examples]
    
    results = list()
    for formatted_prompt in tqdm(formatted_prompts):
        messages = [
            {"role": role,"content":formatted_prompt}
        ]
        results.append(pipe(messages, max_new_tokens=max_new_tokens,do_sample=True)[0]["generated_text"][-1]["content"])

    ret = pd.DataFrame()
    ret[PROMPT] = formatted_prompts
    ret[TEXT] = results
    ret[LABEL] = 1
    ret[MODEL] = pipe.model.config._name_or_path
    return ret



    
    