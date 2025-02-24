from transformers import Pipeline
from tqdm import tqdm


def prompt_slm(pipe: Pipeline, prompts: list, batch_size: int):
    def prompt_generator():
        for prompt in prompts:
            yield prompt
    
    results = list()
    for result in tqdm(pipe(prompt_generator(), batch_size=batch_size), total=len(prompts)):
        results.append(result)
    
    return results


def prompt_slm_one_shot(pipe: Pipeline, prompt: str, one_shot_examples: list, batch_size: int):
    def prompt_generator():
        for example in one_shot_examples:
            formatted_prompt = f"{prompt}\n\nExample:\n\n{example}\n\n"
            yield formatted_prompt
    
    results = list()
    for result in tqdm(pipe(prompt_generator(), batch_size=batch_size), total=len(one_shot_examples)):
        results.append(result)
    
    return results



    
    