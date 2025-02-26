import pandas as pd
from transformers import pipeline
from tinylm.prompt import prompt_slm_zero_shot

if __name__ == "__main__":
    pipe = pipeline("text-generation", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
    print(prompt_slm_zero_shot(pipe, ["Generate a 1-star review for Yelp:"] * 4, 128, 4)["text"].values[0])
    
