import pandas as pd
from transformers import pipeline
from tinylm.prompt import prompt_slm_zero_shot

if __name__ == "__main__":
    pipe = pipeline("") 