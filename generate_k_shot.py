import pandas as pd
from transformers import pipeline, AutoTokenizer
import datasets
from tinylm.prompt import prompt_slm_k_shot
import dotenv
import os
dotenv.load_dotenv()
if __name__ == "__main__":
    model_name ="nvidia/Hymba-1.5B-Instruct"

   # tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model_name,trust_remote_code=True)
   # pipe.model.generation_config.pad_token_id = tokenizer.eos_token_id
    test_set = datasets.load_dataset("yaful/MAGE",split="test")
    test_set = test_set.to_pandas()
    domain = "yelp"
    domain_test_set = test_set[(test_set["src"].str.contains(domain))]
    domain_test_set = domain_test_set[domain_test_set["label"] == 1]
    examples = domain_test_set["text"].sample(n=200)
    examples = [[example] for example in examples]
    prompt = "Generate a new review that mimics the style of the following review. Output only the text of the review."
    prompt_slm_k_shot(pipe, prompt,examples, 128).to_csv("data/yelp_hymba-1.5B-Instruct_generated_1_shot.csv",index=False)
