# Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd
import tqdm
pipe = pipeline("text-classification", model="michellejieli/NSFW_text_classifier")

prompts = pd.read_csv("wp_prompts.csv")["prompt"].to_list()
selected_prompts = list()
for prompt in tqdm.tqdm(prompts):
    output = pipe(prompt)[0]
    if output["label"] == "NSFW":
        continue
    else:
        selected_prompts.append(prompt)
print(len(selected_prompts))
pd.DataFrame({"prompt":selected_prompts}).sample(n=200).to_csv("filtered_prompts.csv",index=False)
