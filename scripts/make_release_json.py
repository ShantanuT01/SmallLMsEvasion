import pandas as pd
import os
import datasets

if __name__ == "__main__":
    test_set = datasets.load_dataset("yaful/MAGE",split="test")
    test_set = test_set.to_pandas()
    files = os.listdir("release/")
    dataframes = list()
    trimmed_strings = ["Here's a new review","Here's a new abstract", "Here is a new review","Here is a new abstract","Here is a potential abstract" ]
    filtered_file = "sci_gen_base_model_zero_shot_finetuned.json"
    for file in files:
        
        if file.endswith(".json"):
            df = pd.read_json(f"release/{file}")
            if "1_shot" in file:
                prompt_strategy = "1-shot"
            elif "zero_shot" in file:
                prompt_strategy = "zero-shot"
            elif "continuation" in file:
                prompt_strategy = "continuation"
            if "finetuned" in file:
                prompt_strategy += "-finetuned"
            df["prompt-strategy"] = prompt_strategy
            texts = list()
            for text in df["text"].values:
                for string in trimmed_strings:
                    if string in text:
                        text = "".join(text.split(":")[1:])
                        text = text.strip()
                        break
                texts.append(text)
            df["text"] = texts
            if file == filtered_file:
                select_indices = list()
                for i in range(len(df)):
                    if df["text"].values[i].startswith("I can"):
                        continue
                    else:
                        select_indices.append(i)
                df = df.iloc[select_indices]
            dataframes.append(df)
    test_frame = pd.concat(dataframes)
    print(test_frame.groupby(["domain","model"])["model"].value_counts())
    domains = test_frame["domain"].unique()
    print("Found domains:",len(domains))
    for domain in domains:
        domain_test_set = test_set[(test_set["src"].str.contains(domain))]
        domain_test_set = domain_test_set[domain_test_set["label"] == 1]
        domain_test_set["label"] = 0
        domain_test_set["domain"] = domain
        domain_test_set["model"] = "human"
        domain_test_set["prompt-strategy"] = "human"
        dataframes.append(domain_test_set[["text","label","domain","model","prompt-strategy"]])
    
    pd.concat(dataframes).to_csv("release/prompting_test.csv",index=False)
    