import pandas as pd
import os


if __name__ == "__main__":
    root_folder = "../data/"
    files = os.listdir(root_folder)
    frames = list()
    for file in files:
        if file == "test.csv":
            continue
        if file.endswith(".csv"):
            df = pd.read_csv(f"{root_folder}{file}")
            domain = file.split("_")[0]
            df["domain"] = domain
            if "Llama" in file:
                # preprocess llama files
                texts = list()
                for text in df["text"].values:
                    start_index = text.find("\n\n") + 2
                    text = text[start_index:]
                    if text.find("Example:") > -1:
                        text = text.strip("Example:").strip()
                    texts.append(text)
                df["text"] = texts
            if "gemma" in file:
                subset = list()
                for i in range(len(df["text"].values)):
                    text = df["text"].values[i]
                    if "I" in text and "generate" in text and "can" in text:
                        continue
                    else:
                        if ":\n\n" in text:
                            text = text.split(":\n\n")[1].strip()
                        subset.append(
                            {
                                "prompt": df["prompt"].values[i],
                                "text": text,
                                "label": 1,
                                "domain": domain,
                                "model":df.model.values[i]
                            }
                        )
                df = pd.DataFrame(subset)
                df = df.iloc[0:200]

                        
                        
            frames.append(df)
    pd.concat(frames).to_csv("../data/test.csv",index=False)
