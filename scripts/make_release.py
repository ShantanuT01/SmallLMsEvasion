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
            frames.append(df)
    pd.concat(frames).to_csv("../data/test.csv",index=False)
