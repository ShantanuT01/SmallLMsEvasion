import pandas as pd
import os
import datasets

if __name__ == "__main__":
    test_set = datasets.load_dataset("yaful/MAGE",split="test")
    test_set = test_set.to_pandas()
    files = os.listdir("release/")
    dataframes = list()
    for file in files:
        if file.endswith(".json"):
            dataframes.append(pd.read_json(f"release/{file}"))
    test_frame = pd.concat(dataframes)
    domains = test_frame["domain"].unique()
    for domain in domains:
        domain_test_set = test_set[(test_set["src"].str.contains(domain))]
        domain_test_set = domain_test_set[domain_test_set["label"] == 1]
        domain_test_set["label"] = 0
        domain_test_set["domain"] = domain
        domain_test_set["model"] = "human"
        dataframes.append(domain_test_set[["text","label","domain","model"]])
    pd.concat(dataframes).to_csv("release/test.csv",index=False)
    