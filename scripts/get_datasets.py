import pandas as pd
from datasets import load_dataset

if __name__ == "__main__":
    test_set = load_dataset("yaful/MAGE",split="test")
    test_set = test_set.to_pandas()
    domains = ["sci_gen"]
    for domain in domains:
        domain_test_set = test_set[(test_set["src"].str.contains(domain))]
        domain_test_set = domain_test_set[domain_test_set["label"] == 1]
        domain_test_set["label"] = 0
        domain_test_set.to_csv(f"../data/human_{domain}.csv",index=False)
