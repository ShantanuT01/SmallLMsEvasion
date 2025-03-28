from llm_attacker.attacker import paraphrase_texts
import pandas as pd
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    df = pd.read_csv("release/prompting_test.csv")
    human_df = df[df["label"] == 0]
   
    domains = ["wp","yelp","sci_gen"]
    domain_groups = df.groupby("domain")
    para_dfs = list()
    
    for domain in domains:
        domain_df = domain_groups.get_group(domain)
        source_texts = domain_df["text"].to_list()
        ids = domain_df["id"].to_list()
        prompt_strategies = domain_df["prompt-strategy"].to_list()
        paraphrased_texts = paraphrase_texts(source_texts, max_new_tokens=50,n_sentences=3)
        paraphrased_domain_attack = pd.DataFrame()
        paraphrased_domain_attack["source_id"] = ids
        paraphrased_domain_attack["source_text"] = source_texts
        paraphrased_domain_attack["text"] = paraphrased_texts
        paraphrased_domain_attack["attack"] = "paraphrased"
        paraphrased_domain_attack["prompt-strategy"] = prompt_strategies
        paraphrased_domain_attack["model"] = domain_df["model"].to_list()
        paraphrased_domain_attack["label"] = 1
        paraphrased_domain_attack["domain"] = domain
        para_dfs.append(paraphrased_domain_attack)
    para_df = pd.concat(para_dfs).sort_values("source_id")
    para_df = pd.concat([para_df, df[(df.domain == "sci_gen") & (df.label == 0)] ])
    para_df.to_csv("release/paraphrasing_test_wp.csv",index=False)
    
        


