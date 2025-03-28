from llm_attacker.attacker import SynonymAttacker, paraphrase_text
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize
from collections import defaultdict

if __name__ == "__main__":
    
    for domain in ["yelp","wp","sci_gen"]:
        df = pd.read_csv("benchmarks/e5_lora_results.csv")
        df = df[(df["label"] == 1) & (df.domain == domain)]
        human_df = df[(df["label"] == 0) & (df.domain == domain)]
        df = df.sort_values("e5_lora_pred",ascending=False)
        df = df.head(100)

        classifier = pipeline("text-classification", model="MayZhou/e5-small-lora-ai-generated-detector",top_k=None)
        mlm = pipeline(
            "fill-mask",
            model="answerdotai/ModernBERT-base",
            torch_dtype=torch.bfloat16,
        )
        edit_pipe = pipeline("text2text-generation", model="grammarly/coedit-large")
        attacker = SynonymAttacker(classifier, mlm, ai_label="LABEL_1")
        rows = list()

        for i, text in enumerate(tqdm(df["text"].values)):
            sentences = sent_tokenize(text)
            filtered_sentences = list()
            for sentence in sentences:
                if len(sentence) >= 50:
                    filtered_sentences.append(sentence)

            sentences = filtered_sentences
            targets = np.random.choice(np.arange(len(sentences),dtype=int),size=min(3, len(sentences)),replace=False)
            paraphrased_only = list(sentences)
            synonym_only = list(sentences)
            synonym_and_paraphrase = list(sentences)
            
            for target in targets:
                row_template = {"source_text": sentences[target], "text": None,"label": 1, "attack":"no_attack", "model": df["model"].values[i],"prompt-strategy":df["prompt-strategy"].values[i],"domain": domain}
                base_row = dict(row_template)
                base_row["text"] = sentences[target] 
                rows.append(base_row)
                para_row = dict(row_template)
                para_row["attack"] = "para"
                para_row["text"] = paraphrase_text(edit_pipe, sentences[target], 50)
                rows.append(para_row)

                syn_row = dict(row_template)
                syn_row["attack"] = "syn"
                syn_row["text"] = attacker.attack_text(sentences[target],max_edits=5,log=False)
                rows.append(syn_row)

                para_syn_row = dict(row_template)
                para_syn_row["attack"] = "syn+para"
                para_syn_row["text"] = paraphrase_text(edit_pipe, syn_row["text"], 50)
                rows.append(para_syn_row)

              
            #new_text = paraphrase_texts([new_text],max_new_tokens=50, n_sentences=3)[0]
        
        sf = pd.concat([pd.DataFrame(rows), human_df])
        sf.to_csv(f"release/synonym_substitution_{domain}.csv",index=False)

    