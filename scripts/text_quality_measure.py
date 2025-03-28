import textdescriptives as td
import spacy
import pandas as pd
from tqdm import tqdm
df = pd.read_csv("release/synonym_substitution_test.csv")
df = df[df.label == 1]
print(df.groupby("domain")["attack"].value_counts())
raise Exception()
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("textdescriptives/quality")
quality_checks = list()
for text in tqdm(df["text"].values):
    doc = nlp(text)
    quality_checks.append(doc._.passed_quality_check)
df["pass_quality_check"] = quality_checks
df[["attack","text","pass_quality_check"]].to_csv("quality_checks.csv",index=False)
print(df.groupby("attack")["pass_quality_check"].value_counts())
