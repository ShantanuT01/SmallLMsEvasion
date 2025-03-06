import pandas as pd
from transformers import pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

if __name__ == "__main__":
    classifier = pipeline("text-classification","yaful/MAGE",return_all_scores=True)

    df = pd.read_csv("../data/test.csv")
    scores = list()
    for text in tqdm(df["text"].values):
        score = classifier(text)
        score = score[0]
        # MAGE classifier has label 0 for AI-generated texts
        for entry in score:
            if entry["label"] == 0:
                scores.append(entry["score"])
    results = pd.DataFrame()
    results["text"] = df["text"].to_list()
    results["model"] = df["model"].to_list()
    results["model"] = results["model"].fillna("human")
    results["domain"] = df["domain"].to_list()
    results["mage_pred"] = scores
    results["label"] = df["label"].to_list()
    results.to_csv("mage_results.csv",index=False)
    print("AUC Score:", roc_auc_score(y_score=scores, y_true=df["label"].to_list()))
    print("AP Score:", average_precision_score(y_score=scores, y_true=df["label"].to_list()))
    for model in results["model"].unique():
        if model == "human":
            continue
        else:
            sf = results[results["model"].isin({"human",model})]
            print(f"{model} AUC Score:", roc_auc_score(y_score=sf["mage_pred"], y_true=sf["label"]))
            print(f"{model} AP Score:", average_precision_score(y_score=sf["mage_pred"], y_true=sf["label"]))