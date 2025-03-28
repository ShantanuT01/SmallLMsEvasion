import pandas as pd
from transformers import pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

if __name__ == "__main__":
    classifier = pipeline("text-classification","yaful/MAGE",return_all_scores=True)

    #df = pd.concat([pd.read_json("../release/yelp_abliterated_model_1_shot_finetuned.json"), pd.read_json("../release/yelp_base_model_1_shot.json"),pd.read_json("../release/yelp_abliterated_model_1_shot.json"), pd.read_csv("../data/yelp_human.csv")])
    df = pd.read_csv("release/synonym_substitution_test.csv")
    scores = list()
    def data():
        for text in df["text"].values:
            yield text
    '''
    for score in tqdm(classifier(data(),batch_size=8), total=len(df)):
        # MAGE classifier has label 0 for AI-generated texts
        for entry in score:
            if entry["label"] == 0:
                scores.append(entry["score"])
   # results = pd.read_csv("benchmarks/mage_results.csv")
    
    results = pd.DataFrame()
    results["text"] = df["text"].to_list()
    results["model"] = df["model"].to_list()
    results["model"] = df["model"].fillna("human")
    results["domain"] = df["domain"].to_list()
    results["mage_pred"] = scores
    results["label"] = df["label"].to_list()
    results["prompt-strategy"] = df["prompt-strategy"].to_list()
    results["attack"] = df["attack"].to_list()
    '''
    #print(results["mage_pred"].describe())
    #results.to_csv("benchmarks/mage_synonym_substitution_results.csv",index=False)
    results = pd.read_csv("benchmarks/mage_synonym_substitution_results.csv")
    results["attack"] = results["attack"].fillna("no_attack_human")
    results.to_csv("benchmarks/mage_synonym_substitution_results.csv",index=False)
    print("AUC Score:", roc_auc_score(y_score=results["mage_pred"], y_true=results["label"].to_list()))
    print("AP Score:", average_precision_score(y_score=results["mage_pred"], y_true=results["label"].to_list()))
    
    for prompt_strategy in results["attack"].unique():
        if prompt_strategy == "no_attack_human":
            continue
        else:
            sf = results[results["attack"].isin({"no_attack_human",prompt_strategy})]
            print(f"{prompt_strategy} AUC Score:", roc_auc_score(y_score=sf["mage_pred"], y_true=sf["label"]))
            print(f"{prompt_strategy} AP Score:", average_precision_score(y_score=sf["mage_pred"], y_true=sf["label"]))
    