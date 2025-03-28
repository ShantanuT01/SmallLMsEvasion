import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

def predict_texts(texts, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer.batch_encode_plus(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    predictions = list()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probabilities = torch.sigmoid(logits).cpu().detach().numpy()
        predictions.append(probabilities)
    #label = 1 if probability >= threshold else 0
    return np.concatenate(predictions)


if __name__ == "__main__":
    # use code from README.md
    
    model_directory = "desklib/ai-text-detector-v1.01"

    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = DesklibAIDetectionModel.from_pretrained(model_directory)
    model.to("cuda")
    
    

    df = pd.read_csv("../release/prompting_test.csv")
    scores = list()
    batch_size = 16
    texts = df["text"].to_list()
    for i in tqdm(range(0, len(texts), batch_size)):
        probs = predict_texts(texts[i:i+batch_size], model, tokenizer,device="cuda")
        scores.extend(probs.flatten())
    results = pd.DataFrame()
    results["text"] = texts
    results["model"] = df["model"].to_list()
    results["model"] = results["model"].fillna("human")
    results["domain"] = df["domain"].to_list()
    results["prompt-strategy"] = df["prompt-strategy"].to_list()
    results["desk_lib_pred"] = scores
    results["label"] = df["label"].to_list()
    results.to_csv("desklib_results.csv",index=False)
    print("AUC Score:", roc_auc_score(y_score=scores, y_true=df["label"].to_list()))
    print("AP Score:", average_precision_score(y_score=scores, y_true=df["label"].to_list()))
    for model in results["prompt-strategy"].unique():
        if model == "human":
            continue
        else:
            sf = results[results["prompt-strategy"].isin({"human",model})]
            print(f"{model} AUC Score:", roc_auc_score(y_score=sf["desk_lib_pred"], y_true=sf["label"]))
            print(f"{model} AP Score:", average_precision_score(y_score=sf["desk_lib_pred"], y_true=sf["label"]))