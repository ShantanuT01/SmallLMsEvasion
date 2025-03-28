import numpy as np
import shap
import polars as pl
from llm_attacker.constants import *
from transformers import pipeline
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

class SynonymAttacker:
    def __init__(self, classifier_pipeline, mlm_model,space_char="Ġ",ai_label="ai"):
        self.__mlm_model = mlm_model
        self.__explainer = shap.Explainer(classifier_pipeline)
        self.__tokenizer = classifier_pipeline.tokenizer
        self.__space_char = space_char
        self.__pipeline = classifier_pipeline
        self.__ai_label = ai_label
    
    def get_shapley_values(self, text):
        shap_values = self.__explainer([text])
        scores = shap_values[0].values[:, 0]
        tokens = self.__tokenizer.tokenize(text)
        values = list(zip(tokens, scores[1:-1]))
        for i in range(len(values)):
            values[i] = [values[i][0], float(values[i][1])]
        return np.array(values,dtype=object)
    
    def attack_greedily(self, tokens_and_scores, max_edits=10, cutoff_score=0.5,log=True):
        index_order = tokens_and_scores[:,1].argsort()[::-1][0:min(max_edits, len(tokens_and_scores))]
        #print(index_order)
        #[0:min(max_edits, len(tokens_and_scores))]
        old_string = self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:,0])
        initial_baseline = self.__pipeline(old_string)
        min_score = pl.DataFrame(initial_baseline[0]).filter(pl.col(LABEL) == self.__ai_label).get_column(SCORE).to_list()[0]
        if min_score < cutoff_score:
            return old_string
        if log:
            print(self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0]), min_score)
        for index in index_order:

            old_string = self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:,0])

            token = tokens_and_scores[index, 0]
            shapley_val = tokens_and_scores[index, 1]
        
            if shapley_val < 0:
                break
            space_index = token.find(self.__space_char)
            rest_of_string = token[space_index + 1:]
            tokens_and_scores[index,0] = token.replace(rest_of_string, "[MASK]")
            
            mask_string = self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:,0])
            outputs = self.__mlm_model(mask_string)
            possibilities = pl.DataFrame(outputs)
            
            possibilities = possibilities.filter(pl.col(SEQUENCE) != old_string)
            scores = self.__pipeline(possibilities.get_column(SEQUENCE).to_list())
            best_token = ""
            for i in range(len(scores)):
                new_score = pl.DataFrame(scores[i]).filter(pl.col(LABEL) == self.__ai_label).get_column(SCORE).to_list()[0]
                if new_score < min_score:
                    best_token = possibilities.get_column("token_str").to_list()[i]
                    min_score = new_score
           # print("Best:", best_token,  min_score)
            if best_token == "":
                tokens_and_scores[index, 0] = token
            else:
                tokens_and_scores[index, 0] = best_token
            if log:
                print(self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0]), min_score)
            if min_score < cutoff_score:
                break
   
        return self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0])


    def attack_text(self, text,max_edits=20,log=True):
        tokens_and_scores = self.get_shapley_values(text)
        
        return self.attack_greedily(tokens_and_scores, max_edits=max_edits,log=log)


def paraphrase_texts(texts,max_new_tokens, n_sentences):
    edit_pipe = pipeline("text2text-generation", model="grammarly/coedit-large")
    all_sentences = [sent_tokenize(text) for text in texts]
    para_sent_ids = list()
    for id, text in enumerate(all_sentences):
        paraphrase_sentences = np.random.choice(np.arange(len(text),dtype=int),size=min(n_sentences, len(text)),replace=False)
        para_sent_ids.extend([(id, para_id) for para_id in paraphrase_sentences])
    def data():
        for text_id, sentence_id in para_sent_ids:
            yield f"Paraphrase this: {all_sentences[text_id][sentence_id]}"
    paraphrased_texts = list()
    for result in tqdm(edit_pipe(data(), max_new_tokens=max_new_tokens, batch_size=4),total=len(para_sent_ids)):
        paraphrased_texts.append(result[0]["generated_text"])
    for i in range(len(paraphrased_texts)):
        text_id, sentence_id = para_sent_ids[i]
        all_sentences[text_id][sentence_id] = paraphrased_texts[i].strip()
    final_para = list()
    for text in all_sentences:
        final_para.append(" ".join(text).strip())
    return final_para


def paraphrase_text(edit_pipe, text,max_new_tokens):
    
    result = edit_pipe(f"Paraphrase this: {text}", max_new_tokens=max_new_tokens)
    return result[0]["generated_text"].strip()
    
