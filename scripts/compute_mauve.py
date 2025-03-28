from evaluate import load
import pandas as pd 

mauve = load('mauve')
files = [
    "release/yelp_base_model_continuation.json",
    "release/sci_gen_base_model_continuation.json",
    "release/wp_base_model_continuation.json"
]
predictions_all = list()
references_all = list()
for file in files:
    df = pd.read_json(file)
    trimmed_strings = ["Here's a new review","Here's a new abstract", "Here is a new review","Here is a new abstract","Here is a potential abstract" ]
    predictions = df["text"].to_list()
    prompts = df["prompt"].to_list()
    references = [examples[0] for examples in df["examples"].to_list()]
    #print(references[0])
    #print("-")
    for i in range(len(prompts)):
       # print(predictions[i])
        #print('-')
        predictions[i] = predictions[i].strip(prompts[i])
       
       # references[i] = references[i].strip(prompts[i])

 
    ''' 
    for i in range(len(predictions)):
        for string in trimmed_strings:
            if string in predictions[i]:
                text = "".join(predictions[i].split(":")[1:])
                text = text.strip()
                predictions[i] = text
                print("Modified")
                break
    '''
    references_all.extend(references)
    predictions_all.extend(predictions)
mauve_results = mauve.compute(predictions=predictions_all, references=references_all,device_id=0, mauve_scaling_factor=2,seed=2025)
print(mauve_results.mauve)
