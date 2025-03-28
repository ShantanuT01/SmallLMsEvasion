import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets
from tinylm.prompt import prompt_slm_k_shot, prompt_slm_zero_shot, prompt_slm_continuation
import dotenv
import os
from peft import PeftModel
import argparse

dotenv.load_dotenv()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--peft_path",type=str,default=None)
    parser.add_argument("--domain",type=str)
    parser.add_argument("--k_shot",type=int, default=0)
    parser.add_argument("--system_prompt",type=str)
    parser.add_argument("--dataset_size",type=int, default=200)
    parser.add_argument("--max_new_tokens",type=int, default=128)
    parser.add_argument("--output_path",type=str)
    parser.add_argument("--new_model_name",type=str,default=None)
    parser.add_argument("--continuation",type=bool,default=False)
    parser.add_argument("--system_prompts",type=str,default=None)
    parser.add_argument("--local_dataset",type=str,default=None)
    parser.add_argument("--trim_continuation",type=bool,default=True)
    args = parser.parse_args()

    # get selected domain from MAGE
    test_set = datasets.load_dataset("yaful/MAGE",split="test")
    test_set = test_set.to_pandas()
    domain = args.domain
    domain_test_set = test_set[(test_set["src"].str.contains(domain))]
    # 1 = human written text
    domain_test_set = domain_test_set[domain_test_set["label"] == 1]
    examples = domain_test_set["text"].sample(n=args.dataset_size).to_list()
    model_name = args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.peft_path is not None:
        print("Using a PEFT model.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        #model.to("cuda")
        model = PeftModel.from_pretrained(model, args.peft_path)
        model.to("cuda")
        model = model.merge_and_unload()
        if args.continuation:
            tokenizer = AutoTokenizer.from_pretrained(model_name,tokenizer=os.environ["HF_TOKEN"],padding_side='left')
            pipe = pipeline("text-generation", model=model,tokenizer=tokenizer, token=os.environ["HF_TOKEN"],device="cuda")
            pipe.model.generation_config.pad_token_id = tokenizer.eos_token_id
        else:
            pipe = pipeline("text-generation", model=model,tokenizer=tokenizer, token=os.environ["HF_TOKEN"],device="cuda")
            #pipe.model = model
            pipe.model.generation_config.pad_token_id = tokenizer.eos_token_id
    elif args.continuation:
         tokenizer = AutoTokenizer.from_pretrained(model_name,tokenizer=os.environ["HF_TOKEN"],padding_side='left')
         pipe = pipeline("text-generation", model=args.model_name,tokenizer=tokenizer, token=os.environ["HF_TOKEN"],device="cuda")
         pipe.model.generation_config.pad_token_id = tokenizer.eos_token_id
    else:
        pipe = pipeline("text-generation", model=model_name,token=os.environ["HF_TOKEN"],device="cuda")
        pipe.model.generation_config.pad_token_id = tokenizer.eos_token_id
    if not args.continuation:
        if args.k_shot > 0:
            
            examples_2d = list()
            for i in range(0, args.dataset_size,args.k_shot):
                examples_2d.append(examples[i:i+args.k_shot])
            if args.system_prompts is not None:
                input_frame = pd.read_json(args.system_prompts)
                examples_2d = input_frame["examples"]
            output_dataframe = prompt_slm_k_shot(pipe, args.system_prompt, examples_2d, args.max_new_tokens)
            output_dataframe["shot"] = args.k_shot
            output_dataframe["examples"] = examples_2d
            output_dataframe["peft"] = (args.peft_path is not None)

            
        else:
            system_prompts = [args.system_prompt for _ in range(args.dataset_size)]
            if args.system_prompts is not None:
                if args.system_prompts.endswith(".csv"):
                    system_prompts = pd.read_csv(args.system_prompts)["prompt"].to_list()
                else:
                    system_prompts = pd.read_json(args.system_prompts)["prompt"].to_list()
                system_prompts = [f"{args.system_prompt} {system_prompt}" for system_prompt in system_prompts]
            output_dataframe = prompt_slm_zero_shot(pipe, system_prompts, args.max_new_tokens)
            output_dataframe["shot"] = args.k_shot
            output_dataframe["examples"] = None
            output_dataframe["peft"] = (args.peft_path is not None)
        
        
    else:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        if args.system_prompts is not None:
            if args.system_prompts.endswith(".csv"):
                examples = pd.read_csv(args.system_prompts)["examples"].to_list()
            else:
                examples = pd.read_json(args.system_prompts)["examples"].to_list()
            flatten_examples = list()
            for list_of_examples in examples:
                for example in list_of_examples:
                    flatten_examples.append(example)
            examples = flatten_examples

        output_dataframe = prompt_slm_continuation(pipe, examples, args.max_new_tokens,batch_size=4)
        output_dataframe["shot"] = args.k_shot
        output_dataframe["examples"] = examples
        output_dataframe["peft"] = (args.peft_path is not None)
        output_dataframe["continuation"] = True

    if args.new_model_name is not None:
        output_dataframe["model"] = args.new_model_name
    output_dataframe["domain"] = args.domain
   # output_dataframe.to_json(args.output_path,index=False,orient="records",indent=4)
   # prompts = pd.read_csv("data/yelp_finetuned-llama-3.2-1B-continuation.csv")
    #pd.DataFrame(rows).to_csv("data/yelp_llama-3.2-1B-continuation.csv",index=False)
    #prompt = "Generate a new review that mimics the style of the following review(s). Output only the text of the review."
    #out = prompt_slm_k_shot(pipe, prompt,examples, 128)
    #out["model"] = "finetuned Llama-3.2-1B Instruct Yelp - 1 Shot"
    #out.to_csv("data/yelp_finetuned-Llama-3.2-1B_generated_1_shot.csv",index=False)
