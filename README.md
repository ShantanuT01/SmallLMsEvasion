# Can Small Language Models Evade Detection?
### Shantanu Thorat

This repository contains project code for University of Cambridge's R255 Advanced Topics in Machine-Learning Lent Term 2025. 

## About

Existing literature (as of February 2025) tends to focus on AI text detection from large language models (LLMs). However, the emergence of small language models (SLMs) has not led to investigative efforts. However, SLMs offer many advantages compared to LLMs:

- Faster runtime
- Works on limited GPU
- Less parameters often equates to faster fine-tuning

In this project, we investigated how well existing detectors perform on Llama-3.2-1B-Instruct generated texts. 
## Code Structure
### Modules

`llm_attacker` is a helper library to create attacked synonym-substituted texts on a detector.  

`tinylm` is a helper library to generate texts from small language models. 

`benchmarks` folder contains example code to test three classifiers. 

## Finetuned Models and Datasets

We provide our testing and finetuning datasets, as well as our models at the following [link](https://huggingface.co/collections/ShantanuT01/r255-67e2f764cf1146a8fbd7f0c8). 

We fine-tune all SLMs using the following hyperparameters:
```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=
        ["q_proj", "v_proj","k_proj", "o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.001,
    save_total_limit=2,
    warmup_steps=45,
    save_strategy="epoch",
    gradient_accumulation_steps=4,
    fp16=True
)
```

You can use the fine-tuned models from HuggingFace as follows:
```python
from transformers import pipeline

model_id = "ShantanuT01/Llama-3.2-1B-Instruct-wp-finetuned" # replace with model you want!
pipe = pipeline("text-generation", model_id)

# for continuation
outputs = pipe("Hello, World!")

# or you can pass in system messages
messages = [  
    {"role": "user", "content": "Hello, World!"}
]
output_from_messages = pipe(messages)
```

For datasets, we recommend downloading the individual CSV files from HuggingFace. 

- [Testing Data](https://huggingface.co/datasets/ShantanuT01/R255-Test-Generations/tree/main)
- [Finetuning Data](https://huggingface.co/datasets/ShantanuT01/R255-Finetuning-Datasets/tree/main)