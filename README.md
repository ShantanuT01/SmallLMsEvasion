# Can Small Language Models Evade Detection?
### Shantanu Thorat

This repository contains project code for University of Cambridge's R255 Advanced Topics in Machine-Learning Lent Term 2025. 

## About

Existing literature (as of February 2025) tends to focus on AI text detection from large language models (LLMs). However, the emergence of small language models (SLMs) has not led to investigative efforts. However, SLMs offer many advantages compared to LLMs:

- Faster runtime
- Works on limited GPU
- Less parameters often equates to faster fine-tuning

## Code Structure
### Modules

`llm_attacker` is a helper library to create attacked texts on a detector. 

`tinylm` is a helper library to generate texts from small language models. 

## Finetuned Models and Datasets

We provide our testing and finetuning datasets, as well as our models at the following [link](https://huggingface.co/collections/ShantanuT01/r255-67e2f764cf1146a8fbd7f0c8). 

We fine-tune all SLMs using the following hyperparameters:
```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=
        ["q_proj", "v_proj","k_proj", "o_proj","gate_proj","up_proj","down_proj"],#"gate_proj"],
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

