# Zero shot no finetuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="sci_gen"  --system_prompt="Write a new abstract for a paper with the following title. Title:" --system_prompts="filtered_abstracts.csv" --max_new_tokens=256 --output_path="release/sci_gen_base_model_zero_shot.json";
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="sci_gen" --system_prompt="Write a new abstract for a paper with the following title. Title:"  --system_prompts="filtered_abstracts.csv" --max_new_tokens=256 --output_path="release/sci_gen_abliterated_model_zero_shot.json"

# Continuation no fine-tuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="sci_gen"  --continuation=True --dataset_size=200 --max_new_tokens=256 --output_path="release/sci_gen_base_model_continuation.json"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="sci_gen"  --continuation=True --dataset_size=200 --max_new_tokens=256 --system_prompts="release/sci_gen_base_model_continuation.json" --output_path="release/sci_gen_abliterated_model_continuation.json"

# 1 shot no finetuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="sci_gen" --k_shot=1 --system_prompt="Generate a new abstract that mimics the style of the following abstract." --system_prompts="release/sci_gen_base_model_continuation.json" --dataset_size=200 --max_new_tokens=256 --output_path="release/sci_gen_base_model_1_shot.json"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="sci_gen" --k_shot=1 --system_prompt="Generate a new abstract that mimics the style of the following abstract." --dataset_size=200 --max_new_tokens=256 --system_prompts="release/sci_gen_base_model_continuation.json" --output_path="release/sci_gen_abliterated_model_1_shot.json"


############################################
# Zero Shot Finetuning

#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="sci_gen"  --system_prompt="Write a new abstract for a paper with the following title. Title:" --system_prompts="filtered_abstracts.csv" --max_new_tokens=256 --output_path="release/sci_gen_abliterated_model_zero_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_sci_gen" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-sci_gen"
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="sci_gen"  --system_prompt="Write a new abstract for a paper with the following title. Title:" --system_prompts="filtered_abstracts.csv" --max_new_tokens=256 --output_path="release/sci_gen_base_model_zero_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_sci_gen" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-sci_gen"

# Continuation fine-tuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="sci_gen"  --continuation=True --dataset_size=200 --max_new_tokens=256 --system_prompts="release/sci_gen_base_model_continuation.json" --output_path="release/sci_gen_base_model_continuation_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_sci_gen" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-sci_gen"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="sci_gen"  --continuation=True --dataset_size=200 --max_new_tokens=256 --system_prompts="release/sci_gen_base_model_continuation.json" --output_path="release/sci_gen_abliterated_model_continuation_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_sci_gen" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-sci_gen"


# 1 shot  finetuning

#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="sci_gen" --k_shot=1 --system_prompt="Generate a new abstract that mimics the style of the following abstract." --system_prompts="release/sci_gen_base_model_continuation.json" --dataset_size=200 --max_new_tokens=256 --output_path="release/sci_gen_base_model_1_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_sci_gen" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-sci_gen"
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="sci_gen" --k_shot=1 --system_prompt="Generate a new abstract that mimics the style of the following abstract." --dataset_size=200 --max_new_tokens=256 --system_prompts="release/sci_gen_base_model_continuation.json" --output_path="release/sci_gen_abliterated_model_1_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_sci_gen" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-sci_gen"
