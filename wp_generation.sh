# Zero Shot No Fine Tuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Write a new fictional story based on the following writing prompt. Prompt:" --system_prompts="filtered_prompts.csv" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot.json"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp" --system_prompt="Write a new fictional story based on the following writing prompt. Prompt:"  --system_prompts="filtered_prompts.csv" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot.json" 




# Continuation no fine-tuning
# python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_base_model_continuation.json"
# python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --system_prompts="release/wp_base_model_continuation.json" --output_path="release/wp_abliterated_model_continuation.json"



# 1 shot no finetuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp" --k_shot=1 --system_prompt="Generate a new story that mimics the style of the following story." --system_prompts="release/wp_base_model_continuation.json" --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_base_model_1_shot.json"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp" --k_shot=1 --system_prompt="Generate a new story that mimics the style of the following story." --dataset_size=200 --max_new_tokens=256 --system_prompts="release/wp_base_model_continuation.json" --output_path="release/wp_abliterated_model_1_shot.json"

############################################
# Zero Shot Finetuning

#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --system_prompt="Write a new fictional story based on the following writing prompt. Prompt:" --system_prompts="filtered_prompts.csv" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp"
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Write a new fictional story based on the following writing prompt. Prompt:" --system_prompts="filtered_prompts.csv" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_wp" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"

# Continuation fine-tuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --system_prompts="release/wp_base_model_continuation.json" --output_path="release/wp_base_model_continuation_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_wp" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --system_prompts="release/wp_base_model_continuation.json" --output_path="release/wp_abliterated_model_continuation_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp"


# 1 shot  finetuning

#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp" --k_shot=1 --system_prompt="Generate a new story that mimics the style of the following story." --system_prompts="release/wp_base_model_continuation.json" --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_base_model_1_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_wp" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"
#python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp" --k_shot=1 --system_prompt="Generate a new story that mimics the style of the following story." --dataset_size=200 --max_new_tokens=256 --system_prompts="release/wp_base_model_continuation.json" --output_path="release/wp_abliterated_model_1_shot_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp"
