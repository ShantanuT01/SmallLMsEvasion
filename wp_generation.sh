# Zero Shot No Fine Tuning
#python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Write a new fictional story based on the following writing prompt. Prompt:" --system_prompts="filtered_prompts.csv" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot.json"
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp" --system_prompt="Write a new fictional story based on the following writing prompt. Prompt:"  --system_prompts="filtered_prompts.csv" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot.json" 




# Continuation no fine-tuning
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_base_model_continuation.json"
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_abliterated_model_continuation.json"



# 1 shot no finetuning
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp" --k_shot=1 --system_prompt="Generate a new story that mimics the style of the following story." --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_base_model_1_shot.json"
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp" --k_shot=1 --system_prompt="Generate a new story that mimics the style of the following story." --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_abliterated_model_1_shot.json"

############################################
# Zero Shot Finetuning
 
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Generate a 1-star wp review that could have been written by a human." --dataset_size=40  --peft_path="models/fine_tuned_llama_3-2_1b_wp" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot-1-star_finetuned.json" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Generate a 2-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_wp" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot-2-star_finetuned.json" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Generate a 3-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_wp" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot-3-star_finetuned.json" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Generate a 4-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_wp" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot-4-star_finetuned.json" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --system_prompt="Generate a 5-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_wp" --max_new_tokens=256 --output_path="release/wp_base_model_zero_shot-5-star_finetuned.json" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"

python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --system_prompt="Generate a 1-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot-1-star_finetuned.json" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp"
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --system_prompt="Generate a 2-star wp review that could have been written by a human." --dataset_size=40  --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot-2-star_finetuned.json" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp"
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --system_prompt="Generate a 3-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot-3-star_finetuned.json" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp";
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --system_prompt="Generate a 4-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot-4-star_finetuned.json" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp";
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --system_prompt="Generate a 5-star wp review that could have been written by a human." --dataset_size=40 --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --max_new_tokens=256 --output_path="release/wp_abliterated_model_zero_shot-5-star_finetuned.json" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp";



# Continuation Finetuning
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_base_model_continuation_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_wp" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp";
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp"  --continuation=True --dataset_size=200 --max_new_tokens=256 --output_path="release/wp_abliterated_model_continuation_finetuned.json" --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp";

# 1 shot finetuning
python generate_data.py --model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated" --domain="wp" --k_shot=1 --system_prompt="Generate a new review that mimics the style of the following review(s)." --dataset_size=200 --peft_path="models/fine_tuned_llama_3-2_1b_abliterated_wp" --max_new_tokens=256 --output_path="release/wp_abliterated_model_1_shot_finetuned.json" --new_model_name="huihui-ai/Llama-3.2-1B-Instruct-abliterated-finetuned-wp"
python generate_data.py --model_name="meta-llama/Llama-3.2-1B-Instruct" --domain="wp" --k_shot=1 --system_prompt="Generate a new review that mimics the style of the following review(s)." --dataset_size=200 --peft_path="models/fine_tuned_llama_3-2_1b_wp" --max_new_tokens=256 --output_path="release/wp_base_model_1_shot_finetuned.json" --new_model_name="meta-llama/Llama-3.2-1B-Instruct-finetuned-wp"