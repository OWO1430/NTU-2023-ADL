import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import argparse
import json
from utils import get_prompt, get_bnb_config
import bitsandbytes as bnb
# import accelerator

# def get_last_checkpoint(checkpoint_dir):
#     if isdir(checkpoint_dir):
#         is_completed = exists(join(checkpoint_dir, 'completed'))
#         if is_completed: return None, True # already finished
#         max_step = 0
#         for filename in os.listdir(checkpoint_dir):
#             if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
#                 max_step = max(max_step, int(filename.replace('checkpoint-', '')))
#         if max_step == 0: return None, is_completed # training started, but no checkpoint
#         checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
#         print(f"Found a previous checkpoint at: {checkpoint_dir}")
#         return checkpoint_dir, is_completed # checkpoint found!
#     return None, False # first training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="",
        required=True,
        help="User question."
    )



    args = parser.parse_args()

    # args_string = '''
    #                 --base_model_path /content/drive/MyDrive/ADL/HW3/Taiwan-LLM-7B-v2.0-chat
    #                 --peft_path /content/drive/MyDrive/ADL/HW3/model_4000_data/checkpoint-250
    #                 --test_data_path /content/drive/MyDrive/ADL/HW3/data/train_test.json
    #                 --output_data_path /content/drive/MyDrive/ADL/HW3/test/zero_shot.json
    #               '''

    # args_string = '''
    #                 --base_model_path /content/drive/MyDrive/ADL/HW3/Taiwan-LLM-7B-v2.0-chat
    #                 --test_data_path /content/drive/MyDrive/ADL/HW3/data/train.json
    #                 --output_data_path /content/drive/MyDrive/ADL/HW3/test/zero_shot.json
    #               '''

    # args = parser.parse_args(args_string.split())

    # TODO: Update variables
    max_new_tokens = 1024
    top_p = 0.5
    temperature=0.7
    bnb_config = get_bnb_config()

    # Base model
    model_name_or_path = args.base_model_path
    # Adapter name on HF hub or local checkpoint path.
    # adapter_path, _ = get_last_checkpoint('qlora/output/guanaco-7b')
    adapter_path = args.peft_path
    test_data_path = args.test_data_path

    with open(test_data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    print (data)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=bnb_config
    )

    if (adapter_path is not None):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # prompt = (
    #     "A chat between a curious human and an artificial intelligence assistant. "
    #     "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    #     "### Human: {user_question}"
    #     "### Assistant: "
    # )

    # prompt = get_prompt()

    def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
        # inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
        inputs = tokenizer(user_question, return_tensors="pt").to('cuda')

        outputs = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        trimmed_text = full_text.replace(user_question, "", 1).strip()  # Removes the first occurrence of the prompt
        print(trimmed_text)
        return trimmed_text
        # return text

    output = []

    for i in range(len(data)):
        user_question = get_prompt(data[i]['instruction'])
        text = generate(model, user_question)
        output.append({
            'id': data[i]['id'],
            'output': text
        })


    print (output)

    with open(args.output_data_path, 'w') as outfile:
        json.dump(output, outfile)
