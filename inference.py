import time
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import logging
import jsonlines
from rouge import Rouge

logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Inference LlaMA')
parser.add_argument('--max_new_tokens', type=int, default=64, required=True, help='No. of max new tokens')
parser.add_argument('--tokenizer_max_length', type=int, default=1024, required=False, help='No. of max new tokens')
parser.add_argument('--target_model', default="", help='Target model (HF Causal LM model)')
parser.add_argument('--bsz', default=32, type=int, help='batch size')
parser.add_argument('--target_task', default="gsm8k", help='task name')
parser.add_argument('--output_path', default="", required=False, help='output path')
parser.add_argument('--test_num', default=None, type=int, required=False, help='test data number')
parser.add_argument('--split_part', default=None, type=str, required=False, help='split part, for example: 1-5')
parser.add_argument('--temperature', default=0, type=float, help='Temperature')
args = parser.parse_args()

task2path={
    "dolly": "data/dolly/valid.jsonl",
    "self-inst": "data/self-inst/valid.jsonl",
    "vicuna": "data/vicuna/valid.jsonl",
    "koala": "data/koala/valid.jsonl",
    "wizardlm": "data/wizardlm/valid.jsonl"
}

def load_data(data_args):
    path=task2path[data_args.target_task]
    data_json=[]
    with open(path, "r", encoding="utf-8") as f:
        reader=jsonlines.Reader(f)
        for item in reader.iter(type=dict, skip_invalid=True):
            if "input" in item.keys() and item["input"] != "" and item["input"] is not None:
                data_json.append({"input": f'Human:\n### Instruction:\n{item["instruction"]}\n\n### Input:\n{item["input"]}\n\nAssistant:\n',
                            "output": item["output"]})
            else: 
                data_json.append({"input": f'Human:\n### Instruction:\n{item["instruction"]}\n\nAssistant:\n',
                            "output": item["output"]})
    return data_json


device = "cuda" if torch.cuda.is_available() else "cpu"

target_model = AutoModelForCausalLM.from_pretrained(args.target_model, use_safetensors=True, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(args.target_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"
train_data=load_data(args)

if args.test_num is not None:
    train_data=train_data[:args.test_num]

start=0
end=len(train_data)
save_path=f"{args.output_path}/prediction.jsonl"

if args.split_part is not None and args.split_part !="None":
    part,_,split=args.split_part.partition(":")
    start=int((int(part)-1)/int(split)*len(train_data))
    end=int(int(part)/int(split)*len(train_data))
    save_path=f"{args.output_path}/{args.split_part}.jsonl"

MAX_NEW_TOKENS = args.max_new_tokens
TEMPERATURE=args.temperature

logger.info(f"Target Model -, {target_model.config._name_or_path}")
logger.info("************\n")


minibatch, idxs, answers=[],[], []

hyps, refs=[],[]
all_results=[]

for i in tqdm(range(start, end)):
    text = train_data[i]["input"]
    answer=train_data[i]["output"]
    minibatch.append(text)
    answers.append(answer)
    idxs.append(i)

    if (i+1) % args.bsz !=0 :
        continue
    else:
        inputs = tokenizer(
            minibatch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output_sequences = target_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False, 
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE
            )
        generated_text=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        for idx in range(len(minibatch)):
            prediction=generated_text[idx][len(minibatch[idx]):]
            result={
                "idx":idxs[idx],
                "instruction":minibatch[idx],
                "answer":answers[idx],
                "prediction":prediction
            }
            all_results.append(result)
            hyps.append(prediction if len(prediction) >1 else "None")
            refs.append(answers[idx])
        
        minibatch, idxs, answers=[],[],[]
        
if len(minibatch)>0:
    inputs = tokenizer(
        minibatch,
        return_tensors="pt",
        padding="max_length",
        max_length=args.tokenizer_max_length,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        output_sequences = target_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )
    generated_text=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    for idx in range(len(minibatch)):
        prediction=generated_text[idx][len(minibatch[idx]):]
        result={
            "idx":idxs[idx],
            "instruction":minibatch[idx],
            "answer":answers[idx],
            "prediction":prediction
        }
        all_results.append(result)
        hyps.append(prediction if len(prediction) >1 else "None")
        refs.append(answers[idx])

with jsonlines.open(save_path, mode='w') as writer:
    for result in all_results:
        writer.write(result)

rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)

with open(f"{args.output_path}/results.json", "w") as r:
    json.dump(scores, r, indent=4)