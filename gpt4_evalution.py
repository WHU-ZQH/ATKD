import argparse
from ast import parse
import json
import jsonlines
import openai
from tqdm import tqdm
import os

import asyncio
from typing import Any
import logging

import httpx
import openai
import sys

openai.api_key=sys.argv[1]


logging.basicConfig(level=logging.INFO)

from multiprocessing import Process, Manager

def openai_api_call(outputs, message, idx, triplet):
    # async_responses = openai_client.chat.completions.create(
    async_responses = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=message,
        temperature=0.0,
        max_tokens=64,
        top_p=0.0,
    )
    output = async_responses['choices'][0]['message']['content']

    meta_output = {
                "idx": idx,
                "review":output
            }
    outputs[idx]=meta_output


if __name__ == "__main__":
    input_review_file=sys.argv[2]
    data=[]
    
    with open(f"{input_review_file}/prediction.jsonl", "r", encoding="utf-8") as f:
        reader=jsonlines.Reader(f)
        for item in reader.iter(type=dict, skip_invalid=True):
            data.append(item)

    prior_results_id=[]
    if os.path.exists(f"{input_review_file}/evaluation.jsonl"):
        with open(f"{input_review_file}/evaluation.jsonl", "r", encoding="utf-8") as f:
            reader=jsonlines.Reader(f)
            for item in reader.iter(type=dict, skip_invalid=True):
                prior_results_id.append(item["idx"])

    system_prompt = "We would like to request your feedback on the performance of two AI assistants in response to the question. Please rate the helpfulness, relevance, accuracy, and level of detail of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease directly output a single line containing only one value indicating the score for Assistant 1 and 2, respectively, by strictly following this format: \"Assistant 1: <score>\nAssistant 2: <score>\". Please avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment."


    ### pos_evaluation
    outputs_final=[]
    with Manager() as manager:
        outputs =  manager.dict()

        process_list = []
        for idx in tqdm(range(len(data))):
            if idx in prior_results_id:
                continue
            eval_prompt = f"Question: {data[idx]['instruction']}\Assistant 1: {data[idx]['prediction']}\Assistant 2: {data[idx]['answer']}"
            message =[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": eval_prompt,
                        },
            ]
            
            p = Process(target=openai_api_call, args=(outputs, message, idx, eval_prompt))
            p.start()
            process_list.append(p)

            if (idx+1)%32==0:
                for res in process_list:
                    res.join()

                for key, value in sorted(dict(outputs).items(), key=lambda item:item[0]):
                    outputs_final.append(value)
                process_list=[]
                outputs =  manager.dict()

                with jsonlines.open(f"{input_review_file}/evaluation.jsonl", mode='a') as writer:
                    for line in outputs_final:
                        writer.write(line)
                outputs_final=[]

        for res in process_list:
            res.join()

        for key, value in sorted(dict(outputs).items(), key=lambda item:item[0]):
            outputs_final.append(value)
        process_list=[]

        with jsonlines.open(f"{input_review_file}/evaluation.jsonl", mode='a') as writer:
            for line in outputs_final:
                writer.write(line)