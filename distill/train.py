# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import jsonlines
from fastchat.model.model_adapter import get_conversation_template
import os
os.environ["WANDB_DISABLED"]="true"

from distill_trainer import DistillTrainer
from transformers import Trainer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

import logging
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    student_model_path: Optional[str] = field(
        default="facebook/opt-125m",  metadata={"help": "Path to student model"})
    teacher_model_path: str = field(
        default=None, metadata={"help": "Path to teacher model"})
    assistant_model_path: str = field(
        default=None, metadata={"help": "Path to assistant model"})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    task: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    decoder_max_length: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_propose_num: int = field(
        default=5,
        metadata={
            "help": "gamma, number of tokens the student model proposes for each step"
        }
    )
    mode: str = field(
        default="offline",
        metadata={
            "help": "online mode or offline mode"
        }
    )
    trainer: str = field(
        default="regular",
        metadata={
            "help": "trainer mode"
        }
    )
    non_diff: bool = field(
        default=False,
        metadata={
            "help": "non_diff for MeZo method"
        }
    )
    online_eval_interval: int = field(
        default=10,
        metadata={
            "help": "evaluation interval for online training"
        }
    )
    zo_eps: float = field(
        default=1e-3,
        metadata={
            "help": "zo_eps for MeZo method"
        }
    )
    nckd_percent: float = field(
        default=0.5,
        metadata={
            "help": "top k in nckd loss"
        }
    )
    online_update_interval: int = field(
        default=1,
        metadata={
            "help": "parameter update interval for online training"
        }
    )
    alpha: float = field(
        default=0.2,
        metadata={
            "help": "alpha for tckd"
        }
    )
    topk: float = field(
        default=None,
        metadata={
            "help": "alpha for tckd"
        }
    )
    beta: float = field(
        default=1.0,
        metadata={
            "help": "beta for nckd"
        }
    )
    sample_source: str = field(
        default="gt",
        metadata = {
            "choices" : ["student", "teacher", "gt", "mix_token", "mix_request_teacher", "mix_request_gt"]
        }
    )
    kl_method: str = field(
        default="forward",
        metadata = {
            "choices" : ["forward", "reverse", "jsd", "dkd", "seqkd", "tvd", "atkd"]
        }
    )
    setting: str = field(
        default="vanilla",
        metadata = {
            "choices" : ["vanilla", "reverse"]
        }
    )


local_rank = None

task2template={
    "alpaca_instruction": "Human:\n{input}\n\nAssistant:\n",
    "alpaca": "Human:\n{input}\n\nAssistant:\n{output}",
}

def rank0_print(*args):
    if local_rank == 0:
        logger.info(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    sample_source: str,
    do_eval: bool,
    args=None
) -> Dict:

    if sample_source =="gt":
        instruction=task2template[f"{args.task}_instruction"].format_map(sources[0])
        conversations=task2template[f"{args.task}"].format_map(sources[0])
    else:
        instruction=sources[0]["input"]
        conversations=sources[0]["input"]+sources[0]["output"]

    if sample_source in ["gt", "teacher"]:
        assert tokenizer.padding_side=="right", f"For gt and teacher sampling, tokenizer.padding_side should be right."
    else:
        tokenizer.padding_side="left"

    tokenizer.add_eos=True
    input_ids_instruction = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    if sample_source in ["student", "mix_token", "mix_request_teacher", "mix_request_gt"]:
        assert tokenizer.padding_side=="left", f"For on-policy sampling and mix_token sampling, tokenizer.padding_side should be left."
        prompt_targets = input_ids_instruction.clone()
        prompt_targets=torch.where(prompt_targets[...,:]==tokenizer.pad_token_id, torch.full_like(prompt_targets, -100), prompt_targets)

        tokenizer.add_bos=False
        tokenizer.padding_side="right"
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        if input_ids[0][0]==tokenizer.bos_token_id:
            input_ids=torch.cat([input_ids_instruction, input_ids[:, 1:]], dim=-1)
        else:
            input_ids=torch.cat([input_ids_instruction, input_ids], dim=-1)

        targets = input_ids.clone()
        targets[...,:input_ids_instruction.size(-1)]=IGNORE_TOKEN_ID
        targets=torch.where(targets[...,:]==tokenizer.pad_token_id, torch.full_like(targets, -100), targets)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            prompt_input_ids=input_ids_instruction,
            prompt_labels=prompt_targets,
            prompt_attention_mask=input_ids_instruction.ne(tokenizer.pad_token_id),
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        if tokenizer.eos_token_id in input_ids_instruction[0]:
            try:
                targets[...,:torch.nonzero(input_ids_instruction[0]==tokenizer.eos_token_id).squeeze()[0]]=IGNORE_TOKEN_ID
            except:
                targets[...,:torch.nonzero(input_ids_instruction[0]==tokenizer.eos_token_id).item()]=IGNORE_TOKEN_ID

        if sample_source !="gt":
            targets=torch.where(targets[...,:]==tokenizer.pad_token_id, torch.full_like(targets, -100), targets)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model: str,
                 sample_source: str,
                 do_eval: bool = False, 
                 args=None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sample_source=sample_source

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model
        self.args=args

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]],
                         self.tokenizer,
                         self.model,
                         self.sample_source,
                         self.do_eval,
                         self.args)
 
        if "prompt_input_ids" in ret.keys():
            ret = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
                prompt_input_ids=ret["prompt_input_ids"][0],
                prompt_labels=ret["prompt_labels"][0],
                prompt_attention_mask=ret["prompt_attention_mask"][0],
            )
        else:
            ret = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
            )
        self.cached_data_dict[i] = ret

        return ret

def load_data(data_args, sample_source):
    if sample_source=="gt":
        if data_args.task=="alpaca":
            data_json=[]
            alpaca_data=json.load(open(f"{data_args.data_path}", "r", encoding="utf-8"))
            for item in alpaca_data:
                data_json.append({"input": f"{item['instruction']} {item['input']}",
                                "output": item["output"]})
            return data_json

    elif sample_source in ["teacher", "student", "mix_token", "mix_request_teacher", "mix_request_gt"]:
        data_json=[]
        with open(f"{data_args.data_path}", "r", encoding="utf-8") as f:
            reader=jsonlines.Reader(f)
            for item in reader.iter(type=dict, skip_invalid=True):
                if sample_source !="mix_request_gt":
                    data_json.append({"input": item["instruction"],
                                    "output": item["teacher_output"]})
                else:
                    data_json.append({"input": item["instruction"],
                                    "output": item["output"]})
        return data_json
        

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
    model: str
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert data_args.lazy_preprocess, "only support lazy process"
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    train_json=load_data(data_args, sample_source=training_args.sample_source)

    train_dataset = dataset_cls(train_json,
                                tokenizer=tokenizer,
                                model=model,
                                sample_source=training_args.sample_source,
                                args=data_args)

    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.student_model_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    # Load model and tokenizer
    # student model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.student_model_path,
        config=config,
        use_safetensors=True,
        cache_dir=training_args.cache_dir, 
    ).to(device)
    model.train()

    if training_args.kl_method !="seqkd":
        # teacher model
        teacher_config = transformers.AutoConfig.from_pretrained(
            model_args.teacher_model_path,
            cache_dir=training_args.cache_dir,
        )
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_path,
            config=teacher_config,
            use_safetensors=True
        ).to(device)
        teacher_model.eval()

        if model_args.assistant_model_path is not None:
            assistant_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.assistant_model_path,
                config=config,
                use_safetensors=True
            ).to(device)
            assistant_model.eval()
        else:
            assistant_model=None

        logger.info(
            f"Teacher Model memory: {teacher_model.get_memory_footprint() / 1024 / 1024} MB")
        logger.info(
            f"Student Model memory: {model.get_memory_footprint() / 1024 / 1024} MB")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.teacher_model_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            trust_remote_code=True,
            use_fast=True,
        )
    else:
        teacher_model=None
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.student_model_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            trust_remote_code=True,
            padding_side="right",
            use_fast=True,
        )

    if training_args.sample_source in ["mix_token", "mix_request_teacher", "mix_request_gt", "student"]:
        copy_model=transformers.AutoModelForCausalLM.from_pretrained(
            model_args.student_model_path,
            use_safetensors=True,
        ).to(model.device)
    else:
        copy_model=None

    # tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.bos_token_id==tokenizer.eos_token_id:
        tokenizer.add_bos_token=False
    tokenizer.pad_token = tokenizer.eos_token
    if training_args.sample_source in ["student", "mix_token", "mix_request_teacher", "mix_request_gt"]:
        tokenizer.padding_side="left"
    else:
        tokenizer.padding_side="right"

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              training_args=training_args,
                                              model=model_args.teacher_model_path)

    if training_args.kl_method =="seqkd":
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module) 
    else:
        if training_args.sample_source == "mix_token":
            model.config.use_cache = True
            teacher_model.config.use_cache = True

        trainer = DistillTrainer(
            model=model, tokenizer=tokenizer,
            teacher_model=teacher_model, copy_model=copy_model, assistant_model=assistant_model,
            args=training_args, **data_module
        )
        

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    # trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
