# Revisiting Knowledge Distillation for Autoregressive Language Models
This is the official implementation of our ACL2024 paper, "[Revisiting Knowledge Distillation for Autoregressive Language Models](https://aclanthology.org/2024.acl-long.587/)" (in Pytorch).

## Requirements and Installation

- fschat == 0.2.23
- transformers
- openai == 0.28
- PyTorch version >= 1.10.0
- Python version >= 3.8
- For training, you'll also need an NVIDIA GPU and NCCL.
- To install **instruct-eval** and develop locally:

``` bash
git clone https://github.com/declare-lab/instruct-eval.git
cd instruct-eval
pip install -r requirements.txt
```


# Getting Started
Here, we introduce how to reproduce our experimental results in the paper.

## Model Training
The training data and scripts can be found in "./data" and "./scripts", respectively. Specifically, the training scripts are as follows:
``` bash
datapath=./outputs
kl=$1 # KD methods: ["forward", "reverse", "jsd", "dkd", "seqkd", "tvd", "atkd"], default: "atkd"
cuda=$2 # cuda ids
gpus=$3 # numbers of gpus
port=$4 # master port
student_size=$5 # student model size
teacher_size=$6 # teacher model size
per_bsz=$7 # batch size, default: 1
setting=$8 # setting for ATKD: ["vanilla", "reverse"], deafult: "vanilla"
sample=$9 #training data type: ["student", "teacher", "gt", "mix_token", "mix_request_teacher", "mix_request_gt"], default: "gt"

target_model=fine-tuned/OPT-${student_size}

draft_model=original/OPT-${teacher_size}

save_path=$datapath/alpaca_${student_size}_vs_OPT_${teacher_size}/alpaca_${sample}_${kl}_${setting}

if [ "$sample" == "gt" ]; then
    data_path=alpaca_gpt4_data.json
else
    data_path=generated_data/OPT_${student_size}/alpaca_generated_data.jsonl
fi

mkdir -p ${save_path}

CUDA_VISIBLE_DEVICES=${cuda} python3 -m torch.distributed.launch \
    --nproc_per_node ${gpus} \
    --master_port ${port} \
     distill/train.py \
    --student_model_path ${draft_model} \
    --teacher_model_path ${target_model} \
    --data_path ${data_path} \
    --task alpaca \
    --setting ${setting} \
    --max_propose_num 5 \
    --bf16 True \
    --output_dir ${save_path} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${per_bsz} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name distillation_${sample}_${kl}\
    --mode offline \
    --sample_source $sample \
    --kl_method $kl \
    2>&1 | tee ${save_path}/training.log
```

## Evaluation
You can use the following scripts to evaluate the student models: 
``` bash
cuda=$1
kl=$2
student_size=$3
teacher_size=$4
per_bsz=$5
setting=$6
sample=$7

bash ./script/evaluation.sh $cuda $kl $student_size $teacher_size $per_bsz $setting $sample
```
## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@inproceedings{zhong2024revisiting,
  title={Revisiting knowledge distillation for autoregressive language models},
  author={Zhong, Qihuang and Ding, Liang and Shen, Li and Liu, Juhua and Du, Bo and Tao, Dacheng},
  booktitle={ACL},
  year={2024}
}
```



