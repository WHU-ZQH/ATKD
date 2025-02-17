datapath=output_dir
sample=$9
kl=$1
cuda=$2
gpus=$3
port=$4
student_size=$5
teacher_size=$6
per_bsz=$7
setting=$8

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
