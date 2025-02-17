datapath=./outputs
cuda=$1
kl=$2
student_size=$3
teacher_size=$4
per_bsz=$5
setting=$6
sample=$7

save_path=$datapath/alpaca_${student_size}_vs_pythia_${teacher_size}/alpaca_${sample}_${kl}_${setting}

target_model=$save_path
output_path=evaluation_results/alpaca_${student_size}_vs_pythia_${teacher_size}/alpaca_${sample}_${kl}_${setting}

for task in dolly vicuna self-inst koala wizardlm
do
mkdir -p ${output_path}/${task}

CUDA_VISIBLE_DEVICES=${cuda} python3 inference.py \
    --target_model ${target_model} \
    --target_task ${task} \
    --output_path ${output_path}/${task} \
    --max_new_tokens 256 \
    --bsz 16 \
    --temperature 0.0

python3 gpt4_evalution.py ${key} ${output_path}/${task}
python3 get_gpt4_score.py ${output_path}/${task}
done

for task in mmlu drop bbh
do
mkdir -p ${output_path}/${task}

CUDA_VISIBLE_DEVICES=${cuda} python3 instruct-eval/main.py ${task} \
    --model_name causal \
    --model_path ${target_model} \
    --save_path ${output_path}/${task}
done