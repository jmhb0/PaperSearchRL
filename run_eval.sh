export CUDA_AVAILABLE_DEVICES=2,3

models=( "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)
dataset_ids=(
    "jmhb/PaperSearchRL_v5_gv3_n3000_test300"
    "jmhb/PaperSearchRL_v5_gv3_n3000_test300_parav1pcnt50"
    "jmhb/PaperSearchRL_v5_gv3_n3000_test300_filterk1"
    "jmhb/PaperSearchRL_v5_gv3_n3000_test300_parav1pcnt50_filterk1"
    "jmhb/bioasq_trainv0_n1609_test100"
)
#methods=("direct" "cot" "rag")
methods=("rag")
retriever_type="e5" # or "bm25"
for dataset_id in "${dataset_ids[@]}"; do
    for method in "${methods[@]}"; do
        for model in "${models[@]}"; do
            python eval/run_inference.py --method $method --model_id $model --dataset_id $dataset_id --retriever_type $retriever_type
        done
    done
done



