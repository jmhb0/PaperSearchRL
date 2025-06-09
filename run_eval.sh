
## 3b models
# python eval/run_inference.py --method direct --run_judge
# python eval/run_inference.py --method cot  --run_judge
# python eval/run_inference.py --method rag  --run_judge

## 7b nmodels
#python eval/run_inference.py --method direct  --model_id Qwen/Qwen2.5-7B-Instruct --run_judge
#python eval/run_inference.py --method cot --model_id Qwen/Qwen2.5-7B-Instruct --run_judge
#python eval/run_inference.py --method rag --model_id Qwen/Qwen2.5-7B-Instruct --run_judge 


## searchr1 models
python eval/run_inference.py --method papersearchr1  --output_path results/eval/papersearchr1-qwen3b.csv --run_judge
exit
python eval/run_inference.py --method searchr1 --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo --run_judge
python eval/run_inference.py --method searchr1 --model_id PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo --model_id Qwen/Qwen2.5-3B-Instruct --run_judge
