# download_data.sh
set -e
source /pasteur/u/jmhb/miniconda3/etc/profile.d/conda.sh
conda activate retriever

save_path=data/
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
# (2) Process the NQ dataset.

python scripts/data_process/nq_search.py


# bm25 content 
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name

