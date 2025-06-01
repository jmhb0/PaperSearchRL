# source /pasteur/u/jmhb/miniconda3/etc/profile.d/conda.sh
source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
conda activate retriever 

# corpus_file=data/pubmed.jsonl
# save_dir=data/pubmed/
# retriever_name=e5 # this is for indexing naming
# retriever_model=intfloat/e5-base-v2

# # change faiss_type to HNSW32/64/128 for ANN indexing
# # change retriever_name to bm25 for BM25 indexing
# CUDA_VISIBLE_DEVICES=0,1,2,3 python search_r1/search/index_builder.py \
#     --retrieval_method $retriever_name \
#     --model_path $retriever_model \
#     --corpus_path $corpus_file \
#     --save_dir $save_dir \
#     --use_fp16 \
#     --max_length 512 \
#     --batch_size 512 \
#     --pooling_method mean \
#     --faiss_type Flat \
#     --save_embedding


corpus_file=data/pubmed.jsonl
save_dir=data/pubmed_bm25/
retriever_name=bm25 # this is for indexing naming
retriever_model='none'

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
# CUDA_VISIBLE_DEVICES=0,1,2,3 
python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
