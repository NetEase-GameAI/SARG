export CUDA_VISIBLE_DEVICES=0
python run_eval.py \
	--test_path Restoration_200k_data/test.txt \
	--model_path sarg-ailab-on-cov-best/eval_loss-xxxx/ \
	--tokenizer_path BertTokenizer \
    --mode ailab \
    --language zh \
    --add_len 15 \
    --batch_size 4
