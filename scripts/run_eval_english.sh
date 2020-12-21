export CUDA_VISIBLE_DEVICES=0
python run_eval.py \
    --test_path CANARD/test.txt \
    --model_path sarg-canard-no-cov-best/eval_loss-xxxx \
    --tokenizer_path bert-base-uncased \
    --mode canard \
    --language en \
    --add_len 30 \
    --batch_size 4 \
    --forbid_repeat False \
    --max_ctx_len 256 \
    --max_src_len 256 \
    --add_len 30
