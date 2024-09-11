huggingface-cli login 

python -m factscore.download_data --llama_7B_HF_path "meta-llama/Llama-2-7b-chat-hf"

pip install -r .\requirements.txt

register new corpus: python -m factscore.preprocess_duc2004

pickle error: re-register the corpus (delete previous pkl of that corpus)

python -m factscore.factscorer --input_path "D:\FActScore-main2\FActScore-main\ChatGPT_bbc.jsonl" --model_name "retrieval+llama+npm" --knowledge_source "bbc_knowledge" --n_samples "50"