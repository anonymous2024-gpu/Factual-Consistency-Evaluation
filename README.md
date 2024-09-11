### FActScore Evaluation

1. Install Required Packages
   - `pip install -r requirements.txt`
2. Login to Huggingface
   - `huggingface-cli login`
3. Download the Model
   - `python -m factscore.download_data --llama_7B_HF_path "meta-llama/Llama-2-7b-chat-hf"`
4. Register a New Corpus
   - `python -m factscore.preprocess`
5. Run FActScore
   - `python -m factscore.factscorer --input_path "ChatGPT_bbc.jsonl" --model_name "retrieval+llama+npm" --knowledge_source "bbc" --n_samples "50"`

### LongDocFACTScore Evaluation

