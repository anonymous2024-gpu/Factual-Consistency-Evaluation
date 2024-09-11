### FActScore Evaluation

1. Install Required Packages
```
pip install -r requirements.txt
```
3. Login to Huggingface
```
huggingface-cli login
```
5. Download the Model
```
python -m factscore.download_data --llama_7B_HF_path "meta-llama/Llama-2-7b-chat-hf"
```
7. Register a New Corpus
```
python -m factscore.preprocess
```
9. Run FActScore
```
python -m factscore.factscorer --input_path "ChatGPT_bbc.jsonl" --model_name "retrieval+llama+npm" --knowledge_source "bbc" --n_samples "50"
```

### LongDocFACTScore Evaluation

1. Run the following
```bash
pip install longdocfactscore
cd evaluation_scripts
git clone https://github.com/neulab/BARTScore.git
pip install -r requirements.txt
```
2. Run scripts,
```bash 
cd ..
python evaluation_scripts/run_evaluation_bbc.py
```
