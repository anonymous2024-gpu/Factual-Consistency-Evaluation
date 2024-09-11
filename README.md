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

Citation:
```bash
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
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

Citation:
```bash
@article{bishop2023longdocfactscore,
  title={LongDocFACTScore: Evaluating the Factuality of Long Document Abstractive Summarisation},
  author={Bishop, Jennifer A and Xie, Qianqian and Ananiadou, Sophia},
  journal={arXiv preprint arXiv:2309.12455},
  year={2023}
}
```
