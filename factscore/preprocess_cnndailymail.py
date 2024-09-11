from factscore.factscorer import FactScorer
from datasets import load_dataset
import json

ds = load_dataset("abisee/cnn_dailymail", "3.0.0")

corpus_path = "cnndm_testcorpus.jsonl"
output_path = "cnndm_test.jsonl"

with open(corpus_path, 'w') as f:
    for i in ds['test']:  # You can also do this for 'validation' and 'train' splits
        corpus_data = {
            'title': i['id'],
            'text': i['article']
        }
        f.write(json.dumps(corpus_data) + "\n")

with open(output_path, 'w') as f:
    for j in ds['test']:  # You can also do this for 'validation' and 'train' splits
        highlight_data = {
            'topic': j['id'],         # Using 'id' as the title
            'output': j['highlights'] # Using 'highlights' as the output
        }
        f.write(json.dumps(highlight_data) + "\n")

fs = FactScorer()
# this will create a database using your file
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source("cnndm_testcorpus",
                             data_path=corpus_path,
                             db_path=None)