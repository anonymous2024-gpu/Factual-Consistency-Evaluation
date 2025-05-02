import pandas as pd
import json
import os

data_path = "./GenData/CNNDM/summarization_llama31_results.csv"

ds = pd.read_csv(data_path)

base_dir = os.path.dirname(data_path)

bart_path = os.path.join(base_dir, "cnndm_llama31_few.jsonl") ##Change here


with open(bart_path, 'w') as f:
    for _, row in ds.iterrows(): 
        orca_data = {
            'topic': row['id'],
            'output': row['few_llama31_summary'] ##Change here
        }
        f.write(json.dumps(orca_data) + "\n")

