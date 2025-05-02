import pandas as pd
import json
import os

data_path = "D:\\FActScore-main2\\FActScore-main\\GenData\\LongSciVerify\\longsciverify_llama2chat7b.csv"

ds = pd.read_csv(data_path)

base_dir = os.path.dirname(data_path)

bart_path = os.path.join(base_dir, "longsciverify_llama2chat7b.jsonl")
# t5_path = os.path.join(base_dir, "dialog_Vic7b13.jsonl")
# gpt_path = os.path.join(base_dir, "aggre_cnndmftsota_gpt35.jsonl")

# def extract_summary(text):
#     summary_keyword = "SUMMARY:\n"
#     if summary_keyword in text:
#         return text.split(summary_keyword, 1)[1].strip()
#     return text

# ds['Orca7b2_Summary_normalized'] = ds['Orca7b2_Summary'].apply(extract_summary)
# ds['Vic7b13_Summary_normalized'] = ds['Vic7b13_Summary'].apply(extract_summary)

with open(bart_path, 'w') as f:
    for _, row in ds.iterrows(): 
        orca_data = {
            'topic': row['article_id'],
            'output': row['llama2chat7b_summary']
        }
        f.write(json.dumps(orca_data) + "\n")

# with open(t5_path, 'w') as f:
#     for _, row in ds.iterrows(): 
#         vic_data = {
#             'topic': row['id'],
#             'output': row['Vic7b13_Summary_normalized']
#         }
#         f.write(json.dumps(vic_data) + "\n")

# with open(gpt_path, 'w') as f:
#     for _, row in ds.iterrows():  
#         gpt_data = {
#             'topic': row['id'], 
#             'output': row['GPT35_Summary'] 
#         }
#         f.write(json.dumps(gpt_data) + "\n")