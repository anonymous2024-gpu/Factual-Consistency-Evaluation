from factscore.openai_lm import OpenAIModel
from factscore.factscorer import FactScorer
import pandas as pd
import tqdm
import json
import openai
from dotenv import load_dotenv
import os

# File downloaded from https://github.com/shauryr/ACL-anthology-corpus
# https://drive.google.com/file/d/1CFCzNGlTls0H-Zcaem4Hg_ETj4ebhcDO/view?usp=sharing
df = pd.read_parquet('D:\\FActScore-main2\\FActScore-main\\preprocessing\\acl-publication-info.74k.parquet')
titles = df['title'].tolist()
full_text = df['full_text'].tolist()

acl_corpus = []
for x, y in zip(titles, full_text):
    if x.strip() == "" or y.strip() == "":
        continue
    acl_corpus.append({"title": x, "text": y})

# with open("acl_corpus.jsonl", 'w') as f:
#     for line in acl_corpus:
#         f.write(json.dumps(line) + "\n")

# fs = FactScorer()
# # this will create a database using your file
# # once DB file is created, you can reuse it by only specifying `db_path`
# fs.register_knowledge_source("acl_corpus",
#                              data_path="acl_corpus.jsonl",
#                              db_path=None)


# prompt_titles = [
#     "Dense Passage Retrieval for Open-Domain Question Answering",
#     "AmbigQA: Answering Ambiguous Open-domain Questions",
#     "MetaICL: Learning to Learn In Context",
#     "Noisy Channel Language Model Prompting for Few-Shot Text Classification",
#     "Joint Passage Ranking for Diverse Multi-Answer Retrieval",
#     "Reformulating Unsupervised Style Transfer as Paraphrase Generation",
#     "Syntactically Supervised Transformers for Faster Neural Machine Translation",
#     "Hurdles to Progress in Long-form Question Answering",
#     "Generating Question-Answer Hierarchies",
#     "Do Long-Range Language Models Actually Use Long-Range Context?"
# ]

# df_sorted = df.sort_values(by='year', ascending=False)
top_50_df = df.head(50)
prompt_titles = top_50_df['title'].tolist()

prompts_list = []

for title in prompt_titles:
    prompts_list.append(f"Give me a summary of the research paper titled \"{title}\". Return only the summary as your response in a paragraph which covers the key points of the research paper. SUMMARY:")

# with open("api.key", 'r') as f:
#     api_key = f.readline()
# openai.api_key = api_key.strip()

def load_model(self):
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    self.client = openai.OpenAI(api_key=api_key.strip())
    self.model = self.model_name

def extract_summary(text):
    summary_keyword = "SUMMARY:\n"
    if summary_keyword in text:
        return text.split(summary_keyword, 1)[1].strip()
    return text

responses = []

cache_dir = ".cache/factscore"
openai_model = OpenAIModel("InstructGPT", cache_file=os.path.join(cache_dir, "InstructGPT_af.pkl"))
openai_model.load_model()

for ptitle, prompt in tqdm.tqdm(zip(prompt_titles, prompts_list)):
    message = [
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": prompt}
    ]
    output, _ = openai_model.call_GPT3(message=message)
    summary = extract_summary(output)
    responses.append({
        "topic": ptitle,
        "output": summary
    })

# # write the corpus to a jsonl file
with open("acl_gptInstruct.jsonl", 'w') as f:
    for line in responses:
        f.write(json.dumps(line) + "\n")
