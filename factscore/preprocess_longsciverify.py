from factscore.factscorer import FactScorer
import pandas as pd
import json

def get_raw_data(filepath):
    with open(filepath) as f:
        data = f.readlines()
    docs = [json.loads(doc) for doc in data]
    return docs

def clean_sent(sentence):
    sentence = sentence.replace("<S>", "")
    sentence = sentence.replace("</S>", "")
    sentence = sentence.replace("<pad>", "")
    sentence = sentence.replace("<br />", "")
    return sentence

def clean_abstract(text_array):
    if isinstance(text_array, str):
        cleaned = clean_sent(text_array)
    else:
        cleaned = ""
        for sentence in text_array:
            sentence = clean_sent(sentence)
            cleaned += f" {sentence} "
    return cleaned

def load_json_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_raw_data(df, raw_path):
    data = load_json_data(raw_path)
    for idx, row in df.iterrows():
        article = [
            a
            for a in data
            if (a["article_id"] == row["id"].strip() or (row["id"] in a["article_id"]))
        ][0]
        df.loc[idx, "abstract_text"] = clean_abstract(article["abstract_text"])
        df.loc[idx, "article_text"] = clean_abstract(article["article_text"])
    return df

arxiv_path = 'D:\\FActScore-main2\\FActScore-main\\raw_data\\LongSciVerify\\arxiv_test.json'
pubmed_path = 'D:\\FActScore-main2\\FActScore-main\\raw_data\\LongSciVerify\\pubmed_test.json'

arxiv_df = pd.DataFrame(load_json_data(arxiv_path))
arxiv_df['id'] = arxiv_df['article_id']
arxiv_df = load_raw_data(arxiv_df, arxiv_path)

pubmed_df = pd.DataFrame(load_json_data(pubmed_path))
pubmed_df['id'] = pubmed_df['article_id']
pubmed_df = load_raw_data(pubmed_df, pubmed_path)

longsciverify_df = pd.concat([arxiv_df, pubmed_df], ignore_index=True)

save_path = 'D:\\FActScore-main2\\FActScore-main\\raw_data\\LongSciVerify\\longsciverify.csv'
longsciverify_df.to_csv(save_path, index=False)

corpus_path = "longsciverify_corpus.jsonl"
output_path = "longsciverify.jsonl"

with open(corpus_path, 'w') as f:
    for _, row in longsciverify_df.iterrows():
        corpus_data = {
            'title': row['article_id'],
            'text': row['article_text']
        }
        f.write(json.dumps(corpus_data) + "\n")

with open(output_path, 'w') as f:
    for _, row in longsciverify_df.iterrows():
        abstract_data = {
            'topic': row['article_id'],    # Using 'id' as the title
            'output': row['abstract_text'] # Using 'abstract' as the output
        }
        f.write(json.dumps(abstract_data) + "\n")

fs = FactScorer()
# this will create a database using your file
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source("longsciverify_corpus",
                             data_path=corpus_path,
                             db_path=None)