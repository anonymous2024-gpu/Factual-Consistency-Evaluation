from factscore.factscorer import FactScorer

fs = FactScorer()
# this will create a database using your file
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source("aggreFact_corpus",
                             data_path="aggreFact_corpus.jsonl",
                             db_path=None)