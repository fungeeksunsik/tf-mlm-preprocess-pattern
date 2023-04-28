# parameters required to download IMDb dataset
LOCAL_DIR = "/tmp/mlmPattern"
SOURCE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_NAME = "imdb.tar.gz"

# sentencepiece tokenizer parameters
CORPUS_FILE_NAME = "corpus.txt"
SPM_TRAINER_CONFIG = {
    "input": f"{LOCAL_DIR}/{CORPUS_FILE_NAME}",
    "model_prefix": f"{LOCAL_DIR}/tokenizer",
    "vocab_size": 25000,
    "model_type": "bpe",
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "pad_piece": "[PAD]",
    "unk_piece": "[UNK]",
    "bos_piece": "[CLS]",
    "eos_piece": "[SEP]",
    "add_dummy_prefix": True,
    "train_extremely_large_corpus": False,
}
