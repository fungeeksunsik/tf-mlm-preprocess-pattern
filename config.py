# parameters required to download IMDb dataset
LOCAL_DIR = "/tmp/mlmPattern"
SOURCE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_NAME = "imdb.tar.gz"

# sentencepiece tokenizer parameters
CORPUS_FILE_NAME = "corpus.txt"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
VOCAB_SIZE = 25000
MASK_TOKEN = "[MASK]"
SPM_TRAINER_CONFIG = {
    "input": f"{LOCAL_DIR}/{CORPUS_FILE_NAME}",
    "model_prefix": f"{LOCAL_DIR}/tokenizer",
    "vocab_size": VOCAB_SIZE,
    "model_type": "bpe",
    "pad_id": PAD_ID,
    "unk_id": UNK_ID,
    "bos_id": BOS_ID,
    "eos_id": EOS_ID,
    "pad_piece": "[PAD]",
    "unk_piece": "[UNK]",
    "bos_piece": "[CLS]",
    "eos_piece": "[SEP]",
    "user_defined_symbols": MASK_TOKEN,
    "split_by_number": True,
    "add_dummy_prefix": True,
    "train_extremely_large_corpus": False,
}

# MLM task parameters
SEQUENCE_MAX_LEN = 5
SELECTION_PROB = 0.2
MAX_SELECTION = int(SEQUENCE_MAX_LEN * SELECTION_PROB)
