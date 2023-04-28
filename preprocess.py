import requests
import pathlib
import shutil
import glob
import pandas as pd
import sentencepiece as spm
from typing import Dict, Union


def download_and_unpack_imdb(
    source_url: str,
    local_dir: pathlib.Path,
    archive_name: str,
) -> str:
    """
    Download archived imdb sentiment analysis dataset from designated data source and unpack archive
    :param source_url: URL to download dataset from
    :param local_dir: local directory to save downloaded dataset
    :param archive_name: file name of archived dataset
    :return: local path to unpacked dataset
    """
    archive_path = local_dir.joinpath(archive_name)
    response = requests.get(source_url)
    with open(archive_path, "wb") as file:
        file.write(response.content)
    shutil.unpack_archive(archive_path, local_dir)
    return str(local_dir.joinpath("aclImdb"))


def extract_data_from_imdb(imdb_path: str) -> pd.DataFrame:
    """
    Extract following information from each review file in IMDb dataset
     - review : actual review that user had left
     - rating : rating value extracted from file name format(reviewId_rating)
     - is_positive : 1 if certain review is positive(i.e. rating >= 5)
     - data_type : 'train' if certain review belongs to train dataset else 'test'
    :param imdb_path: path to unpacked archive
    :return: pandas dataframe with columns ["review", "rating", "is_positive", "data_type"]
    """
    imdb = []
    for data_type in ("train", "test"):
        review_paths = glob.glob(f"{imdb_path}/{data_type}/pos/*.txt")
        review_paths += glob.glob(f"{imdb_path}/{data_type}/neg/*.txt")
        for review_path in review_paths:
            with open(review_path, "r") as file:
                rating = int(review_path.split("_")[1].strip(".txt"))
                review = {
                    "review": file.read(),
                    "rating": rating,
                    "is_positive": int(rating >= 5),
                    "data_type": data_type,
                }
                imdb.append(review)
    return pd.DataFrame(imdb)


def split_save_imdb_data(imdb_data: pd.DataFrame, local_dir: pathlib.Path) -> None:
    """
    Split preprocessed IMDb dataset into train, test and save each respectively into local directory
    :param imdb_data: IMDb data whose values in review column are preprocessed
    :param local_dir: local directory to save train, test datasets
    :return: None
    """
    (
        imdb_data.query("data_type == 'train'")
        .sample(frac=1.0, replace=False)[["review", "rating", "is_positive"]]
        .to_csv(local_dir.joinpath("train.csv"), index=False)
    )
    (
        imdb_data.query("data_type == 'test'")
        .sample(frac=1.0, replace=False)[["review", "rating", "is_positive"]]
        .to_csv(local_dir.joinpath("test.csv"), index=False)
    )


def extract_corpus(local_dir: pathlib.Path):
    """
    Extract corpus from train data with text normalization
    :param local_dir: local directory to save extracted corpus
    :return: save extracted corpus into local_dir and return nothing
    """
    train_data_path = local_dir.joinpath("train.csv")
    corpus = (
        pd.read_csv(train_data_path)["review"]
        .str.lower()
        .str.replace("[^a-z0-9 ]", " ", regex=True)
    )
    corpus_file_path = str(local_dir.joinpath("corpus.txt"))
    with open(corpus_file_path, "w") as file:
        file.writelines("\n".join(corpus))


def train_and_save_tokenizer(spm_trainer_config: Dict[str, Union[int, str, bool]]):
    """
    For more configuration options, see: https://github.com/google/sentencepiece/blob/master/doc/options.md
    :param spm_trainer_config: dictionary that maps configuration parameter to its value
    :return: train and save sentencepiece tokenizer and return nothing
    """
    spm.SentencePieceTrainer.Train(**spm_trainer_config)
