import re
import os
import ast
import argparse
import joblib
import platform
import numpy as np
import pandas as pd
import sentencepiece as spm

from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from typing import List

if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu일 경우
    from konlpy.tag import Mecab


def data_preprocess(
    corpus: List[str],
    save_dir: str,
    stopwords_path: str = None,
    vocab_size: int = 10000,
):
    if stopwords_path:
        stopwords = get_stopwords(stopwords_path)
    else:
        stopwords = None

    # CountVectorization
    vectorizer = CountVectorizer(stop_words=stopwords)
    vectorizer.fit(corpus)

    joblib.dump(vectorizer, os.path.join(save_dir, "sklearn_vectorizer.model"))

    # vocabulary
    vocab = vectorizer.vocabulary_

    vocab_cnt = vectorizer.transform(corpus).sum(axis=0)
    vocab_cnt = sorted(
        [(k, vocab_cnt[0, vocab[k]]) for k in vocab.keys() if not should_filter_word(k)],
        key=lambda x: x[1],
    )
    vocab = {word: vocab[word] for word, _ in vocab_cnt[-vocab_size:]}

    # Train/Test Split
    train, test = train_test_split(corpus, test_size=0.2, shuffle=True, random_state=42)
    train = vectorizer.transform(train)
    test = vectorizer.transform(test)

    save_dataset(save_dir, [train, test], vocab)

    return None


def save_dataset(save_dir, corpus, vocab):
    train, test = corpus
    new_vocab = OrderedDict()
    for k in vocab.keys():
        new_vocab[k] = len(new_vocab) + 1
    itos = {idx: word for word, idx in vocab.items()}

    def _bow(data):
        bow, wf = {}, {}
        for i, j in tqdm(zip(*data.nonzero()), total=len(data.nonzero()[0])):
            if i not in bow:
                bow[i] = []
            if i not in wf:
                wf[i] = []
            if j not in itos:
                continue
            freq = int(data[i, j])
            word = itos[j]

            wf[i].extend([word] * freq)
            bow[i].append(f"{new_vocab[word]}:{freq}")
        bow = [" ".join(v) for v in bow.values() if len(v) > 0]
        wf = [" ".join(v) for v in wf.values() if len(v) > 0]
        return bow, wf

    train_bow, train_txt = _bow(train)
    test_bow, test_txt = _bow(test)

    # save data
    os.makedirs(os.path.join(save_dir, "corpus"), exist_ok=True)

    def _write_lines(dst, lines):
        with open(dst, "w") as f:
            for line in lines:
                f.write(f"{line}\n")

    _write_lines(os.path.join(save_dir, "corpus/train.txt"), train_txt)
    _write_lines(os.path.join(save_dir, "corpus/test.txt"), test_txt)
    _write_lines(
        os.path.join(save_dir, "train.feat"),
        [f"1 {line}" for line in train_bow],
    )
    _write_lines(
        os.path.join(save_dir, "test.feat"),
        [f"1 {line}" for line in test_bow],
    )
    _write_lines(
        os.path.join(save_dir, "vocab"),
        [f"{k} {v}" for k, v in new_vocab.items()],
    )

    return None


def should_filter_word(w):
    REMOVE = r"[a-z가-힣]+"
    return re.fullmatch(REMOVE, w) is None


# Stopwords
def get_stopwords(file_path: str):
    with open(file_path) as f:
        stopwords = [line.strip() for line in f if line.strip()]
    return stopwords


def tokenize_corpus(corpus: List["str"], tokenizer_path: str = None, tokenizer_type: str = "spm"):
    tokenized_corpus = []
    if tokenizer_type == "spm":
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)

        for doc in tqdm(corpus):
            doc = tokenizer.EncodeAsPieces(doc)
            doc = [token for token in doc if len(token) > 2]
            tokenized_corpus.append(" ".join(doc))
    elif tokenizer_type == "mecab":
        tokenizer = Mecab()
        for doc in tqdm(corpus):
            doc = tokenizer.nouns(doc)
            doc = [token for token in doc if len(token) > 1]
            tokenized_corpus.append(" ".join(doc))

    return tokenized_corpus


def get_data(data_path: str):
    df = pd.read_csv(data_path, low_memory=False)
    df = df.rename(columns={"date": "Date", "cleaned_text": "text"})
    df = df[["Date", "text"]]

    data = df["text"].tolist()
    data = [" ".join(ast.literal_eval(doc)) for doc in data]

    return data


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Preprocess for NVDM")
    parser.add_argument("--data_path", type=str, default="data/naver_main_news.csv")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--tokenizer_type", type=str, default="mecab")
    parser.add_argument("--stopwords_path", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default="data")
    args = parser.parse_args()

    # data_dir = args.data_dir
    # data_path = os.path.join(data_dir, "naver_main_news.csv")
    # tokenizer_path = os.path.join(data_dir, "tokenizer/spm_naver_news.model")
    # stopwords_path = os.path.join(data_dir, "stopwords.txt")

    data = get_data(args.data_path)
    print("===== 1. Tokenizing the corpus =====")
    corpus = tokenize_corpus(
        data,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
    )

    print("===== 2. Preprocessing for NVDM dataset =====")
    data_preprocess(
        corpus,
        save_dir=args.save_dir,
        stopwords_path=args.stopwords_path,
        vocab_size=args.vocab_size,
    )

    print("===== Preprocess Done!!! =====")
