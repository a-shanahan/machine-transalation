import os
import tempfile
from typing import Tuple

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text
import functools
from dataclasses import dataclass
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tokeniser_utils import *


config = Config()

train_dataset = load_tf_data('train_data')
test_dataset = load_tf_data('test_data')

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=config.VOCAB_SIZE,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

#  Takes about 13 mins
imdb_vocab = bert_vocab.bert_vocab_from_dataset(
    train_dataset,
    **bert_vocab_args
)

lookup_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=imdb_vocab,
        key_dtype=tf.string,
        values=tf.range(
            tf.size(imdb_vocab, out_type=tf.int64), dtype=tf.int64),
        value_dtype=tf.int64
    ),
    num_oov_buckets=1
)

_START_TOKEN = imdb_vocab.index("[CLS]")
_END_TOKEN = imdb_vocab.index("[SEP]")
_MASK_TOKEN = imdb_vocab.index("[MASK]")
_UNK_TOKEN = imdb_vocab.index("[UNK]")

trimmer = text.RoundRobinTrimmer(max_seq_length=config.MAX_SEQ_LEN)

random_selector = text.RandomItemSelector(
    max_selections_per_batch=config.MAX_PREDICTIONS_PER_BATCH,
    selection_rate=0.2,
    unselectable_ids=[_START_TOKEN, _END_TOKEN, _UNK_TOKEN]
)

mask_values_chooser = text.MaskValuesChooser(config.VOCAB_SIZE, _MASK_TOKEN, 0.8)

train_dataset_masked = make_batches(train_dataset, lookup_table)
test_dataset_masked = make_batches(test_dataset, lookup_table)

