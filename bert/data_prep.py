import os
import tempfile
from typing import Tuple

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text
import functools
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

import glob
import pandas as pd
import re


def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name):
    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df


train_df = get_data_from_text_files("train")
test_df = get_data_from_text_files("test")

all_data = train_df.append(test_df)


def cln_text(text: str) -> str:
    return re.sub(r'<br />', ' ', text.lower())


all_data['review_cln'] = all_data['review'].map(cln_text)
dataset = tf.data.Dataset.from_tensor_slices(all_data['review_cln'])

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]", "[MASK]", "[RANDOM]", "[CLS]", "[SEP]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

#  Takes about 5 mins
imdb_vocab = bert_vocab.bert_vocab_from_dataset(
    dataset,
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
_MAX_SEQ_LEN = 10
_MAX_PREDICTIONS_PER_BATCH = 5

_VOCAB_SIZE = len(imdb_vocab)


@tf.function
def bert_pretrain_preprocess(vocab_table, features):
    # Input is a string Tensor of documents, shape [batch, 1].
    text_a = features

    # Tokenize segments to shape [num_sentences, (num_words)] each.
    tokenizer = text.BertTokenizer(
        vocab_table,
        token_out_type=tf.int64)

    segments = tokenizer.tokenize(text_a).merge_dims(1, -1)

    # Truncate inputs to a maximum length.
    trimmer = text.RoundRobinTrimmer(max_seq_length=6)
    trimmed_segments = trimmer.trim([segments])

    # Combine segments, get segment ids and add special tokens.
    segments_combined, segment_ids = text.combine_segments(
        trimmed_segments,
        start_of_sequence_id=_START_TOKEN,
        end_of_segment_id=_END_TOKEN)

    # Apply dynamic masking task.
    masked_input_ids, masked_lm_positions, masked_lm_ids = (
        text.mask_language_model(
            segments_combined,
            random_selector,
            mask_values_chooser,
        )
    )

    padded_inputs, _ = text.pad_model_inputs(
        segments_combined, max_seq_length=_MAX_SEQ_LEN)

    # Prepare and pad combined segment inputs
    input_word_ids, input_mask = text.pad_model_inputs(
        masked_input_ids, max_seq_length=_MAX_SEQ_LEN)

    input_type_ids, _ = text.pad_model_inputs(
        segment_ids, max_seq_length=_MAX_SEQ_LEN)

    # # Prepare and pad masking task inputs
    # masked_lm_positions, masked_lm_weights = text.pad_model_inputs(
    #     masked_lm_positions, max_seq_length=_MAX_PREDICTIONS_PER_BATCH)

    masked_lm_ids, _ = text.pad_model_inputs(
        masked_lm_ids, max_seq_length=_MAX_PREDICTIONS_PER_BATCH)

    # model_inputs = {
    #     "segment_ids": padded_inputs,
    #     "input_word_ids": input_word_ids,
    #     "input_mask": input_mask,
    #     "input_type_ids": input_type_ids,
    #     "masked_lm_ids": masked_lm_ids,
    #     "masked_lm_positions": masked_lm_positions,
    #     "masked_lm_weights": masked_lm_weights,
    # }
    return padded_inputs, input_word_ids


def make_batches(ds, lk_up, BUFFER_SIZE: int = 20000, BATCH_SIZE: int = 64):
    """
    It tokenizes the text, and filters out the sequences that are too long. (The batch/unbatch is included because the
    tokenizer is much more efficient on large batches). The cache method ensures that that work is only executed once.
    Then shuffle and, dense_to_ragged_batch randomize the order and assemble batches of examples. Finally, prefetch runs
    the dataset in parallel with the model to ensure that data is available when needed. See Better performance with the
    tf.data for details.
    :param ds: Tensorflow dataset
    :param BUFFER_SIZE: Size of buffer (randomly samples elements from buffer)
    :param BATCH_SIZE: No. of elements within a batch
    :return:
    """
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(functools.partial(bert_pretrain_preprocess, lookup_table))
        .prefetch(buffer_size=tf.data.AUTOTUNE))


dataset = make_batches(tf.data.Dataset.from_tensor_slices(all_data['review_cln']), lookup_table)

