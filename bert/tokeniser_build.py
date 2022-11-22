import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from data_prep import save_tf_data
from tokeniser_utils import *

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

#  Code to build vocab
#  Takes about 13 mins
# imdb_vocab = bert_vocab.bert_vocab_from_dataset(
#     train_dataset,
#     **bert_vocab_args
# )
# with open('imdb_vocab.txt', 'w') as f:
#     for line in imdb_vocab:
#         f.write(f"{line}\n")

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

train_dataset_masked = make_batches(train_dataset, lookup_table)
test_dataset_masked = make_batches(test_dataset, lookup_table)

save_tf_data(train_dataset_masked, 'train_mask')
save_tf_data(test_dataset_masked, 'test_mask')
