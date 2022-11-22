import tensorflow as tf
import tensorflow_text as text


@dataclass
class Config:
    MAX_SEQ_LEN = 256
    MAX_PREDICTIONS_PER_BATCH = 5
    VOCAB_SIZE = 30000
    BATCH_SIZE = 32
    EMBED_DIM = 256  # Dimensionality of embeddings
    NUM_HEAD = 4  # No. of attention heads
    FF_DIM = 512  # Dimensionality of feed forward network
    NUM_LAYERS = 4  # No. of layers
    DROPOUT = 0.1


def load_tf_data(dir_name: str) -> tf.data.Dataset:
    path = os.path.join('data', dir_name)
    return tf.data.Dataset.load(path)


@tf.function
def bert_pretrain_preprocess(vocab_table, feature):
    # Input is a string Tensor of documents, shape [batch, 1].
    # Tokenize segments to shape [num_sentences, (num_words)] each.
    tokenizer = text.BertTokenizer(
        vocab_table,
        token_out_type=tf.int64)

    segments = tokenizer.tokenize(feature).merge_dims(1, -1)

    # Truncate inputs to a maximum length.
    trimmer = text.RoundRobinTrimmer(max_seq_length=config.MAX_SEQ_LEN)
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
        segments_combined, max_seq_length=config.MAX_SEQ_LEN)

    # Prepare and pad combined segment inputs
    masked_word_ids, input_mask = text.pad_model_inputs(
        masked_input_ids, max_seq_length=config.MAX_SEQ_LEN)

    return masked_word_ids, padded_inputs


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
        .map(functools.partial(bert_pretrain_preprocess, lk_up))
        .prefetch(buffer_size=tf.data.AUTOTUNE))
