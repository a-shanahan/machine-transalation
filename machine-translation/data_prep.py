import os
import tempfile
from typing import Tuple

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

# Download dataset
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)

tokenizers = tf.saved_model.load(model_name)

for pt_examples, en_examples in train_examples.batch(3).take(1):
    print('> Examples in Portuguese:')
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()

    print('> Examples in English:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))


def prepare_batch(input_lang, output_lang, MAX_TOKENS: int = 128) -> Tuple:
    """
    Tokenizes inputs into ragged batches. It trims each to be no longer than MAX_TOKENS. It splits the target tokens
    into inputs and labels. These are shifted by one step so that at each input location the label is
    the id of the next token. It converts the RaggedTensors to padded dense Tensors. It returns an (inputs,
    labels) pair.

    :param input_lang: Batch of input language
    :param output_lang: Batch of output language
    :param MAX_TOKENS: Maximum number of tokens
    :return: inputs and labels pair
    """
    input_lang = tokenizers.pt.tokenize(input_lang)  # Output is ragged.
    input_lang = input_lang[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    input_lang = input_lang.to_tensor()  # Convert to 0-padded dense Tensor

    output_lang = tokenizers.en.tokenize(output_lang)
    output_lang = output_lang[:, :(MAX_TOKENS + 1)]
    en_inputs = output_lang[:, :-1].to_tensor()  # Drop the [END] tokens
    # Labels are inputs shifted to the left
    en_labels = output_lang[:, 1:].to_tensor()  # Drop the [START] tokens

    return (input_lang, en_inputs), en_labels


def make_batches(ds, BUFFER_SIZE: int = 20000, BATCH_SIZE: int = 64):
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
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))


# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

train_path = os.path.join(tempfile.gettempdir(), "saved_train_data")
val_path = os.path.join(tempfile.gettempdir(), "saved_val_data")

train_batches.save(train_path)
val_batches.save(val_path)
