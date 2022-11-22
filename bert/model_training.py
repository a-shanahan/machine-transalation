from model import *
import tensorflow_text as text
import tensorflow as tf
from tokeniser_utils import Config

config = Config()

_START_TOKEN = imdb_vocab.index("[CLS]")
_END_TOKEN = imdb_vocab.index("[SEP]")
_MASK_TOKEN = imdb_vocab.index("[MASK]")
_UNK_TOKEN = imdb_vocab.index("[UNK]")

transformer = imdbBERT(
    num_layers=config.NUM_LAYERS,
    d_model=config.EMBED_DIM,
    num_heads=config.NUM_HEAD,
    dff=config.FF_DIM,
    input_vocab_size=config.VOCAB_SIZE,
    target_vocab_size=config.VOCAB_SIZE,
    dropout_rate=config.DROPOUT)


learning_rate = CustomSchedule(config.EMBED_DIM)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

#  Text to test with: "i have watched this [MASK] and it was awesome"
sample_tokens = tf.ragged.constant([[28, 100, 365, 86, 2, 80, 85, 88, 1223]])

masked_token_ids, _ = text.pad_model_inputs(
    sample_tokens, max_seq_length=config.MAX_SEQ_LEN)

generator_callback = MaskedTextGenerator(masked_token_ids, imdb_vocab)

transformer.fit(train_dataset_masked,
                epochs=3,
                validation_data=test_dataset_masked,
                callbacks=[generator_callback])


