from model import *
import tensorflow_text as text
import tensorflow as tf
from tokeniser_utils import Config, load_tf_data

with open('imdb_vocab.txt', 'r') as f:
    imdb_vocab = f.read().splitlines()

config = Config(imdb_vocab)

train_dataset = load_tf_data('train_mask')
test_dataset = load_tf_data('test_mask')

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

transformer.fit(train_dataset,
                epochs=10,
                validation_data=test_dataset,
                callbacks=[generator_callback])

try:
    os.mkdir('checkpoints')
except FileExistsError:
    print('Directory already exists')

transformer.save_weights('/checkpoints/my_checkpoint')


# To reload model and perform inference:
# loaded_model = imdbBERT(
#     num_layers=config.NUM_LAYERS,
#     d_model=config.EMBED_DIM,
#     num_heads=config.NUM_HEAD,
#     dff=config.FF_DIM,
#     input_vocab_size=config.VOCAB_SIZE,
#     target_vocab_size=config.VOCAB_SIZE,
#     dropout_rate=config.DROPOUT)
#
# loaded_model.load_weights('/checkpoints/my_checkpoint')
#
# prediction = loaded_model.predict(masked_token_ids)
# masked_index = np.where(masked_token_ids == 2)
# masked_index = masked_index[1]
# mask_prediction = prediction[0][masked_index]
# top_indices = mask_prediction[0].argsort()[-5 :][::-1]
# values = mask_prediction[0][top_indices]
# values

