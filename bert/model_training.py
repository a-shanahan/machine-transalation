import os
import tempfile
from model import *
import tensorflow_text
from tensorflow.keras.callbacks import Callback

num_layers = 4  # No. of layers
d_model = 128  # Dimensionality of embeddings
dff = 512  # Dimensionality of feed forward network
num_heads = 8  # No. of attention heads
dropout_rate = 0.1

transformer = imdbBERT(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=len(imdb_vocab),
    target_vocab_size=len(imdb_vocab),
    dropout_rate=dropout_rate)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])


sample_tokens = tokenizer.tokenize(["watched this film and it was awesome"])

selected = random_selector.get_selection_mask(
    sample_tokens, axis=1)

masked_token_ids, masked_pos, masked_lm_ids = text.mask_language_model(
  sample_tokens,
  item_selector=random_selector, mask_values_chooser=mask_values_chooser)

sample_tokens = masked_token_ids.to_tensor()


from tensorflow.keras.callbacks import Callback


class MaskedTextGenerator(Callback):
    def __init__(self, sample_tokens, top_k=5):
        print('Generator __init__')
        self.sample_tokens = sample_tokens
        self.k = top_k
        self.mask_token_id = 2

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            print(result)

generator_callback = MaskedTextGenerator(sample_tokens)
generator_callback.sample_tokens

transformer.fit(dataset,
                epochs=5,
                callbacks=[generator_callback])

