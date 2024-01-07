import tensorflow as tf
import numpy as np

from model_utils import *


def loss_fn(labels, targets):
    labels_mask = tf.math.is_nan(labels)
    labels = tf.where(labels_mask, tf.zeros_like(labels), labels)
    mask_count = tf.math.reduce_sum(
        tf.where(labels_mask, tf.zeros_like(labels), tf.ones_like(labels))
    )
    loss = tf.math.abs(labels - targets)
    loss = tf.where(labels_mask, tf.zeros_like(loss), loss)
    loss = tf.math.reduce_sum(loss) / mask_count
    return loss


def get_model(config, hidden_dim=384, max_len=206):

    # Configure Strategy. Assume TPU...if not set default for GPU
    tpu = None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            tpu="local"
        )  # "local" for 1VM TPU
        strategy = tf.distribute.TPUStrategy(tpu)
        print("on TPU")
        print("REPLICAS: ", strategy.num_replicas_in_sync)
    except:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        inp = tf.keras.Input([max_len])

        x = inp

        x = tf.keras.layers.Embedding(config["num_vocab"], hidden_dim, mask_zero=True)(
            x
        )

        pos = positional_encoding_layer(
            num_vocab=config["num_vocab"], maxlen=500, hidden_dim=hidden_dim
        )(x)

        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(pos)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)

        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)

        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim * 4)(x)

        x = x + pos

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(2)(x)

        model = tf.keras.Model(inp, x)

        loss = loss_fn
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)
        model.compile(loss=loss, optimizer=optimizer, steps_per_execution=100)
        return model
