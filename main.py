import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset import get_tfrec_dataset
from model import get_model
from train_utils import lrfn, plot_lr_schedule


config = {
    "DEBUG": False,
    "PAD_x": 0.0,
    "PAD_y": np.nan,
    "X_max_len": 206,
    "batch_size": 128,
    "val_batch_size": 5512,
    "num_vocab": 5,
    "hidden_dim": 192,
    "tffiles_path": "/input/srrf-tfrecords-ds/tfds",
    "num_training_steps": 300,
    "num_warmup_steps": 0,
    "lr_max": 5e-4,
    "WD_RATIO": 0.05,
    "WARMUP_METHOD": "exp",
}


if config["DEBUG"]:
    config["batch_size"] = 2
    config["val_batch_size"] = 2
    config["num_training_steps"] = 5


def select_tpu():
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


def get_dataset(config):
    tffiles_path = config["tffiles_path"]
    tffiles = [f"{tffiles_path}/{x}.tfrecord" for x in range(164)]

    val_len = 5
    if config["DEBUG"]:
        val_len = 1

    val_files = tffiles[:val_len]

    if config["DEBUG"]:
        train_files = tffiles[val_len : val_len + 1]
    else:
        train_files = tffiles[val_len:]

    train_dataset, num_train = get_tfrec_dataset(
        train_files,
        shuffle=-1,
        batch_size=config["batch_size"],
        config=config,
        cache=True,
        to_filter="filter_2",
        calculate_sample_num=True,
        to_repeat=True,
    )

    val_dataset, num_val = get_tfrec_dataset(
        val_files,
        shuffle=False,
        batch_size=config["val_batch_size"],
        config=config,
        cache=True,
        to_filter="filter_1",
        calculate_sample_num=True,
    )

    return (train_dataset, num_train), (val_dataset, num_val)


def main():

    (train_dataset, num_train), (val_dataset, num_val) = get_dataset(config)

    tf.keras.backend.clear_session()

    model = get_model(config=config, hidden_dim=192, max_len=config["X_max_len"])
    model.summary()

    # Learning rate for encoder
    LR_SCHEDULE = [
        lrfn(step, config=config, num_cycles=0.50)
        for step in range(config["num_training_steps"])
    ]
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda step: LR_SCHEDULE[step], verbose=0
    )

    if config["DEBUG"]:
        plot_lr_schedule(LR_SCHEDULE, epochs=config["num_training_steps"])

    save_folder = "/working"
    try:
        os.mkdir(f"{save_folder}/weights")
    except:
        pass

    class save_model_callback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()

        def on_epoch_end(self, epoch: int, logs=None):
            if epoch == 3 or (epoch + 1) % 25 == 0:
                self.model.save_weights(f"{save_folder}/weights/model_epoch_{epoch}.h5")

    steps_per_epoch = num_train // config["batch_size"]
    val_steps_per_epoch = num_val // config["val_batch_size"]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["num_training_steps"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        verbose=2,
        callbacks=[
            save_model_callback(),
            lr_callback,
        ],
    )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])


if __name__ == "__main__":
    main()
