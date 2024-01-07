import tensorflow as tf
from data_utils import *


def get_tfrec_dataset(
    tffiles,
    shuffle,
    batch_size,
    config,
    cache=False,
    to_filter=False,
    calculate_sample_num=True,
    to_repeat=False,
):
    ds = tf.data.TFRecordDataset(
        tffiles, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP"
    ).prefetch(tf.data.AUTOTUNE)

    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    if to_filter == "filter_1":
        ds = ds.filter(filter_function_1)
    elif to_filter == "filter_2":
        ds = ds.filter(filter_function_2)
    ds = ds.map(nan_below_filter, tf.data.AUTOTUNE)
    ds = ds.map(concat_target, tf.data.AUTOTUNE)

    if config["DEBUG"]:
        ds = ds.take(8)

    if cache:
        ds = ds.cache()

    samples_num = 0
    if calculate_sample_num:
        samples_num = ds.reduce(0, lambda x, _: x + 1).numpy()

    if shuffle:
        if shuffle == -1:
            ds = ds.shuffle(samples_num, reshuffle_each_iteration=True)
        else:
            ds = ds.shuffle(shuffle, reshuffle_each_iteration=True)

    if to_repeat:
        ds = ds.repeat()

    if batch_size:
        ds = ds.padded_batch(
            batch_size,
            padding_values=(config["PAD_x"], config["PAD_y"]),
            padded_shapes=([config["X_max_len"]], [config["X_max_len"], 2]),
            drop_remainder=True,
        )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, samples_num
