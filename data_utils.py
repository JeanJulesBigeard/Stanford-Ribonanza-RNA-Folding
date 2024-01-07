import tensorflow as tf
import numpy as np


def decode_tfrec(record_bytes):
    schema = {}
    schema["id"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["seq"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["dataset_name_2A3"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["dataset_name_DMS"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["reads_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reads_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["signal_to_noise_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["signal_to_noise_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["SN_filter_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["SN_filter_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_error_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_error_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    features = tf.io.parse_single_example(record_bytes, schema)

    sample_id = tf.sparse.to_dense(features["id"])
    seq = tf.sparse.to_dense(features["seq"])
    dataset_name_2A3 = tf.sparse.to_dense(features["dataset_name_2A3"])
    dataset_name_DMS = tf.sparse.to_dense(features["dataset_name_DMS"])
    reads_2A3 = tf.sparse.to_dense(features["reads_2A3"])
    reads_DMS = tf.sparse.to_dense(features["reads_DMS"])
    signal_to_noise_2A3 = tf.sparse.to_dense(features["signal_to_noise_2A3"])
    signal_to_noise_DMS = tf.sparse.to_dense(features["signal_to_noise_DMS"])
    SN_filter_2A3 = tf.sparse.to_dense(features["SN_filter_2A3"])
    SN_filter_DMS = tf.sparse.to_dense(features["SN_filter_DMS"])
    reactivity_2A3 = tf.sparse.to_dense(features["reactivity_2A3"])
    reactivity_DMS = tf.sparse.to_dense(features["reactivity_DMS"])
    reactivity_error_2A3 = tf.sparse.to_dense(features["reactivity_error_2A3"])
    reactivity_error_DMS = tf.sparse.to_dense(features["reactivity_error_DMS"])

    out = {}
    out["seq"] = seq
    out["SN_filter_2A3"] = SN_filter_2A3
    out["SN_filter_DMS"] = SN_filter_DMS
    out["reads_2A3"] = reads_2A3
    out["reads_DMS"] = reads_DMS
    out["signal_to_noise_2A3"] = signal_to_noise_2A3
    out["signal_to_noise_DMS"] = signal_to_noise_DMS
    out["reactivity_2A3"] = reactivity_2A3
    out["reactivity_DMS"] = reactivity_DMS
    return out


def f1():
    return True


def f2():
    return False


def filter_function_1(x):
    SN_filter_2A3 = x["SN_filter_2A3"]
    SN_filter_DMS = x["SN_filter_DMS"]
    return tf.cond(
        (SN_filter_2A3 == 1) and (SN_filter_DMS == 1), true_fn=f1, false_fn=f2
    )


def filter_function_2(x):
    reads_2A3 = x["reads_2A3"]
    reads_DMS = x["reads_DMS"]
    signal_to_noise_2A3 = x["signal_to_noise_2A3"]
    signal_to_noise_DMS = x["signal_to_noise_DMS"]
    cond = (reads_2A3 > 100 and signal_to_noise_2A3 > 0.60) or (
        reads_DMS > 100 and signal_to_noise_DMS > 0.60
    )
    return tf.cond(cond, true_fn=f1, false_fn=f2)


def nan_below_filter(x):
    reads_2A3 = x["reads_2A3"]
    reads_DMS = x["reads_DMS"]
    signal_to_noise_2A3 = x["signal_to_noise_2A3"]
    signal_to_noise_DMS = x["signal_to_noise_DMS"]
    reactivity_2A3 = x["reactivity_2A3"]
    reactivity_DMS = x["reactivity_DMS"]

    if reads_2A3 < 100 or signal_to_noise_2A3 < 0.60:
        reactivity_2A3 = np.nan + reactivity_2A3
    if reads_DMS < 100 or signal_to_noise_DMS < 0.60:
        reactivity_DMS = np.nan + reactivity_DMS

    x["reactivity_2A3"] = reactivity_2A3
    x["reactivity_DMS"] = reactivity_DMS
    return x


def concat_target(x):
    reactivity_2A3 = x["reactivity_2A3"]
    reactivity_DMS = x["reactivity_DMS"]
    target = tf.concat(
        [reactivity_2A3[..., tf.newaxis], reactivity_DMS[..., tf.newaxis]], axis=1
    )
    target = tf.clip_by_value(target, 0, 1)
    return x["seq"], target
