"""Reusable data, model, and explainability helpers for the MRI notebook."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf


def build_df(path):
    """Build a labeled image table from class subdirectories."""
    records = [
        (label, os.path.join(path, label, image_name))
        for label in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, label))
        for image_name in sorted(os.listdir(os.path.join(path, label)))
    ]
    return pd.DataFrame(records, columns=['Class', 'Class Path'])[
        ['Class Path', 'Class']
    ]


def load_all_data(training_dir, testing_dir):
    """Load training, testing, and combined image metadata tables."""
    missing_dirs = [
        directory for directory in (training_dir, testing_dir)
        if not os.path.isdir(directory)
    ]
    if missing_dirs:
        raise FileNotFoundError(
            'MRI dataset folders were not found: '
            f'{missing_dirs}. Set MRI_DATA_DIR to the folder containing '
            'Training and Testing subfolders.'
        )
    training_data = build_df(training_dir)
    testing_data = build_df(testing_dir)
    all_data = pd.concat([training_data, testing_data], ignore_index=True)
    return training_data, testing_data, all_data


def make_augmentation(name='training_augmentation'):
    """Create the shared spatial augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.05, fill_mode='constant', fill_value=0.0),
        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10,
            fill_mode='constant',
            fill_value=0.0
        )
    ], name=name)


def build_dataset(
    images,
    labels,
    preprocess_fn,
    augment=False,
    batch_size=8,
    shuffle=False,
    seed=42,
    augmentation=None
):
    """Build a normalized, optionally augmented TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    if augment:
        augmentation = augmentation or make_augmentation()

    def prepare(image, label):
        image = tf.cast(image, tf.float32)
        if augment:
            image = augmentation(image, training=True)
        return preprocess_fn(image), label

    return dataset.map(
        prepare, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_efficientnet(num_classes):
    """Build the frozen-backbone EfficientNetB0 classifier."""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='efficientnet_transfer')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model


def build_gradcam_model(model, base_model):
    """Build the reusable Grad-CAM graph and its classifier metadata."""
    classifier_layers = [layer for layer in model.layers if layer.name != base_model.name]
    conv_layer_names = [
        layer.name for layer in base_model.layers
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
        and len(layer.output.shape) == 4
    ]
    target_layer_name = next(
        (name for name in ['top_conv', 'top_activation'] if name in conv_layer_names),
        conv_layer_names[-1]
    )
    target_layer = base_model.get_layer(target_layer_name)
    grad_model = tf.keras.models.Model(
        base_model.inputs,
        [target_layer.output, base_model.output]
    )
    return grad_model, classifier_layers, target_layer_name, conv_layer_names


def make_gradcam_heatmap(
    model,
    base_model,
    inputs,
    class_index,
    grad_model=None,
    classifier_layers=None
):
    """Compute a Grad-CAM heatmap using a reusable graph when provided."""
    if grad_model is None or classifier_layers is None:
        grad_model, classifier_layers, _, _ = build_gradcam_model(model, base_model)
    image_batch = tf.expand_dims(inputs, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(image_batch, training=False)
        tape.watch(conv_outputs)
        predictions = features
        for layer in classifier_layers:
            predictions = layer(predictions, training=False)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv_outputs * channel_weights[:, tf.newaxis, tf.newaxis, :], axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)[0]
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return tf.image.resize(heatmap[..., tf.newaxis], (224, 224))[..., 0].numpy()