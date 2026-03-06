"""
    @file resmet.py
    @brief Arquivo voltado à implementação do modelo ResNet - Ref.: https://github.com/tanjeffreyz/deep-residual-learning.
"""

"""
Imports
"""
from tensorflow.keras import layers, models

"""
    @brief Implementa um bloco residual.
    @param 'x' Tensor de entrada.
    @param 'filters' Número de filtros da convolução.
    @param 'downsample' Indica se deve reduzir a dimensão espacial (stride=2).
    @return Tensor resultante após aplicação do bloco residual.
"""
def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if downsample:
        shortcut = layers.Conv2D(filters, 1, strides=2, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

"""
    @brief Cria e compila um modelo ResNet simplificado.
    @param 'input_shape' Tupla representando o formato da entrada.
    @param 'num_classes' Número de classes para classificação.
    @return Modelo ResNet compilado.
"""
def criarResNet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 128, downsample=True)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model