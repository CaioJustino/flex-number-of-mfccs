"""
    @file cnn.py
    @brief Arquivo voltado à implementação do modelo CNN - Ref.: https://github.com/ZainNasrullah/music-artist-classification-crnn.
"""

"""
    Imports
"""
from tensorflow.keras import layers, models

"""
    @brief Cria e compila um modelo CNN.
    @param 'input_shape' Tupla representando o formato da entrada (altura, largura, canais).
    @return Modelo CNN compilado.
"""
def criarCNN(input_shape):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='elu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='elu'))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model