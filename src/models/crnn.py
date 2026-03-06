"""
    @file crnn.py
    @brief Arquivo voltado à implementação do modelo CRNN (CNN + GRU) - Ref.: https://github.com/ZainNasrullah/music-artist-classification-crnn.
"""

"""
    Imports
"""
from tensorflow.keras import layers, models

"""
    @brief Cria e compila um modelo CRNN para classificação.
    @param 'input_shape' Tupla representando o formato da entrada (altura, largura, canais).
    @param 'num_classes' Número de classes para classificação.
    @return Modelo CRNN compilado.
"""
def criarCRNN(input_shape, num_classes):
    nb_layers = 4
    nb_filters = [64, 128, 128, 128]
    kernel_size = (3, 3)
    activation = 'elu'
    pool_size = [(1,2), (1,2), (1,2), (1,2)]

    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=input_shape))
    model.add(layers.Conv2D(nb_filters[0], kernel_size, padding='same', activation=activation))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size[0]))
    model.add(layers.Dropout(0.1))

    for i in range(nb_layers - 1):
        model.add(layers.Conv2D(nb_filters[i+1], kernel_size, padding='same', activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=pool_size[i+1]))
        model.add(layers.Dropout(0.1))

    model.add(layers.Permute((2,1,3)))

    model.add(layers.Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))

    model.add(layers.GRU(32, return_sequences=True))
    model.add(layers.GRU(32))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model