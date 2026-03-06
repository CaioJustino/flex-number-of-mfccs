# @file training.py
# @brief Arquivo voltado para o treinamento dos modelos.

# Imports
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping
from ..models.crnn import criarCRNN
from ..models.cnn import criarCNN
from ..models.resnet import criarResNet

# Configuração de Otimização 1: Otimizar o Tempo de Treino.
tf.config.optimizer.set_jit(True)


# @brief Realiza o treinamento do modelo utilizando validação cruzada por atores.
# @param 'X' Matriz de entrada (amostras, altura, largura, canais).
# @param 'y' Vetor de rótulos.
# @param 'atores' Identificação do ator para GroupKFold.
# @param 'arquitetura' Arquitetura a ser utilizada.
# @param 'n_mfcc' Número de coeficientes MFCC utilizados.
# @param 'n_splits' Número de folds para validação cruzada.
# @param 'epochs' Número de épocas de treinamento.
# @param 'batch_size' Tamanho do batch.
# @param 'verbose' Nível de verbosidade do treinamento.
# @return Acurácia média dos folds.
def treinarModelo(X, y, atores, arquitetura="CRNN", n_mfcc=13, n_splits=5, epochs=50, batch_size=32, verbose=1):
    pasta_saida = f"resultados/{arquitetura}_MFCC{n_mfcc}"
    os.makedirs(pasta_saida, exist_ok=True)

    gkf = GroupKFold(n_splits=n_splits)
    acc_folds = []
    historicos = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=atores)):
        print(f"\nFold {fold+1}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        input_shape = X_train.shape[1:]

        if arquitetura == "CRNN":
            model = criarCRNN(input_shape, num_classes=8)

        elif arquitetura == "CNN":
            model = criarCNN(input_shape)

        elif arquitetura == "ResNet":
            model = criarResNet(input_shape, num_classes=8)

        else:
            raise ValueError("Arquitetura inválida")

        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE))

        test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )

        history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, verbose=verbose, callbacks=[early_stop], workers=os.cpu_count(), use_multiprocessing=True)

        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f"Fold: {fold+1} | Accuracy: {acc:.4f}")

        acc_folds.append(acc)
        historicos.append(history.history)

    final_acc = acc_folds
    std_acc = np.std(acc_folds)

    resultados_finais = {"arquitetura": arquitetura, "n_mfcc": n_mfcc, "acc_por_fold": acc_folds, "final_acc": final_acc, "std_acc": std_acc}

    with open(os.path.join(pasta_saida, "resultado_final.pkl"), "wb") as f:
        pickle.dump(resultados_finais, f)

    print("\nRESULTADO FINAL")
    print(f"Acurácia Média: {final_acc:.4f}")
    print(f"Desvio Padrão : {std_acc:.4f}")

    return final_acc