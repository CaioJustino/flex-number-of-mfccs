# @file main.py
# @brief Arquivo principal para a execução do projeto - Dataset: RAVDESS.

# Imports
from .processing import carregarDataset
from .training import treinarModelo
from .utils import fixarSeeds

# @brief Executa o pipeline completo de treinamento.
# @param 'pasta_dataset' Caminho para a pasta raiz do dataset.
# @param 'n_mfcc' Número de coeficientes MFCC a serem extraídos.
# @param 'epochs' Número de épocas de treinamento.
# @param 'batch_size' Tamanho do batch.
# @param 'arquitetura' Nome da arquitetura ("CRNN", "CNN", "ResNet").
# @return Acurácia média final.
def rodar(pasta_dataset, n_mfcc, epochs, batch_size, arquitetura):
    fixarSeeds(42)
    print(f"\nExecutando: {arquitetura} | MFCC: {n_mfcc}")

    X, y, atores = carregarDataset(pasta=pasta_dataset, n_mfcc=n_mfcc)
    acc = treinarModelo(X=X, y=y, atores=atores, arquitetura=arquitetura, n_mfcc=n_mfcc, epochs=epochs, batch_size=batch_size)

    print(f"Acurácia Final: {arquitetura} | MFCC {n_mfcc}: {acc:.4f}")
    return acc

if __name__ == "__main__":
    DATASET = "data/raw/ravdess"
    epochs = 50
    BATCH_SIZE = 32

    rodar(pasta_dataset=DATASET, n_mfcc=13, epochs=epochs, batch_size=BATCH_SIZE, arquitetura="CRNN")