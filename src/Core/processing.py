# @file processing.py
# @brief Pré-processamento de áudio e extração de características MFCC.

# Imports
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

# Variáveis Principais
DATASET = "data/raw/ravdess/audio_speech_actors_01-24"
SR = 16000
DURACAO = 5
SAMPLES = SR * DURACAO
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

# @brief Carrega um arquivo de áudio do dataset.
# @param pasta_ator Caminho da pasta do ator.
# @param arquivo Nome do arquivo de áudio.
# @return Vetor contendo o sinal de áudio carregado.
def carregar_audio(pasta_ator, arquivo):
    caminho_arquivo = os.path.join(pasta_ator, arquivo)
    audio, _ = librosa.load(caminho_arquivo, sr=SR)
    return audio

# @brief Padroniza o tamanho do áudio para uma duração fixa.
# @param audio Vetor de áudio carregado.
# @return Vetor de áudio com tamanho padronizado.
def padronizar_audio(audio):
    return librosa.util.fix_length(audio, size=SAMPLES)

# @brief Extrai os coeficientes MFCC do áudio.
# @param audio Vetor de áudio padronizado.
# @param n_mfcc Número de coeficientes MFCC a serem extraídos.
# @return Matriz contendo os coeficientes MFCC.
def converter_audio(audio, n_mfcc):
    return librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

# @brief Normaliza os coeficientes MFCC utilizando CMVN.
# @param mfcc Matriz de coeficientes MFCC.
# @return Matriz MFCC normalizada com média zero e variância unitária.
def normalizar_mfcc(mfcc):
    scaler = StandardScaler()
    mfcc_normalizado = scaler.fit_transform(mfcc.T).T
    return mfcc_normalizado

# @brief Carrega o dataset RAVDESS e realiza o pré-processamento completo.
# @param pasta Caminho da pasta raiz do dataset.
# @param n_mfcc Número de coeficientes MFCC a serem extraídos.
# @return Tupla contendo:
# - X: matriz de características MFCC
# - y: vetor de rótulos de emoção
# - atores: vetor com identificação dos atores
def carregarDataset(pasta=DATASET, n_mfcc=13):
    X = []
    y = []
    atores = []

    for ator in np.sort(os.listdir(pasta)):
        pasta_ator = os.path.join(pasta, ator)

        if not os.path.isdir(pasta_ator):
            continue

        for arquivo in os.listdir(pasta_ator):
            if not arquivo.endswith(".wav") or ":Zone.Identifier" in arquivo:
                continue

            audio = carregar_audio(pasta_ator=pasta_ator, arquivo=arquivo)
            audio_padronizado = padronizar_audio(audio=audio)

            mfcc_bruto = converter_audio(audio_padronizado, n_mfcc=n_mfcc)

            mfcc_final = normalizar_mfcc(mfcc_bruto)

            emocao = int(arquivo.split("-")[2]) - 1

            X.append(mfcc_final)
            y.append(emocao)
            atores.append(ator)

    X = np.array(X)
    y = np.array(y)
    atores = np.array(atores)

    X = X[..., np.newaxis]

    return X, y, atores