"""
    @file processing.py
    @brief Arquivo voltado para lidar com o pré-processamento dos modelos antes do treinamento - Dataset: RAVDESS.
"""

"""
    Imports
"""
import librosa
import numpy as np
import os

"""
    Parâmetros padrões para o treinamento 
"""
SR = 16000
DURACAO = 3
SAMPLES = SR * DURACAO
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

"""
    @brief Ajusta o áudio para duração fixa definida por SAMPLES.
    @param 'audio' Vetor de áudio carregado.
    @return Áudio com tamanho padronizado.
"""
def padronizarAudio(audio):
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]

    else:
        pad = SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad))

    return audio

"""
    @brief Extrai coeficientes MFCC do sinal de áudio.
    @param 'audio' Vetor de áudio padronizado.
    @param 'n_mfcc' Número de coeficientes MFCC.
    @return Matriz MFCC (n_mfcc x tempo).
"""
def extrairMFCC(audio, n_mfcc):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    return mfcc

"""
    @brief Segmenta a matriz MFCC em blocos menores.
    @param 'mfcc' Matriz MFCC.
    @param 'tamanho' Tamanho da janela temporal.
    @return Lista de segmentos MFCC.
"""
def segmentarAudio(mfcc, tamanho=50):
    segmentos = []
    total = mfcc.shape[1]

    for i in range(0, total - tamanho + 1, tamanho):
        segmentos.append(mfcc[:, i:i+tamanho])

    return segmentos

"""
    @brief Carrega o dataset RAVDESS e extrai os segmentos MFCC.
    @param 'pasta' Caminho da pasta raiz do dataset.
    @param 'n_mfcc' Número de coeficientes MFCC.
    @return X (dados), y (rótulos), atores (identificação para GroupKFold).
"""
def carregarDataset(pasta, n_mfcc):
    X = []
    y = []
    atores = []

    for ator in os.listdir(pasta):
        pastaAtor = os.path.join(pasta, ator)

        if not os.path.isdir(pastaAtor):
            continue

        for arquivo in os.listdir(pastaAtor):
            if not arquivo.endswith(".wav") or ":Zone.Identifier" in arquivo:
                continue

            caminho = os.path.join(pastaAtor, arquivo)
            audio, _ = librosa.load(caminho, sr=SR)
            audio = padronizarAudio(audio)

            mfcc = extrairMFCC(audio, n_mfcc)
            segmentos = segmentarAudio(mfcc)

            emocao = int(arquivo.split("-")[2]) - 1

            for seg in segmentos:
                X.append(seg)
                y.append(emocao)
                atores.append(ator)

    X = np.array(X)
    y = np.array(y)
    atores = np.array(atores)

    X = X[..., np.newaxis]

    print(f"Total de Segmentos: {len(X)}")
    return X, y, atores