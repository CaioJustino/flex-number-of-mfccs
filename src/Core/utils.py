# @file utils.py
# @brief Arquivo voltado às funções utilitárias adicionais - Dataset: RAVDESS.

# Imports
import numpy as np
import tensorflow as tf
import random
import os

"""
    @brief Fixa seeds para reprodutibilidade do experimento.
    @param 'seed' Valor inteiro para inicialização aleatória.
"""
def fixarSeeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)