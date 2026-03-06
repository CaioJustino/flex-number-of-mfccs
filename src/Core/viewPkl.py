"""
    Imports
"""
import pickle
import pprint

with open("resultados/CRNN_MFCC13/resultado_final.pkl", "rb") as f:
    dados = pickle.load(f)

print("Tipo:", type(dados))
pprint.pprint(dados)