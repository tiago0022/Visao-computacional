import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

caminho_imagem = 'rsc/primavera.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]


n_colunas = 3
n_linhas = 3

global indice
indice = 0


def exibe(imagem, nome=''):
    global indice
    indice = indice + 1
    plt.subplot(n_linhas, n_colunas, indice), plt.imshow(imagem, cmap='gray')
    plt.title(nome), plt.xticks([]), plt.yticks([])


img = cv.imread(caminho_imagem)

lista_sp = [5, 12, 25]
lista_cr = [19, 24, 25]
lista_L = [3, 5, 7]

quant = len(lista_sp) + len(lista_cr) + len(lista_L)


def calcula_mean_shift(sp, cr, L):
    imagem_mean_shift = img.copy()
    imagem_mean_shift = cv.pyrMeanShiftFiltering(imagem_mean_shift, sp=sp, sr=cr, maxLevel=L)
    exibe(imagem_mean_shift, f'sp:{sp} cr:{cr} L:{L}')
    global indice
    print('Calculando... ', round((indice/quant) * 100), '%')


for sp in lista_sp:
    calcula_mean_shift(sp, lista_cr[1], lista_L[1])

for cr in lista_cr:
    calcula_mean_shift(lista_sp[1], cr, lista_L[1])

for L in lista_L:
    calcula_mean_shift(lista_sp[1], lista_cr[1], L)

plt.show()
