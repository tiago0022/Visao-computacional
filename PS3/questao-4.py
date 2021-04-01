import math
import random as rd
import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

n_colunas = 2
n_linhas = 1
global indice
indice = 0


def exibe(imagem, nome=''):
    global indice
    indice = indice + 1
    plt.subplot(n_linhas, n_colunas, indice), plt.imshow(imagem, cmap='gray')
    plt.title(nome), plt.xticks([]), plt.yticks([])


densidade = 0.05
numero_de_linhas = 4

alcance_linha = 2
densidade_linha = 0.4
tamanho_imagens = 175

for x in range(1, len(sys.argv), 2):
    if sys.argv[x] == '-d':
        densidade = float(sys.argv[x+1])
    if sys.argv[x] == '-n':
        numero_de_linhas = int(sys.argv[x+1])

imagem = np.zeros((tamanho_imagens, tamanho_imagens), dtype=np.uint8)

max_p = tamanho_imagens - 1
lista_linha = np.zeros((numero_de_linhas, 2, 2))
for i in range(numero_de_linhas):
    lista_linha[i][0] = np.array([rd.randint(0, max_p), rd.randint(0, max_p)])  # ponto P1
    lista_linha[i][1] = np.array([rd.randint(0, max_p), rd.randint(0, max_p)])  # ponto P2

for x in range(tamanho_imagens):
    for y in range(tamanho_imagens):

        esta_perto_linha = False
        for linha in lista_linha:
            p1 = linha[0]
            p2 = linha[1]
            p3 = np.array([x, y])
            distancia_linha = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
            if distancia_linha <= alcance_linha:
                esta_perto_linha = True
                break

        if esta_perto_linha:
            eh_branco = rd.random() > densidade_linha
        else:
            eh_branco = rd.random() > densidade

        imagem[x][y] = np.uint8(eh_branco)

    print('Gerando imagem... ', round((x/tamanho_imagens) * 100), '%')

limiar = 250

imagem_deteccao = imagem.copy()
imagem_exibicao = cv.cvtColor(imagem.copy() * 255, cv.COLOR_GRAY2BGR)

lista_linha = cv.HoughLinesP(imagem_deteccao, rho=1, theta=np.pi/180, threshold=limiar, minLineLength=100, maxLineGap=50)

for line in lista_linha:
    for x1, y1, x2, y2 in line:
        cv.line(imagem_exibicao, (x1, y1), (x2, y2), (0, 0, 255), 1)

exibe(imagem, 'Original')
exibe(imagem_exibicao, 'Line detection')

plt.show()
