import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

caminho_imagem = 'rsc/mulher.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(caminho_imagem, 0)

n_colunas = 3
n_linhas = 4

global indice
indice = 0


def exibe(imagem, nome=''):
    global indice
    indice = indice + 1
    plt.subplot(n_linhas, n_colunas, indice), plt.imshow(imagem, cmap='gray')
    plt.title(nome), plt.xticks([]), plt.yticks([])


lista_imagem_suavisada = [cv.blur(img, (3, 3))]
lista_imagem_residuo = [img - lista_imagem_suavisada[0]]

exibe(img, 'Original')

for i in range(1, 30):
    suavisada = cv.blur(lista_imagem_suavisada[i-1], (3, 3))
    lista_imagem_suavisada.append(suavisada)
    lista_imagem_residuo.append(img - suavisada)

exibe(lista_imagem_suavisada[29], 'Suavisada (30)')
exibe(lista_imagem_residuo[29], 'Resíduo (30)')

A = np.array([[0, 1]])

matriz_coocorrencia_original = np.zeros((256, 256))
lista_matriz_coocorrencia_suavisado = np.zeros((30, 256, 256))
lista_matriz_coocorrencia_residuo = np.zeros((30, 256, 256))

for pi in range(len(img)):
    for pj in range(len(img[pi])):

        adjacencia = A + [pi, pj]
        for a in adjacencia:

            ai = a[0]
            aj = a[1]

            if ai < 0 or ai == len(img) or aj < 0 or aj == len(img[pi]):
                continue

            u_original = img[pi][pj]
            v_original = img[ai][aj]
            matriz_coocorrencia_original[u_original][v_original] += 1

            for i in range(30):

                u_s = lista_imagem_suavisada[i][pi][pj]
                v_s = lista_imagem_suavisada[i][ai][aj]
                lista_matriz_coocorrencia_suavisado[i][u_s][v_s] += 1

                u_r = lista_imagem_residuo[i][pi][pj]
                v_r = lista_imagem_residuo[i][ai][aj]
                lista_matriz_coocorrencia_residuo[i][u_r][v_r] += 1

    print('Calculando... ', round((pi/len(img)) * 50, 2), '%')

soma_T = np.sum(matriz_coocorrencia_original)

homogeneidade_original = 0
uniformidade_original = 0

lista_homogeneidade_suavisada = np.zeros(30)
lista_uniformidade_suavisada = np.zeros(30)

lista_homogeneidade_residuo = np.zeros(30)
lista_uniformidade_residuo = np.zeros(30)

for u in range(256):
    for v in range(256):
        homogeneidade_original += (matriz_coocorrencia_original[u][v] / (1 + np.abs(u - v))) / soma_T
        uniformidade_original += (matriz_coocorrencia_original[u][v] ** 2) / soma_T
        for i in range(30):

            lista_homogeneidade_suavisada[i] += (lista_matriz_coocorrencia_suavisado[i][u][v] / (1 + np.abs(u - v))) / soma_T
            lista_uniformidade_suavisada[i] += (lista_matriz_coocorrencia_suavisado[i][u][v] ** 2) / soma_T

            lista_homogeneidade_residuo[i] += (lista_matriz_coocorrencia_residuo[i][u][v] / (1 + np.abs(u - v))) / soma_T
            lista_uniformidade_residuo[i] += (lista_matriz_coocorrencia_residuo[i][u][v] ** 2) / soma_T

    print('Calculando... ', round(50 + (u/256) * 50, 2), '%')

print('\nT =', soma_T)

print('\nHomogeneidade original:', homogeneidade_original)

print('\nUniformidade original:', uniformidade_original)

exibe(matriz_coocorrencia_original, 'M. Coocorrência original')
exibe(lista_matriz_coocorrencia_suavisado[29], 'M. Coocorrência suavisado (30)')
exibe(lista_matriz_coocorrencia_residuo[29], 'M. Coocorrência resíduo (30)')

indice += 1
plt.subplot(n_linhas, n_colunas, indice), plt.text(f'Homogeneidade original: {homogeneidade_original}')
indice += 1
plt.subplot(n_linhas, n_colunas, indice), plt.plot(lista_homogeneidade_suavisada), plt.title('Homogeneidade suavisadas')
indice += 1
plt.subplot(n_linhas, n_colunas, indice), plt.plot(lista_homogeneidade_residuo), plt.title('Homogeneidade resíduos')
indice += 1
plt.subplot(n_linhas, n_colunas, indice), plt.text(f'Uniformidade original: {uniformidade_original}')
indice += 1
plt.subplot(n_linhas, n_colunas, indice), plt.plot(lista_uniformidade_suavisada), plt.title('Uniformidade suavisadas')
indice += 1
plt.subplot(n_linhas, n_colunas, indice), plt.plot(lista_uniformidade_residuo), plt.title('Uniformidade resíduos')

plt.show()
