import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

caminho_video = 'carro.gif'
if len(sys.argv) >= 2:
    caminho_video = sys.argv[1]
vid = cv.VideoCapture(caminho_video)


def normaliza(funcao_referencia, funcao_normalizada):

    media_f = np.average(funcao_referencia)
    media_g = np.average(funcao_normalizada)
    desvio_padrao_f = np.std(funcao_referencia)
    desvio_padrao_g = np.std(funcao_normalizada)

    alpha = (desvio_padrao_g / desvio_padrao_f) * media_f - media_g
    beta = desvio_padrao_f / desvio_padrao_g

    return np.array(beta * (funcao_normalizada + alpha))


def obtem_contraste(quadro):
    soma_contraste = 0
    for x in range(len(quadro)):
        for y in range(len(quadro[x])):
            valor_pixel = np.average(quadro[x][y])
            valor_adjacencia = np.average(quadro[max(x - 1, 0):x + 1,
                                                 max(y - 1, 0):y + 1])
            soma_contraste = soma_contraste + abs(valor_pixel - valor_adjacencia)
    return soma_contraste / (len(quadro) * len(quadro[0]))


lista_contraste = np.array([])
lista_media = np.array([])
lista_variancia = np.array([])

contem_quadro, quadro = vid.read()
iteracao = 0

while contem_quadro:

    lista_contraste = np.append(lista_contraste, obtem_contraste(quadro))
    lista_media = np.append(lista_media, np.average(quadro))
    lista_variancia = np.append(lista_variancia, np.var(quadro))

    contem_quadro, quadro = vid.read()
    iteracao = iteracao + 1

    print('Quadro', iteracao, 'analisado')

print('\n======== FIM DE ANÁLISE ========\n')

lista_contraste = normaliza(lista_media, lista_contraste)
lista_variancia = normaliza(lista_media, lista_variancia)

distancia_contraste_media = sum(abs(lista_contraste - lista_media)) / len(lista_contraste)
distancia_contraste_variancia = sum(abs(lista_contraste - lista_variancia)) / len(lista_contraste)
distancia_media_variancia = sum(abs(lista_media - lista_variancia)) / len(lista_media)

print('Diferença L_1 entre contraste e média:', round(distancia_contraste_media, 2))
print('Diferença L_1 entre contraste e variância:', round(distancia_contraste_variancia, 2))
print('Diferença L_1 entre média e variância:', round(distancia_media_variancia, 2))

print('\nVermelho: contraste / Amarelo: média / Azul: variância\n')

plt.plot(lista_contraste, 'r')
plt.plot(lista_media, 'y')
plt.plot(lista_variancia, 'b')
plt.show()
