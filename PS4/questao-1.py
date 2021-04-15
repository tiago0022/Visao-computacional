import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

caminho_video = 'rsc/transito.gif'

if len(sys.argv) >= 2:
    caminho_video = sys.argv[1]

n_colunas = 4
n_linhas = 3
global indice
indice = 0


def exibe(imagem, nome=''):
    global indice
    indice = indice + 1
    plt.subplot(n_linhas, n_colunas, indice), plt.imshow(imagem, cmap='gray')
    plt.title(nome), plt.xticks([]), plt.yticks([])


def calcula_Ix(x, y, I):

    t = 0

    termo1 = int(I[t][x+1][y])
    termo2 = int(I[t+1][x+1][y])
    termo3 = int(I[t][x+1][y+1])
    termo4 = int(I[t+1][x+1][y+1])
    soma1 = termo1 + termo2 + termo3 + termo4

    termo5 = int(I[t][x][y])
    termo6 = int(I[t+1][x][y])
    termo7 = int(I[t][x][y+1])
    termo8 = int(I[t+1][x][y+1])
    soma2 = termo5 + termo6 + termo7 + termo8

    return ((1/4) * soma1) - ((1/4) * soma2)


def calcula_Iy(x, y, I):

    t = 0

    termo1 = int(I[t][x][y+1])
    termo2 = int(I[t+1][x][y+1])
    termo3 = int(I[t][x+1][y+1])
    termo4 = int(I[t+1][x+1][y+1])
    soma1 = termo1 + termo2 + termo3 + termo4

    termo5 = int(I[t][x][y])
    termo6 = int(I[t+1][x][y])
    termo7 = int(I[t][x+1][y])
    termo8 = int(I[t+1][x+1][y])
    soma2 = termo5 + termo6 + termo7 + termo8

    return ((1/4) * soma1) - ((1/4) * soma2)


def calcula_It(x, y, I):

    t = 0

    termo1 = int(I[t+1][x][y])
    termo2 = int(I[t+1][x][y+1])
    termo3 = int(I[t+1][x+1][y])
    termo4 = int(I[t+1][x+1][y+1])
    soma1 = termo1 + termo2 + termo3 + termo4

    termo5 = int(I[t][x][y])
    termo6 = int(I[t][x][y+1])
    termo7 = int(I[t][x+1][y])
    termo8 = int(I[t][x+1][y+1])
    soma2 = termo5 + termo6 + termo7 + termo8

    return ((1/4) * soma1) - ((1/4) * soma2)


def calcula_Ix_Iy_It(x, y, I, aproximacao_Ixyt='Horn S.'):
    if aproximacao_Ixyt == 'Horn S.':
        Ix = calcula_Ix(x, y, I)
        Iy = calcula_Iy(x, y, I)
        It = calcula_It(x, y, I)
    if aproximacao_Ixyt == 'Sobel':
        Ix = (np.absolute(cv.Sobel(I[0], -1, 1, 0)) + np.absolute(cv.Sobel(I[1], -1, 1, 0)))[x][y]
        Iy = (np.absolute(cv.Sobel(I[0], -1, 0, 1)) + np.absolute(cv.Sobel(I[1], -1, 0, 1)))[x][y]
        It = (I[1]-I[0])[x][y]
    return Ix, Iy, It


def media_adjacencia(x, y, matriz):

    max_x = len(matriz) - 1
    max_y = len(matriz[0]) - 1

    cma = 0 if x == 0 else matriz[x-1][y]
    bxo = 0 if x == max_x else matriz[x+1][y]
    esq = 0 if y == 0 else matriz[x][y-1]
    dir = 0 if y == max_y else matriz[x][y+1]

    return np.mean([cma, bxo, esq, dir])


def calcula_alpha_e_medias(x, y, u, v, Ix, Iy, It, lmbd):

    u_media = media_adjacencia(x, y, u)
    v_media = media_adjacencia(x, y, v)

    numerador = (Ix * u_media) + (Iy * v_media) + It
    denominad = (lmbd ** 2) + (Ix ** 2) + (Iy ** 2)

    alpha = numerador / denominad

    return alpha, u_media, v_media


def calcula_Q(p1, p2):
    p3 = (0, 0)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    if det == 0:
        return (0, 0)
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return x1+a*dx, y1+a*dy


def aprox_sobel(I):
    fx = (np.absolute(cv.Sobel(I[0], -1, 1, 0)) + np.absolute(cv.Sobel(I[1], -1, 1, 0)))
    fy = (np.absolute(cv.Sobel(I[0], -1, 0, 1)) + np.absolute(cv.Sobel(I[1], -1, 0, 1)))
    ft = (I[1]-I[0])
    return fx, fy, ft


def aprox_horn_s(I):
    n_rows = len(I[0])
    n_cols = len(I[0][0])
    fx = np.zeros((n_rows, n_cols))
    fy = np.zeros((n_rows, n_cols))
    ft = np.zeros((n_rows, n_cols))
    for x in range(n_rows - 1):
        for y in range(n_cols - 1):
            fx[x][y], fy[x][y], ft[x][y] = calcula_Ix_Iy_It(x, y, I)
    return fx, fy, ft


def horn_s(I, inicializacao='0', aproximacao_Ixyt='Horn S.'):

    T = 7
    lmbd = 0.1

    n_rows = len(I[0])
    n_cols = len(I[0][0])

    u: np.ndarray = np.zeros((n_rows, n_cols))
    v: np.ndarray = np.zeros((n_rows, n_cols))

    msg_calc = f'Inicialização {inicializacao} / aproximacao {aproximacao_Ixyt}...'

    print(msg_calc, '0 %')

    if aproximacao_Ixyt == 'Sobel':
        fx, fy, ft = aprox_sobel(I)
    elif aproximacao_Ixyt == 'Horn S.':
        fx, fy, ft = aprox_horn_s(I)

    if inicializacao == 'Q':
        for x in range(n_rows - 1):
            for y in range(n_cols - 1):
                Ix, Iy, It = fx[x][y], fy[x][y], ft[x][y]
                if Iy != 0 and Ix != 0:
                    u[x][y], v[x][y] = calcula_Q((0, -It/Iy), (-It/Ix, 0))

    for n in range(1, T):

        print(msg_calc, round((n/T) * 100), '%')
        u_anterior = u.copy()
        v_anterior = v.copy()

        for x in range(n_rows - 1):
            for y in range(n_cols - 1):
                Ix, Iy, It = fx[x][y], fy[x][y], ft[x][y]
                alpha, u_media, v_media = calcula_alpha_e_medias(x, y, u, v, Ix, Iy, It, lmbd)
                u[x][y] = u_media - (alpha * Ix)
                v[x][y] = v_media - (alpha * Iy)

        media_diferenca_u = np.mean(np.abs(np.subtract(u, u_anterior).flatten()))
        media_diferenca_v = np.mean(np.abs(np.subtract(v, v_anterior).flatten()))

        print(f'--> iteração {n}: Variação em u: {round(media_diferenca_u, 2)}   | Variação em v: {round(media_diferenca_v, 2)}\n')

    print(msg_calc, '100 %')

    return np.uint8(u), np.uint8(v)


vid = cv.VideoCapture(caminho_video)

_, quadro_1 = vid.read()
quadro_1 = cv.cvtColor(quadro_1, cv.COLOR_BGR2GRAY)

_, quadro_2 = vid.read()
quadro_2 = cv.cvtColor(quadro_2, cv.COLOR_BGR2GRAY)

exibe(quadro_1, 'Quadro 1')
exibe(quadro_2, 'Quadro 2')

img_seq = np.array([quadro_1, quadro_2])

print('\n===================== PASSO 1/4 ====================\n')
u, v = horn_s(img_seq)
exibe(u, 'U inicialização 0 / aproximação Horn S.')
exibe(v, 'V inicialização 0 / aproximação Horn S.')

print('\n===================== PASSO 2/4 ====================\n')
u_Q, v_Q = horn_s(img_seq, inicializacao='Q')
exibe(u_Q, 'U inicialização Q / aproximação Horn S.')
exibe(v_Q, 'V inicialização Q / aproximação Horn S.')

print('\n===================== PASSO 3/4 ====================\n')
u_sobel, v_sobel = horn_s(img_seq, aproximacao_Ixyt='Sobel')
exibe(u_sobel, 'U inicialização 0 / aproximação Sobel')
exibe(v_sobel, 'V inicialização 0 / aproximação Sobel')

print('\n===================== PASSO 4/4 ====================\n')
u_sobel_Q, v_sobel_Q = horn_s(img_seq, inicializacao='Q', aproximacao_Ixyt='Sobel')
exibe(u_sobel_Q, 'U inicialização Q / aproximação Sobel')
exibe(v_sobel_Q, 'V inicialização Q / aproximação Sobel')

plt.show()
