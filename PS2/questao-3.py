import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

caminho_imagem = 'rsc/rosto1.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]


def normaliza(vetor):
    min = np.min(vetor)
    max = np.max(vetor)
    for i in range(len(vetor)):
        for j in range(len(vetor)):
            u = vetor[i][j]
            vetor[i][j] = int(((u/max) * 255) - min)


img = cv.imread(caminho_imagem, cv.COLOR_BGR2GRAY)

tamanho = 201
termo = int(tamanho/2)
x = 300
y = 300
x0, x1, y0, y1 = map(lambda t: max(0, t), [x - termo, x + termo, y - termo, y + termo])
janela = cv.cvtColor(img[y0:y1, x0:x1], cv.COLOR_BGR2GRAY)

fshift_1 = np.fft.fftshift(np.fft.fft2(janela))
fase = np.angle(fshift_1)
amplitude = 20*np.log(np.abs(fshift_1))

janela_filtro_sigma = cv.bilateralFilter(janela, 1, 100, 100)
bordas = cv.Laplacian(janela_filtro_sigma, cv.COLOR_BGR2GRAY, ksize=5)

T = 100

normaliza(fase)
normaliza(amplitude)

contador_fase = 0
contador_amplitude = 0
contador_bordas = 0

for i in range(tamanho - 1):
    for j in range(tamanho - 1):
        if fase[i][j] > T:
            contador_fase = contador_fase + 1
        if amplitude[i][j] > T:
            contador_amplitude = contador_amplitude + 1
        if bordas[i][j] > T:
            contador_bordas = contador_bordas + 1
        fase[i][j] = 1 if fase[i][j] > T else 0
        amplitude[i][j] = 1 if amplitude[i][j] > T else 0
        bordas[i][j] = 1 if bordas[i][j] > T else 0


plt.subplot(2, 2, 1), plt.imshow(janela, cmap='gray')
plt.title('Janela original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(fase, cmap='gray')
plt.title(f'Fase ({contador_fase} pixels > T)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(amplitude, cmap='gray')
plt.title(f'Magnitude ({contador_amplitude} pixels > T)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(bordas, cmap='gray')
plt.title(f'Bordas ({contador_bordas} pixels > T)'), plt.xticks([]), plt.yticks([])

plt.show()
