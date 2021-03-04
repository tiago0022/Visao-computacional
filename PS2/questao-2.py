import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

caminho_imagem = 'rsc/objetos.png'

if len(sys.argv) >= 2:
    caminho_imagem = sys.argv[1]

img = cv.imread(caminho_imagem, 0)

imagem_filtro_sigma = cv.bilateralFilter(img, 1, 100, 100)

laplacian = cv.Laplacian(imagem_filtro_sigma, cv.CV_64F, ksize=5)
sobelx = cv.Sobel(imagem_filtro_sigma, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(imagem_filtro_sigma, cv.CV_64F, 0, 1, ksize=5)

plt.subplot(3, 2, 1), plt.imshow(imagem_filtro_sigma, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

combinacao = sobely + sobelx + laplacian

for i in range(len(combinacao)):
    for j in range(len(combinacao[i])):
        if combinacao[i][j] < 0:
            combinacao[i][j] = 0

media = np.average(combinacao)

for i in range(len(combinacao)):
    for j in range(len(combinacao[i])):
        if combinacao[i][j] > media * 1.9:
            combinacao[i][j] = 255
        else:
            combinacao[i][j] = 0

plt.subplot(3, 2, 5)
plt.imshow(combinacao, cmap='gray')
plt.title('Combinação'), plt.xticks([]), plt.yticks([])

plt.show()
