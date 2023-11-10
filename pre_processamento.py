import cv2
import numpy as np
import matplotlib.pyplot as plt

from utilidades import ImagemTipo, mostrar_imagem


def para_escala_de_cinza(imagen: ImagemTipo) -> ImagemTipo:
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def mudar_resolucao(imagen: ImagemTipo, fator: int | float, pixelado: bool = False) -> ImagemTipo:
    # pixelated: cv2.INTER_NEAREST
    # smooth: cv2.INTER_AREA
    # linear: cv2.INTER_LINEAR

    return cv2.resize(imagen, (0, 0), fx=fator, fy=fator,
                      interpolation=cv2.INTER_LINEAR if not pixelado else cv2.INTER_NEAREST)


def normalizacao(imagen: ImagemTipo) -> ImagemTipo:
    pass


if __name__ == '__main__':
    imagem = cv2.imread('./Parafuso/G1/imagem_G1_01.jpg')
    # imagem = cv2.imread('./Parafuso/G2/imagem_G2_01.jpg')
    # mostrar_imagem(imagem)
    imagem = para_escala_de_cinza(imagem)
    # mostrar_imagem(imagem)
    imagem = mudar_resolucao(imagem, 0.1, True)
    imagem = mudar_resolucao(imagem, 3, True)
    # mostrar_imagem(imagem)
    scaled = np.round(imagem / 255, 2)
    scaled[scaled < 0.55] = 0
    # print(scaled)

    # plt.imshow(scaled, cmap='gray')
    # plt.show()
    mostrar_imagem(scaled)
