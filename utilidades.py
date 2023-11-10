from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

ImagemTipo = cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] | np.ndarray


class ImagemGrupo(StrEnum):
    G1 = 'G1'
    G2 = 'G2'


def mostrar_imagem(img: ImagemTipo):
    cv2.imshow('Imagem', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ler_imagem(caminho: str) -> ImagemTipo:
    return cv2.imread(caminho)


def iterar_imagens(grupo: ImagemGrupo = None) -> Iterator[ImagemTipo]:
    if not grupo:
        caminho = './Parafuso/'
    else:
        caminho = f'./Parafuso/{grupo}/'
    itens = Path(caminho).glob('**/*.jpg')
    for item in itens:
        yield ler_imagem(str(item))
