from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

ImagemTipo = cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] | np.ndarray


class ImagemGrupo(StrEnum):
    G1 = 'G1'
    G2 = 'G2'


class FiguraTipo(StrEnum):
    ELIPSE = 'elipse'
    RETANGULO = 'retangulo'


@dataclass
class Figura:
    tipo: FiguraTipo
    centro: Sequence[float]
    largura: int
    altura: int
    angulo: float
    area: float
    box: np.ndarray = None

    def tuple(self) -> tuple[Sequence[float], Sequence[int], float]:
        return self.centro, (self.largura, self.altura), self.angulo

    def dentro(self, x, y) -> bool:
        if self.tipo == FiguraTipo.ELIPSE:
            return (
                    (
                            (x - self.centro[0]) ** 2 / (self.largura / 2) ** 2
                            + (y - self.centro[1]) ** 2 / (self.altura / 2) ** 2
                    ) <= 1
            )
        else:
            return cv2.pointPolygonTest(self.box, (x, y), False) >= 0


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
