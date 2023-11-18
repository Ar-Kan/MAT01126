from typing import Sequence

import cv2
import numpy as np

from utilidades import ImagemTipo, mostrar_imagem, Figura, FiguraTipo


def cor_para_cinza(imagem: ImagemTipo) -> ImagemTipo:
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def cinza_para_cor(imagem: ImagemTipo) -> ImagemTipo:
    return cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)


def mudar_resolucao(imagem: ImagemTipo, fator: int | float, pixelado: bool = False) -> ImagemTipo:
    # pixelated: cv2.INTER_NEAREST
    # smooth: cv2.INTER_AREA
    # linear: cv2.INTER_LINEAR

    return cv2.resize(imagem, (0, 0), fx=fator, fy=fator,
                      interpolation=cv2.INTER_LINEAR if not pixelado else cv2.INTER_NEAREST)


def encontra_contornos(imagem: ImagemTipo, limite: int) -> Sequence[np.ndarray]:
    blur = cv2.GaussianBlur(imagem, (5, 5), 0)
    thresh = cv2.threshold(blur, limite, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # mostrar_imagem(thresh)

    # Remove ruído com operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    abertura = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invertida = 255 - abertura
    # mostrar_imagem(invertida)

    contornos, _ = cv2.findContours(invertida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def corrigir_persepctiva(imagem: ImagemTipo) -> tuple[ImagemTipo, int]:
    im_h, im_w = imagem.shape[:2]
    area = im_h * im_w
    imagem_corrigida = None
    lim = 0
    img = imagem.copy()

    for lim in range(255, 0, -20):
        _, transformada = cv2.threshold(img, lim, 255, cv2.CHAIN_APPROX_NONE)
        contornos, _ = cv2.findContours(transformada, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # img1 = cv2.drawContours(img.copy(), contornos, -1, (0, 255, 0), 3)
        # mostrar_imagem(img1)
        for c in contornos:
            a = cv2.contourArea(c)
            if area * 0.5 > a or a > area * 0.9:
                continue
            if cv2.isContourConvex(c):
                continue
            if len(c) < 4:
                continue
            if cv2.arcLength(c, True) < 100:
                continue
            poligonos = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            if len(poligonos) == 4:
                *_, largura, altura = cv2.boundingRect(poligonos)
                pA, pB, pC, pD = poligonos
                wAD = int(np.linalg.norm(pA - pD))
                wBC = int(np.linalg.norm(pB - pC))
                hAB = int(np.linalg.norm(pA - pB))
                hCD = int(np.linalg.norm(pC - pD))
                max_w = max(wAD, wBC)
                max_h = max(hAB, hCD)
                M = cv2.getPerspectiveTransform(
                    np.array([pA, pB, pC, pD], dtype=np.float32),
                    np.array([
                        [0, 0],
                        [0, max_h - 1],
                        [max_w - 1, max_h - 1],
                        [max_w - 1, 0]
                    ], dtype=np.float32)
                )
                imagem_corrigida = cv2.warpPerspective(imagem, M, (max_w, max_h), flags=cv2.INTER_LINEAR)
                # im = cv2.drawContours(imagem.copy(), [poligonos], 0, (0, 0, 0), 5)
                # mostrar_imagem(im)
                return imagem_corrigida, lim
            # NOTA: alternativa para encontrar o retângulo
            #       se na imagem não houver contornos com 4 lados podemos criar um retângulo no contorno
            #       e comparar com o contorno original
            # else:
            #     box = np.int16(cv2.boxPoints(cv2.minAreaRect(c)))
            #     # match shapes
            #     match = cv2.matchShapes(box, poligonos, cv2.CONTOURS_MATCH_I1, 0.0)
            #     print(f'match: {match}')

    if imagem_corrigida is None:
        raise Exception('Não foi possível corrigir a perspectiva da imagem')


def encontrar_figuras(imagem: ImagemTipo, limite: int) -> list[Figura]:
    im_h, im_w = imagem.shape[:2]
    area = im_h * im_w
    imagem = imagem.copy()

    contornos = encontra_contornos(imagem, limite)
    figuras: list[Figura] = []
    for c in contornos:
        area_contorno = cv2.contourArea(c)
        area_relativa = area_contorno / area
        if not (10 < area_contorno < area * 0.2) or area_relativa < 0.0005:
            continue

        if len(c) > 4:
            # elipse
            eli_centro, (eli_largura, eli_altura), eli_angulo = cv2.fitEllipse(c)
            eli_area = np.pi * eli_largura * eli_altura / 4
            if 0.5 < eli_largura / eli_altura < 2:
                figuras.append(
                    Figura(
                        tipo=FiguraTipo.ELIPSE,
                        centro=eli_centro,
                        largura=eli_largura,
                        altura=eli_altura,
                        angulo=eli_angulo,
                        area=eli_area
                    )
                )
                continue

        # retangulo
        ret_centro, (ret_largura, ret_altura), ret_angulo = cv2.minAreaRect(c)
        box = cv2.boxPoints((ret_centro, (ret_largura, ret_altura), ret_angulo))
        box = np.int16(box)
        figuras.append(
            Figura(
                tipo=FiguraTipo.RETANGULO,
                centro=ret_centro,
                largura=ret_largura,
                altura=ret_altura,
                angulo=ret_angulo,
                area=ret_largura * ret_altura,
                box=box
            )
        )
    return figuras


def classifica_figuras(imagem: ImagemTipo, figuras: list[Figura]) -> ImagemTipo:
    def label(centro, texto: str):
        cv2.putText(imagem,
                    texto,
                    np.intp(centro),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)

    im_h, im_w = imagem.shape[:2]
    area = im_h * im_w

    imagem = cinza_para_cor(imagem)

    elipses = sorted(filter(lambda f: f.tipo == FiguraTipo.ELIPSE, figuras), key=lambda f: f.area, reverse=True)
    retangulos = filter(lambda f: f.tipo == FiguraTipo.RETANGULO, figuras)

    for i, elipse in enumerate(elipses):
        # remove elipses dentro de outras elipses
        for j in range(i + 1, len(elipses)):
            if elipses[j].dentro(*elipse.centro):
                # espera-se apenas uma elipse dentro de outra
                elipses.pop(j)
                break

    for elipse in elipses:
        # desenha elipse
        # cv2.ellipse(imagem, elipse.tuple(), (0, 0, 255), 2)
        # cv2.ellipse(mask, elipse.tuple(), (255, 255, 255), -1)
        area_relativa = np.round(elipse.area / area, 3)
        if area_relativa < 0.003:
            nome = "Porca"
        elif area_relativa < 0.005:
            nome = "Arr. Peq."
        else:
            nome = "Arr. Gra."
        label(elipse.centro, nome)

    for retangulo in retangulos:
        area_relativa = np.round(retangulo.area / area, 4)
        if area_relativa >= 0.002:
            nome = "3/4''"
        else:
            nome = "1/2''"
        label(retangulo.centro, nome)
        # desenha bbox
        # for i in range(4):
        #     cv2.line(imagem,
        #              tuple(retangulo.box[i]),
        #              tuple(retangulo.box[(i + 1) % 4]),
        #              (0, 0, 255),
        #              2)
    return imagem


def run(nome: str | ImagemTipo) -> None:
    if isinstance(nome, str):
        imagem = cv2.imread(nome)
    else:
        imagem = nome
    # mostrar_imagem(imagem)
    imagem = cor_para_cinza(imagem)
    # mostrar_imagem(imagem)
    imagem = cv2.GaussianBlur(imagem, (3, 3), sigmaX=0, sigmaY=0)
    imagem = mudar_resolucao(imagem, 0.5, False)
    imagem, limite = corrigir_persepctiva(imagem)
    figuras = encontrar_figuras(imagem, limite)
    imagem = classifica_figuras(imagem, figuras)
    mostrar_imagem(imagem)


if __name__ == '__main__':
    import os

    # run(f'./Parafuso/G1/imagem_G1_06.jpg')
    for i in os.listdir('./Parafuso/G1'):
        print(f'Processando {i}')
        run(f'./Parafuso/G1/{i}')
    for i in os.listdir('./Parafuso/G2'):
        print(f'Processando {i}')
        run(f'./Parafuso/G2/{i}')
