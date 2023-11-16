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
    # mostrar_imagem(invert)

    contornos, _ = cv2.findContours(invertida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def corrigir_perspectiva_e_dimensoes(imagem: ImagemTipo) -> tuple[ImagemTipo, int]:
    im_h, im_w = imagem.shape[:2]
    area = im_h * im_w
    imagem_corrigida = None
    transformada = None
    lim = 255
    img = imagem.copy()

    # corrigir perspectiva
    for lim in range(255, 0, -20):
        _, transformada = cv2.threshold(img, lim, 255, cv2.CHAIN_APPROX_NONE)
        contornos, _ = cv2.findContours(transformada, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # img = cv2.drawContours(img, contornos, -1, (0, 255, 0), 3)
        # mostrar_imagem(img)
        for c in contornos:
            poligonos = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            if len(poligonos) == 4 and area * 0.8 > cv2.contourArea(c) > area * 0.5:
                *_, largura, altura = cv2.boundingRect(poligonos)
                M = cv2.getPerspectiveTransform(
                    poligonos.astype(np.float32),
                    np.array([[0, 0], [0, altura], [largura, altura], [largura, 0]], dtype=np.float32)
                )
                imagem_corrigida = cv2.warpPerspective(imagem, M, (largura, altura))
                # cv2.drawContours(imagem, [poligonos], 0, (0, 0, 0), 5)

    if imagem_corrigida is None:
        raise Exception('Não foi possível corrigir a perspectiva da imagem')

    # corrigir dimensões
    fator = 1
    # mostrar_imagem(imagem_corrigida)
    im_h, im_w = imagem_corrigida.shape[:2]
    while True:
        imagem = cv2.resize(imagem_corrigida.copy(), (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        # mostrar_imagem(imagem)
        # _, transformada = cv2.threshold(imagem, lim, 255, cv2.CHAIN_APPROX_NONE)
        # contornos, _ = cv2.findContours(transformada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contornos = encontra_contornos(imagem, lim)
        # cv2.drawContours(imagem, contornos, -1, (0, 255, 0), 3)
        # mostrar_imagem(imagem)

        # encontra elipses
        elipses: list[Figura] = []
        for c in contornos:
            area_contorno = cv2.contourArea(c)
            if not (10 < area_contorno < area * 0.4):
                continue

            el = cv2.fitEllipse(c)
            eli_centro, (eli_largura, eli_altura), eli_angulo = el
            if 0.5 < eli_largura / eli_altura < 2:
                elipses.append(
                    Figura(
                        tipo=FiguraTipo.ELIPSE,
                        centro=eli_centro,
                        largura=eli_largura,
                        altura=eli_altura,
                        angulo=eli_angulo,
                        area=np.pi * eli_largura * eli_altura / 4
                    )
                )
        elipses = sorted(
            filter(lambda f: f.tipo == FiguraTipo.ELIPSE, elipses),
            key=lambda f: f.area,
            reverse=True
        )
        if len(elipses) == 0:
            raise Exception('Não foi possível corrigir as dimensões da imagem')
        elipse = elipses[0]
        aspect_ratio = elipse.largura / elipse.altura
        if 0.9 < aspect_ratio < 1.1:
            return imagem, lim
        if aspect_ratio < 1:
            fator -= 0.1
        else:
            fator += 0.1
        im_w = int(im_w * fator)


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

        else:
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


if __name__ == '__main__':
    # imagem = cv2.imread('./Parafuso/G1/imagem_G1_01.jpg')  # peças regulares
    # imagem = cv2.imread('./Parafuso/G2/imagem_G2_01.jpg')  # pouca luminosidade
    imagem = cv2.imread('./Parafuso/G2/imagem_G2_06.jpg')  # parafuso torto
    # mostrar_imagem(imagem)
    imagem = cor_para_cinza(imagem)
    # mostrar_imagem(imagem)
    imagem = cv2.GaussianBlur(imagem, (3, 3), sigmaX=0, sigmaY=0)
    imagem = mudar_resolucao(imagem, 0.5, False)
    imagem, limite = corrigir_perspectiva_e_dimensoes(imagem)
    figuras = encontrar_figuras(imagem, limite)
    imagem = classifica_figuras(imagem, figuras)
    mostrar_imagem(imagem)
