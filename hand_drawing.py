import cv2
import mediapipe as mp
import numpy as np
import time
import os       # para acessar nossas imagens

from typing import Union

import hand_tracking_module as htm

# Tipagem =================
'''
Type Hints podem ser usadas por ferramentas como IDEs e verificadores de tipo para fornecer informações adicionais sobre o código.
    Por exemplo, uma IDE pode usar Type Hints para fornecer sugestões de código e verificar se você está usando o tipo correto de dados em uma variável.
Type Hints podem ser usadas para documentar o código. Isso pode ser útil se você estiver trabalhando em um projeto com várias pessoas, 
    pois permite que você especifique o tipo de dados que uma variável deve conter.
Type Hints podem ser usadas para verificar se o código está usando o tipo correto de dados em uma variável. Isso pode ser útil para encontrar 
    erros de digitação ou erros de lógica que podem ser difíceis de encontrar em tempo de execução.
Type Hints podem ser usadas para otimizar o código em tempo de execução. Por exemplo, se você usar Type Hints para especificar que uma variável contém um número inteiro, 
    o interpretador Python pode usar uma implementação mais rápida de operações matemáticas em vez de uma implementação genérica que funciona com qualquer tipo de dados.
'''

# =========================

# lendo todas files no nosso folder e criando uma lista de imagens que serão lidas pelo cv2
folder_path = "drawing_options"
files = os.listdir(folder_path)
overlay_images = []

# para cada imagem, leremos e apendamos na lista de imagens
for image_path in files:
    image = cv2.imread(os.path.join(folder_path, image_path))
    overlay_images.append(image)

# o header principal (inicial) vai ser a nossa primeira imagem, dado a maneira que gravamos o nome dos pngs
header = overlay_images[0]

# inicia-se a camera e coleta suas specs
capture = cv2.VideoCapture(0)
cam_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# hand_tracking class   
Vanze = htm.VanzeDetector(min_detec_confidence=0.85)    # minimizar erros

# loop para colar o header e pintar a tela
while True:
    '''
    1. import image
    2. achar os landmarks - usando o module
    3. Checar quais dedos estão levantados - para selecionar os necessários
    4. If selection mode - dois dedos acima
        - Select, not draw
    5. Drawing mode - dedo principal acima
        - Free drawing
    '''
    # 1. import image
    _, img = capture.read()
    img = cv2.flip(img, 1)  # invertendo a imagem para que o desenho seja natural, intuitivo
    
    # 2. achar os landmarks - usando o module
    img = Vanze.find_hands(img, draw_hands=False)
    landmark_list = Vanze.find_position(img, draw_hands=False)

    if len(landmark_list) != 0:
        #     print(landmark_list)
        x1, y1 = landmark_list[8][1:]       # dedo principal - acessar imagem nos assets
        x1, y1 = landmark_list[12][1:]      # dedo do meio

        

    # 3. Checar quais dedos estão levantados - para selecionar os necessários
    # 4. If selection mode
    # 5. Drawing mode

    # colando o header correto
    img[0:header.shape[0], 0:cam_width] = header

    cv2.imshow("Image", img)
    cv2.waitKey(1)


