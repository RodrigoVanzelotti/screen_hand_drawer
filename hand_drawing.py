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

# Definir o tamanho inicial do pincel
thickness = 5

# loop para colar o header e pintar a tela
while True:
    '''
    1. import image
    2. achar os landmarks - usando o module
    3. Checar quais dedos estão levantados - para selecionar os necessários
    4. If eraser mode - dois dedos acima
        - Select, not draw
    5. Drawing mode - dedo principal acima
        - Free drawing
    6. Selecting size mode - três dedos pra cima
    '''
    # reset hands_on feature
    hands_on = False
    # 1. import image
    _, img = capture.read()
    img = cv2.flip(img, 1)  # invertendo a imagem para que o desenho seja natural, intuitivo
    
    # 2. achar os landmarks - usando o module
    img = Vanze.find_hands(img, draw_hands=True)
    landmark_list = Vanze.find_position(img, draw_hands=False)

    if len(landmark_list) != 0:
        hands_on = True
        #     print(landmark_list)
        x1, y1 = landmark_list[8][1:]       # dedo indicador - acessar imagem nos assets
        x2, y2 = landmark_list[12][1:]      # dedo do meio
        x3, y3 = landmark_list[16][1:]      # dedo anelar

    # 3. Checar quais dedos estão levantados - para selecionar os necessários
        fingers = Vanze.fingers_up()
        # print(fingers)

    # 5. Drawing mode
        if fingers[1] and not (fingers[2] or fingers[3]):
            print('drawing mode')
            img = Vanze.draw_in_position(img, [x1], [y1], (0, 0, 255), thickness)
            cv2.circle(img, (x1, y1), thickness, (0, 255, 0), -1)
            header = overlay_images[1]

    # 4. Eraser mode
        elif fingers[1] and fingers[2] and not fingers[3]:
            print('Eraser mode')
            img = Vanze.draw_in_position(img, [x1, x2], [y1, y2], (0, 0, 255), thickness)
            header = overlay_images[2]
    
    # 6. Selecting size mode
        elif fingers[1] and fingers[2] and fingers[3]:
            print('sizing mode')
            img = Vanze.draw_in_position(img, [x1, x2, x3], [y1, y2, y3], (255, 0, 0), thickness)
            cv2.circle(img, (x1, y1), thickness, (255, 0, 0), -1)
            header = overlay_images[3]
            
    # If none of those commands, return to main header
        else: hands_on = False

    if not hands_on:    
        header = overlay_images[0]

    # colando o header correto
    img[0:header.shape[0], 0:cam_width] = header

    cv2.imshow("Image", img)
    cv2.waitKey(1)



