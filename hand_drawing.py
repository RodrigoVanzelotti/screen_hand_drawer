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

# Functions ===============
def draw_and_return_coords(xa: int, 
                           ya: int, 
                           x: int, 
                           y: int):
    if xa == 0 and ya == 0:
        xa, ya = x, y
    # caso não seja, desenha a linha do P0 ao ponto atual
    else:
        cv2.line(img, (xa, ya), (x, y), draw_color, thickness)
        cv2.line(drawing_canvas, (xa, ya), (x, y), draw_color, thickness)

    # e se reinicia o processo
    xa, ya = x, y

    return xa, ya

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
x_anterior, y_anterior = 0, 0

# Definir o tamanho inicial do pincel
thickness = 5

# Definir a cor da pintura
draw_color = (9, 232, 225)

'''
Definindo um novo canva para registrar o desenho
para isso vamos usar uma matriz do numpy (height, width, channels[cores])
np.uint8 -> unsigned integer de 0 a 255 (2^8)
agora, ao invés de desenhar na imagem da camera, sobreporemos esse canva, retirando os 0
'''
drawing_canvas = np.zeros((cam_height, cam_width, 3), np.uint8) 

# loop para colar o header e pintar a tela
while True:
    '''
    1. import image
    2. achar os landmarks - usando o module
    3. Checar quais dedos estão levantados - para selecionar os necessários
    4. Drawing mode - dedo indicador acima
        - Free drawing
    5. If eraser mode - dois dedos acima
        - Select, not draw
    6. Selecting size mode - três dedos pra cima
    '''
    # 1. import image
    _, img = capture.read()
    img = cv2.flip(img, 1)  # invertendo a imagem para que o desenho seja natural, intuitivo
    
    # 2. achar os landmarks - usando o module
    img = Vanze.find_hands(img, draw_hands=True)
    landmark_list = Vanze.find_position(img, draw_hands=False)

    if len(landmark_list) != 0:
        #     print(landmark_list)
        x1, y1 = landmark_list[8][1:]       # dedo indicador - acessar imagem nos assets
        x2, y2 = landmark_list[12][1:]      # dedo do meio
        x3, y3 = landmark_list[16][1:]      # dedo anelar

    # 3. Checar quais dedos estão levantados - para selecionar os necessários
        fingers = Vanze.fingers_up()
        # print(fingers)

    # 4. Drawing mode
        if fingers[1] and not (fingers[2] or fingers[3]):
            print('drawing mode')
            img = Vanze.draw_in_position(img, [x1], [y1], (0, 0, 255), thickness)
            cv2.circle(img, (x1, y1), thickness, draw_color, -1)
            header = overlay_images[1]

            x_anterior, y_anterior = draw_and_return_coords(x_anterior, y_anterior, x1, y1)
            # # se for a primeira iteração, atribuimos o x_anterior e y_anterior ao x1, y1
            # if x_anterior == 0 and y_anterior == 0:
            #     x_anterior, y_anterior = x1, y1
            # # caso não seja, desenha a linha do P0 ao ponto atual
            # else:
            #     cv2.line(img, (x_anterior, y_anterior), (x1, y1), draw_color, thickness)
            #     cv2.line(drawing_canvas, (x_anterior, y_anterior), (x1, y1), draw_color, thickness)

            # # e se reinicia o processo
            # x_anterior, y_anterior = x1, y1

    # 5. Eraser mode
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
        else: header = overlay_images[0]

    # colando o header correto
    img[0:header.shape[0], 0:cam_width] = header

    cv2.imshow("Image", img)
    cv2.imshow("Drawing Canvas", drawing_canvas) # -> mostrar na aula
    cv2.waitKey(1)  

