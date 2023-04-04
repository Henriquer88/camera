# Projeto utilizando o Mediapipe junto com o opencv #
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#video  #

import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
hand = mp.solutions.hands
Hands =  hand.Hands(max_num_hands=2)  # Número de mãos utilizadas
mpDraw =  mp.solutions.drawing_utils   # Desenha os pontos das mãos

while True:
    check, img =  video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversão de BGR para RGB
    results =  Hands.process(imgRGB)
    handsPoints = results.multi_hand_landmarks # Extrai os pontos das coordenadas dos pontos

    if handsPoints : #  FOR só é executado se a varável não estiver vazia
        for points in handsPoints:  # vamos percorrer as coordenadas dos pontos
         print(points)
         mpDraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)

    cv2.imshow("Imagem",img)
    cv2.waitKey(1)


