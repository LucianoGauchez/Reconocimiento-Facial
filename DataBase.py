#-------------importamos librerias----------------
import cv2
import mediapipe as mp
import os
from mediapipe.python.solutions.drawing_utils import BLUE_COLOR, RED_COLOR

#Creacion de la carpeta donde almacenaremos las fotos
nombre = input('Ingrese su nombre: ')
direccion = '' #Direccion donde se guardaran las carpetas de los rostros
folder = direccion + '/' + nombre

if not os.path.exists(folder):
    print('carpeta creada')
    os.makedirs(folder)
#Inicializamos el contador
cont = 0
#---------------Declaracion del detector--------

detector = mp.solutions.face_detection #detector
plot = mp.solutions.drawing_utils #dibujo

#-------------------Realizar la videocaptura---------------
captura = cv2.VideoCapture(0)

#----------------Inicializamos parametros-----------
with detector.FaceDetection(min_detection_confidence = 0.75) as rostros: ##mayor a 0.75 para que sea detectado como rostro
    while True:
        #Realizamos la lectura de la videocaptura
        ret, frame = captura.read()
        
        #Eliminar error espejo
        frame = cv2.flip(frame, 1)
        
        #correccion de color 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Deteccion de rostros
        result = rostros.process(rgb)

        #Filtro de seguridad
        if result.detections is not None:
            for rostro in result.detections:
                #plot.draw_detection(frame, rostro, plot.DrawingSpec(color=RED_COLOR))

                for id, coordenadas in enumerate(result.detections):
                    #mostramos las coordenadas
                    #print('coordenadas ', coordenadas)
                    
                    #conversiond e coordenadas
                    al, an, c = frame.shape
                    
                    #extraer X inicial e Y inicial
                    xi = coordenadas.location_data.relative_bounding_box.xmin
                    yi = coordenadas.location_data.relative_bounding_box.ymin

                    #Extraer ancho y alto
                    ancho = coordenadas.location_data.relative_bounding_box.width
                    alto = coordenadas.location_data.relative_bounding_box.height

                    #conversion a pixeles
                    xi = int(xi * an)
                    yi = int(yi*al)
                    ancho = int(ancho * an)
                    alto = int(alto * al)

                    #Hallamos X final e Y final
                    xf = xi + ancho
                    yf = yi + alto
                   
                    #Extraccion de pixeles
                    cara = frame[yi:yf, xi:xf]

                    #redimencionar las fotos
                    cv2.resize(cara, (500, 750), interpolation=cv2.INTER_CUBIC)
                    
                    #almacenar nuestras imagenes
                    cv2.imwrite(folder + '/rostro_{}.jpg'.format(cont), cara)
                    cont = cont + 1
                    #Extraer el punto central de nuestro rostro
                    # cx = (xi + (xi + xf))//2
                    # cy = (yi + (yi + yf))//2
                    
                    #mostramos esa coordenada
                    #cv2.circle(frame, (cx,cy), 5, (255,100,255), cv2.FILLED)

        cv2.imshow('Reconocimiento Facial', frame) #en ret se almacena si la lectura de los fotogramas es correcta. en frame se almacenan los fotogramas
        
        #leemos la tecla esc en ascii 27
        t = cv2.waitKey(1) 
        if t == 27 or cont >= 500: #Podemos aumentar el numero de fotos que nos tome para asi mejorar el reconocimiento
            break

captura.release()
cv2.destroyAllWindows()