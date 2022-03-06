#importamos librerias
import cv2
import os
import mediapipe as mp

cont = 0

#----------------------importamos los nombres de las carpetas-------------------
direccion = ''#Direccion donde se encuentran las carpetas de los rostros
etiquetas = os.listdir(direccion)
print('Nombres ', etiquetas)

#------------------------llamar el modelo entrenado
modelo = cv2.face.LBPHFaceRecognizer_create()

#-----------------------Leer modelo-----------------
modelo.read('Modelo_Entrenado.xml')

#---------------Declaracion del detector--------

detector = mp.solutions.face_detection #detector
plot = mp.solutions.drawing_utils #dibujo

#-------------------Realizar la videocaptura---------------
captura = cv2.VideoCapture(0)

#----------------Inicializamos parametros-----------
with detector.FaceDetection(min_detection_confidence = 0.85) as rostros: ##mayor a 0.75 para que sea detectado como rostro
    while True:
        #Realizamos la lectura de la videocaptura
        ret, frame = captura.read()
        copia = frame.copy()
        #Eliminar error espejo
        frame = cv2.flip(copia, 1)
        
        #correccion de color 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia_color = rgb.copy()
        #Deteccion de rostros
        result = rostros.process(copia_color)

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
                    cara = copia_color[yi:yf, xi:xf]

                    #redimencionar las fotos
                    cara = cv2.resize(cara, (150, 200), interpolation=cv2.INTER_CUBIC)
                    cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
                    #Realizar la prediccion
                    prediccion = modelo.predict(cara)

                    #Mostrar resultados en pantalla
                    if prediccion[0] == 0:
                        cv2.putText(frame, '{}'.format(etiquetas[0]), (xi, yi - 5), 1, 1.3, (0,0,250), 1, cv2.LINE_8)
                        cv2.rectangle(frame, (xi, yi), (xf, yf), (0,0,100), 2)
                    elif prediccion[0] == 1:
                        cv2.putText(frame, '{}'.format(etiquetas[1]), (xi, yi - 5), 1, 1.3, (250,0,0), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xi, yi), (xf, yf), (100,0,0), 2)

        cv2.imshow('Reconocimiento Facial', frame) #en ret se almacena si la lectura de los fotogramas es correcta. en frame se almacenan los fotogramas
        
        #leemos la tecla esc en ascii 27
        t = cv2.waitKey(1) 
        if t == 27:
            break

captura.release()
cv2.destroyAllWindows()