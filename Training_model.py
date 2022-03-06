# importamos librerias
import cv2
import numpy as np
import os

#----------- importamos las fotos tomadas anteriormente------------------
direccion = ''    #Carpeta donde se guardan los rostros
lista = os.listdir(direccion)

etiquetas = []
rostros = []
cont = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir #Leemos las fotos de los rostros
    for filename in os.listdir(nombre):
        etiquetas.append(cont) #asignamos las etiquetas
        rostros.append(cv2.imread(nombre + '/' + filename, 0))
    
    cont = cont + 1


#Creamos el modelo

reconocimiento = cv2.face.LBPHFaceRecognizer_create()


#entrenamiento
reconocimiento.train(rostros, np.array(etiquetas))

#Guardamos el modelo
reconocimiento.write('Modelo_Entrenado.xml')

print('modelo creado')