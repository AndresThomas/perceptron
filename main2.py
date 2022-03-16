import ctypes
import math
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication


class Perceptron(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("menu.ui", self)
        self.btn_aleatorio.clicked.connect(self.entrenamiento_aleatorio)
        self.btn_manual.clicked.connect(self.entrenamiento_manual)
        self.entradas = np.loadtxt('entradas.txt', dtype=int)
        self.salidas = np.loadtxt('salidas.txt', dtype=int)
        self.tasas_aprendizaje = []  # para almacenar las n tasas de aprendizaj
        self.w = []  # para almacenar los pesos
        self.norma = 0  # ||V||
        self.error = 0
        self.lista_ek1 = []  # almacena los errores obtenidos para la tasa de aprendizaje 1
        self.lista_ek2 = []  # almacena los errores obtenidos para la tasa de aprendizaje 2
        self.lista_ek3 = []  # almacena los errores obtenidos para la tasa de aprendizaje 3
        self.lista_ek4 = []  # almacena los errores obtenidos para la tasa de aprendizaje 4
        self.lista_ek5 = []  # almacena los errores obtenidos para la tasa de aprendizaje 5
        self.lista_pesos1 = []  # almacena los pesos obtenidos para la tasa de aprendizaje 1
        self.lista_pesos2 = []  # almacena los pesos obtenidos para la tasa de aprendizaje 2
        self.lista_pesos3 = []  # almacena los pesos obtenidos para la tasa de aprendizaje 3
        self.lista_pesos4 = []  # almacena los pesos obtenidos para la tasa de aprendizaje 4
        self.lista_pesos5 = []  # almacena los pesos obtenidos para la tasa de aprendizaje 5

    def generar_pesos_aleatorios(self):
        w = []
        for i in range(len(self.entradas)):
            w.append(random.uniform(-1, 1))
        return w  # el vector W con pesos generados aleatoriamente

    def generar_tasa_aprendizaje_aleatoria(self):
        aux = random.random()  # generamos una tasa aleatoria
        while aux == 0:  # que nunca sea 0
            aux = random.random()
        return aux

    def entrenar(self, w, tasa_aprendizaje):
        self.w = w
        # detener el proceso si la magnitud del error es menor a un umbral є, |e| < є
        # while(self.norma > self.tazas_aprendizaje[tasa_aprendizaje]):
        while(self.norma > self.error):
            # U_k
            u = np.dot(self.w, self.entradas)

            # Y^c(u) = FA(u)
            yc = []

            # 0 si u < 0, 1 si u >= 0
            for i in u:
                yc.append(0) if i < 0 else yc.append(1)

            #ek = y^d - y^c, ydeseada - ycalculada
            error = self.salidas - yc

            # equivalente a e^t * X
            etx = np.dot(error, self.entradas)
            aux = []
            for x in etx:
                # n * e^t* x
                aux.append(
                    self.tasas_aprendizaje[tasa_aprendizaje] * x)
            etx = aux

            aux = []
            # Wk+1 = Wk + n * e^t* x
            for i in range(0, len(w)):
                aux.append(w[i]+etx[i])
            # wk+1
            self.w = aux

            # para sacar la norma es la raiz cuadrada de la sumatoria de los cuadrados de la lista de errores
            # sumatoria de cuadrados
            temp = 0
            for e in error:
                temp += e**2
            self.norma = temp
            # raiz cuadrada ||V||
            self.norma = math.sqrt(self.norma)
            print(self.norma, ' norma')
            print(self.error, ' norma')
            print(u, ' u')
            print(w)
            print(self.salidas)
            print(yc, ' yc')
            print(error, ' error')
            print(etx, ' etx')
            print()

    def entrenamiento_aleatorio(self):
        self.error = float(self.error_field.text())
        self.tasas_aprendizaje = [
            self.generar_tasa_aprendizaje_aleatoria() for i in range(5)]
        # ciclo para utilizar todas las tazas de aprendizaje
        for tasa_aprendizaje in range(len(self.tasas_aprendizaje)):
            print('Iteracion: ', tasa_aprendizaje)
            self.norma = random.randint(1, 9)  # se reinicia la norma
            # se pasa wk
            self.entrenar(w=self.generar_pesos_aleatorios(),
                          tasa_aprendizaje=tasa_aprendizaje)

            if(tasa_aprendizaje == 0):
                self.lista_pesos1.append(self.w)
                self.lista_ek1.append(self.norma)
            if(tasa_aprendizaje == 1):
                self.lista_pesos2.append(self.w)
                self.lista_ek2.append(self.norma)
            if(tasa_aprendizaje == 2):
                self.lista_pesos3.append(self.w)
                self.lista_ek3.append(self.norma)
            if(tasa_aprendizaje == 3):
                self.lista_pesos4.append(self.w)
                self.lista_ek4.append(self.norma)
            if(tasa_aprendizaje == 4):
                self.lista_pesos5.append(self.w)
                self.lista_ek5.append(self.norma)
            # self.graficar()

    def entrenamiento_manual(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = Perceptron()
    GUI.show()
    sys.exit(app.exec_())
