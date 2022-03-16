import ctypes
import math
import random
import numpy as np
import sys
# import dl2
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
        self.tazas_aprendizaje = []
        self.w = []  # array de pesos
        self.norma = 0
        self.error = 0
        self.lista_ek = []
        self.lista_ek1 = []
        self.lista_ek2 = []
        self.lista_ek3 = []
        self.lista_ek4 = []
        self.lista_ek5 = []
        self.lista_pesos1 = []
        self.lista_pesos2 = []
        self.lista_pesos3 = []
        self.lista_pesos4 = []
        self.lista_pesos5 = []

    def reset(self):
        self.entradas = np.loadtxt('entradas.txt', dtype=int)
        self.salidas = np.loadtxt('salidas.txt', dtype=int)
        self.tazas_aprendizaje = []
        self.w = []  # array de pesos
        self.norma = 0
        self.error = 0
        self.lista_ek = []
        self.lista_ek1 = []
        self.lista_ek2 = []
        self.lista_ek3 = []
        self.lista_ek4 = []
        self.lista_ek5 = []
        self.lista_pesos1 = []
        self.lista_pesos2 = []
        self.lista_pesos3 = []
        self.lista_pesos4 = []
        self.lista_pesos5 = []
    def generar_pesos_aleatorios(self):
        w = []
        
        for i in range(len(self.entradas)):
            w.append(random.uniform(-1, 1))
        return w  # el vector W con pesos generados aleatoriamente
    
    def lista_pesos(self):
        columns=('w0','w1','w2','w3','w4')
        rows =['η1','η2','η3','η4','η5']
        cell_text = [self.lista_pesos1[-1],self.lista_pesos2[-1],self.lista_pesos3[-1],self.lista_pesos4[-1],self.lista_pesos5[-1]]
        plt.table(cellText=cell_text,colLabels=columns,rowLabels=rows)
        plt.show()


    def graficar(self):        
        plt.plot(self.lista_ek1,label=' n1(eta):'+str(self.tazas_aprendizaje[0]),marker='D', linestyle='dashed')
        plt.plot(self.lista_ek2,label=' n2(eta):'+str(self.tazas_aprendizaje[1]),marker='o', linestyle='dashed')
        plt.plot(self.lista_ek3,label=' n3(eta):'+str(self.tazas_aprendizaje[2]),marker='>', linestyle='dashed')
        plt.plot(self.lista_ek4,label=' n4(eta):'+str(self.tazas_aprendizaje[3]),marker='p', linestyle='dashed')
        plt.plot(self.lista_ek5,label=' n5(eta):'+str(self.tazas_aprendizaje[4]),marker='h', linestyle='dashed')
        plt.xlabel('Iteraciones')
        plt.ylabel('Tamano del error')
        plt.legend()
        plt.show()
        self.lista_pesos()
        

    def entrenamiento_aleatorio(self):
        self.reset()
    
        self.error = float(self.error_field.text())
        for i in range(5):
            a = random.random()
            while(a == 0):
                a = random.random()
            # tazas de aprendizaje generadas de manera aleatoria
            self.tazas_aprendizaje.append(a)
        # ciclo para utilizar todas las tazas de aprendizaje
        for taza_aprendizaje in range(len(self.tazas_aprendizaje)):
            # con cada taza nueva los pesos deben ser generados, si no al cambiar de taza
            # queda con 1 iteracion
            w = self.generar_pesos_aleatorios()
            self.norma = 3
        # detener el proceso si la magnitud del error es menor a un umbral є, |e| < є
            while(self.norma > self.error):
                # para que este funcione deben coincidir en dimension por ejemplo
                # 4 columnas en self.entradas con 4 columnas del w
                # U_k 
                u = np.dot(w,self.entradas)

                # Y^c(u) = FA(u)
                yc = []
                # 0 si u < 0, 1 si u >= 0
                print(u,' u')
                for i in u:
                    yc.append(1) if i >= 0 else yc.append(0)

                #ek = y^d - y^c, ydeseada - ycalculada
                print(self.salidas,' ***')
                print(yc,'**{f')
                error = self.salidas - yc
                # transpose no tiene efecto con unidimensionales
                # equivalente a e^t * X
                etx = np.dot(error, self.entradas)
                aux = []

                for x in etx:
                    # n * e^t* x
                    aux.append(
                        self.tazas_aprendizaje[taza_aprendizaje] * x)
                etx = aux

                aux = []
                # Wk+1 = Wk + n * e^t* x
                for i in range(0, len(w)):
                    aux.append(w[i]+etx[i])
                # wk+1
                w = aux

                # para sacar la norma es la raiz cuadrada de la sumatoria de los cuadrados de la lista de errores
                # sumatoria de cuadrados
                temp = 0
                for e in error:
                    temp += e**2
                self.norma = temp

                # raiz cuadrada
                self.norma = math.sqrt(self.norma)                    
                print(self.norma,' norma')
                print(self.error,' norma')
                print(u, ' u')
                print(yc,' yc')
                print(self.salidas)
                print(error,' error')
                print(etx,' etx')
                print()
                if(taza_aprendizaje == 0):
                    self.lista_pesos1.append(w)
                    self.lista_ek1.append(self.norma)
                if(taza_aprendizaje == 1):
                    self.lista_pesos2.append(w)
                    self.lista_ek2.append(self.norma)
                if(taza_aprendizaje == 2):
                    self.lista_pesos3.append(w)
                    self.lista_ek3.append(self.norma)
                if(taza_aprendizaje == 3):
                    self.lista_pesos4.append(w)
                    self.lista_ek4.append(self.norma)
                if(taza_aprendizaje == 4):
                    self.lista_pesos5.append(w)
                    self.lista_ek5.append(self.norma)
        self.graficar()
        

    def entrenamiento_manual(self):
        self.reset()
    
        self.tazas_aprendizaje.append(float(self.taza_aprendizaje1.text()))
        self.tazas_aprendizaje.append(float(self.taza_aprendizaje2.text()))
        self.tazas_aprendizaje.append(float(self.taza_aprendizaje3.text()))
        self.tazas_aprendizaje.append(float(self.taza_aprendizaje4.text()))
        self.tazas_aprendizaje.append(float(self.taza_aprendizaje5.text()))
        self.error = float(self.error_field.text())
        # ciclo para utilizar todas las tazas de aprendizaje
        for taza_aprendizaje in range(len(self.tazas_aprendizaje)):
            w = self.w0_field.text().split(",")
            w = [float(w[i]) for i in range(len(w))]
            print('iteracion: ',taza_aprendizaje)
            self.norma = 3
        # detener el proceso si la magnitud del error es menor a un umbral є, |e| < є
            #while(self.norma > self.tazas_aprendizaje[taza_aprendizaje]):
            while(self.norma > self.error):
                # para que este funcione deben coincidir en dimension por ejemplo
                # 4 columnas en self.entradas con 4 columnas del w
                # U_k
                u = np.dot(w,self.entradas)

                # Y^c(u) = FA(u)
                yc = []
                # 0 si u < 0, 1 si u >= 0
                for i in u:
                    yc.append(1) if i >= 0 else yc.append(0)

                #ek = y^d - y^c, ydeseada - ycalculada
                error = self.salidas - yc
                # transpose no tiene efecto con unidimensionales
                # equivalente a e^t * X
                etx = np.dot(error, self.entradas)
                aux = []

                for x in etx:
                    # n * e^t* x
                    aux.append(
                        self.tazas_aprendizaje[taza_aprendizaje] * x)
                etx = aux

                aux = []
                # Wk+1 = Wk + n * e^t* x
                for i in range(0, len(w)):
                    aux.append(w[i]+etx[i])
                # wk+1
                w = aux

                # para sacar la norma es la raiz cuadrada de la sumatoria de los cuadrados de la lista de errores
                # sumatoria de cuadrados
                temp = 0
                for e in error:
                    temp += e**2
                self.norma = temp

                # raiz cuadrada
                self.norma = math.sqrt(self.norma)                    
                print(self.norma,' norma')
                print(self.error,' error')
                print(u, ' u')
                print(yc,' yc')
                print(self.salidas)
                print(error,' error')
                print(etx,' etx')
                print()
                if(taza_aprendizaje == 0):
                    self.lista_pesos1.append(w)
                    self.lista_ek1.append(self.norma)
                if(taza_aprendizaje == 1):
                    self.lista_pesos2.append(w)
                    self.lista_ek2.append(self.norma)
                if(taza_aprendizaje == 2):
                    self.lista_pesos3.append(w)
                    self.lista_ek3.append(self.norma)
                if(taza_aprendizaje == 3):
                    self.lista_pesos4.append(w)
                    self.lista_ek4.append(self.norma)
                if(taza_aprendizaje == 4):
                    self.lista_pesos5.append(w)
                    self.lista_ek5.append(self.norma)                    
            self.graficar()

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = Perceptron()
    GUI.show()
    sys.exit(app.exec_())
