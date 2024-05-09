import os
import numpy as np
import matplotlib.pyplot as plt

# Función para normalizar un vector si se diera el caso
def normalize_vector(vector):
    min_val = 0
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector

def calcular_medias(datos, ventana=100):
    medias = []
    for i in range(0, len(datos), ventana):
        media = np.mean(datos[i:i+ventana])
        medias.append(media)
    return medias

# Cargar los nombres de los archivos y los nombres de las series según la carpeta a analizar
carpeta = "AnalisisNeuronas"
normalizar = False
eliminarPrimeros = False

archivos = os.listdir(carpeta)
archivos = [os.path.join(carpeta, archivo) for archivo in archivos if archivo.endswith('.txt')]
nombres_series = [os.path.splitext(os.path.basename(archivo))[0] for archivo in archivos]

# Inicializar lista para almacenar los datos normalizados
datos_normalizados = []
if eliminarPrimeros:
    eje_x = np.arange(800, 3000, 100)
else:
    eje_x = np.arange(0, 3000, 100)
# Cargar y normalizar los datos de cada archivo
for archivo in archivos:
    data = np.loadtxt(archivo, delimiter=',')[0:3000]
    if normalizar:
        data = normalize_vector(data)
    medias = calcular_medias(data)
    if eliminarPrimeros:
        medias = medias[8:]
    datos_normalizados.append(medias)


for i, datos in enumerate(datos_normalizados):
    plt.plot(eje_x, datos, label=nombres_series[i])
    if i % 3 == 2 or i == len(datos_normalizados)-1:
        plt.xlabel('Iteraciones')
        plt.ylabel('Recompensa media')
        #plt.title('Gráfica de Datos Normalizados')
        plt.legend()
        plt.savefig(f'Gráfica{carpeta}generada{i//3}')
        plt.clf()
