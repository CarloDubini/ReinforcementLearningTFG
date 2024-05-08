import numpy as np
import matplotlib.pyplot as plt

# Función para normalizar un vector si se diera el caso
def normalize_vector(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector

def calcular_medias(datos, ventana=100):
    medias = []
    for i in range(0, len(datos), ventana):
        media = np.mean(datos[i:i+ventana])
        medias.append(media)
    return medias

# Cargar los nombres de los archivos y los nombres de las series
archivos = ["archivo1.txt", "archivo2.txt", "archivo3.txt", "archivo4.txt", "archivo5.txt", "archivo6.txt"]
nombres_series = ["Neg. Euclidea", "Neg. Euclidea Cuadrado", "Neg. Euclidea + TTG", "Neg. Euclidea Cuadrado + TTG", "Esparcida", "Esparcida + TTG"]

# Inicializar lista para almacenar los datos normalizados
datos_normalizados = []

# Cargar y normalizar los datos de cada archivo
for archivo in archivos:
    data = np.loadtxt(archivo, delimiter=',')
    data_normalized = normalize_vector(data[0:3000])
    medias = calcular_medias(data_normalized)
    datos_normalizados.append(medias)


for i, datos in enumerate(datos_normalizados):
    plt.plot(datos, label=nombres_series[i])

plt.xlabel('Iteraciones')
plt.ylabel('Valor Normalizado')
plt.title('Gráfica de Datos Normalizados')
plt.legend()
plt.savefig('Gráficagenerada')
