import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, kstest, f, t
from statistics import multimode
import matplotlib.pyplot as plt
import seaborn as sns

"""
ESTADÍSTICA: "Estadística descriptiva e inferencia", Pedro Pablo Vicente González 24/04/2025
"""


#   CARGA DE DATOS

muestras = pd.read_excel("C:/Users/ppabl/Desktop/Tarea Estadística/datosejercicioevaluacionanchuras.xlsx")

mptemprano = pd.DataFrame(muestras.loc[muestras["Época histórica"] == 1, "Anchura del cráneo"]) # Muestra predinástico temprano
mptardio = pd.DataFrame(muestras.loc[muestras["Época histórica"] == 2, "Anchura del cráneo"]) # Muestra predinástico tardío


#   EJERCICIO 1

# 1.A.

# ESTADÍSTICOS

def estadisticos(muestra):
    """
    Ofrece una versión ampliada del método describe() con estadísticos para la moda, rango, varianza curtosis y asimetría.
    
    Input: Dataframe (trabaja con la primera columna del dataframe/ la anchura de cráneo).
    """
    muestra = muestra.iloc[:,0] # Primera columna/ Anchura del cráneo

    estadisticos = muestra.describe()            # Inicia una lista con los estadísticos del método "describe()". 
    estadisticos["moda"] = multimode(muestra)    # Vamos añadiendo otros estadísticos a la listas "estadisticos"
    estadisticos["rango"] = muestra.max()- muestra.min()
    estadisticos["varianza"] = muestra.var(ddof=1)
    estadisticos["curtosis"] = kurtosis(muestra, fisher=True) # Usamos el coeficiente de curtosis de Fisher
    estadisticos["coef_asimetria"] = skew(muestra)

    estadisticos = pd.DataFrame(estadisticos)

    return estadisticos 


estad_temprano= estadisticos(mptemprano)
print(estad_temprano)

estad_tardio= estadisticos(mptardio)
print(estad_tardio)


# DIAGRAMAS DE CAJAS Y BIGOTES

def bp(muestra,titulo):
    """
    Crea un diagrama de cajas y bigotes en el que se marcan la media y la mediana.
    
    Input: Dataframe y título del gráfico.
    """
    meanlineprops = dict(linestyle='--', linewidth=1, color='yellow')  # Línea que señaliza la media
    sns.boxplot(y=muestra.iloc[:, 0], medianprops={"color": "r"},showmeans=True, meanprops=meanlineprops, meanline=True) # Línea que señaliza la mediana
    plt.title(titulo)
    plt.grid(True, alpha=0.3, linestyle= '--')
    plt.show()

bp(mptemprano, "Predinástico temprano")
bp(mptardio, "Predinástico tardío")


# HISTOGRAMAS

def hg(muestra,titulo):
    """
    Crea un hoistograma en el que se marcan la media y la mediana.
    
    Input: Dataframe y título del gráfico.
    """
    sns.histplot(muestra, legend= False)
    median= np.median(muestra)
    mean= np.mean(muestra)
    plt.axvline(median, color='red', linestyle='-', linewidth=2, label=f'Mediana = {median:.1f}') # Línea que señaliza la mediana
    plt.axvline(mean, color='yellow', linestyle='--', linewidth=2, label=f'Media = {median:.1f}') # Línea que señaliza la media
    print(muestra.value_counts())
    plt.title(titulo)
    plt.grid(True, alpha=0.3, linestyle= '--')
    plt.xlabel("Anchura del cráneo (mm)")
    plt.show()

hg(mptemprano, "Predinástico temprano")
hg(mptardio, "Predinástico tardío")



# 1.B.

def kolmogorov_smirnov_test(data, n_significacion):
    """
    Test de bondad de ajuste de Kolmogorov-Smirnov, comparación con la distribución normal (0,1).
    
    Input: Dataframe y nivel de significación del test.
    """

    data = data.iloc[:, 0]  # Aseguramos que elegimos la columna Anchura del cráneo

    data_z = (data - data.mean()) / data.std()  # Tipificación de los datos
    
    ksstat, p_value = kstest(data_z, "norm")  # Contraste con distribución normal
    
    print(f"Estadístico Kolmogorov-Smirnov: {ksstat:.4f}")  # Resultados
    print(f"P-valor: {p_value:.4f}")
    # Interpretación de resultados
    if p_value < n_significacion:
        print(f"Como el p-valor es {p_value:.4f}, rechazamos la hipótesis nula: al {(1-n_significacion)*100:.0f}% de confianza, los datos no siguen una distribución normal.")
    else:
        print(f"Como el p-valor es {p_value:.4f}, no rechazamos la hipótesis nula: al {(1-n_significacion)*100:.0f}% de confianza, no rechazamos que los datos sigan una distribución normal.")


kolmogorov_smirnov_test(mptemprano, 0.05)
# kolmogorov_smirnov_test(mptemprano, 0.0488) # niveles inferiores de significación hacen que no rechacemos la hipotesis nula

kolmogorov_smirnov_test(mptardio, 0.05)



#   EJERCICIO 2

# 2.A.

# TEST DE IGUALDAD DE VARIANZAS POBLACIONALES DESCONOCIDAS

def test_ig_varpob_desconocidas(data1,data2,n_significacion):
    """
    Test de igualdad de varianzas poblacionales para poblaciones normales con muestras independientes.
    
    Input: Dos muestras y nivel de significación del test.
    """
    data1 = data1.iloc[:,0]                 
    data2 = data2.iloc[:,0]

    n, m = len(data1), len(data2) # Tamaños muestrales

    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1) # Varianzas de las muestras

    F = var1 / var2 # Valor del estadístico
    dfn, dfd = n - 1, m - 1 # Grados de libertad del numerador y denominador

    # P-valor teniendo en cuenta las dos colas
    if F > 1:
        p_value = 2 * (1 - f.cdf(F, dfn, dfd))  # Cola derecha
    else:
        p_value = 2 * f.cdf(F, dfn, dfd)        # Cola izquierda

    print(f"F = {F:.4f}")

    print(f"p-value = {p_value:.4f}")

    if p_value < n_significacion:
        print(f"Como el p-valor es {p_value:.4f}, rechaza la hipótesis nula;  al {(1-n_significacion)*100:.0f}% de confianza no aceptamos que las muestras presenten varianzas poblacionales iguales.")
    else:
        print(f"Como el p-valor es {p_value:.4f}, no rechaza la hipótesis nula:  al {(1-n_significacion)*100:.0f}% de confianza podemos considerar que las muestras presentan varianzas poblacionales iguales.")


test_ig_varpob_desconocidas(mptemprano,mptardio, 0.1)


# INTERVALO DE CONFIANZA PARA LA DIFERENCIA DE MEDIAS CON VARIANZAS DESCONOCIDAS PERO IGUALES
def ic_igualdad_medpob(data1,data2,n_significacion):
    """
    Intervalo de confianza para la diferencia de medias con varianzas poblacionales desconocidas pero iguales; para muestras aleatorias simples independientes distribuidas normalmente.
    
    Input: Dos muestras y nivel de significación.
    """
    data2 = data2.iloc[:,0]
    data1 = data1.iloc[:,0]

    n, m = len(data1), len(data2)   # Tamaños muestrales

    mean1, mean2 = np.mean(data1), np.mean(data2) # Medias muestrales
    
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1) # Varianzas de las muestras

    se = np.sqrt(((n * var1 + m * var2)*((1/n)+(1/m)))/(n + m - 2)) # Término de desviación estándar

    df = n + m - 2 # Grados de libertad

    t_crit = t.ppf(1 - n_significacion/2, df) # Valor crítico t

    # Intervalo de confianza
    dif = mean1 - mean2
    ic_1 = dif - t_crit * se
    ic_2 = dif + t_crit * se

    print(f"Intervalo de confianza al {(1-n_significacion)*100:.0f}% de grado de confianza para la diferencia de medias: ({ic_1:.4f}, {ic_2:.4f})")
    print()

ic_igualdad_medpob(mptemprano, mptardio, 0.1)  # IC al 90% 
ic_igualdad_medpob(mptemprano, mptardio, 0.05) # IC al 95%
ic_igualdad_medpob(mptemprano, mptardio, 0.01) # IC al 99%
    # A menor n.significación, mayor grado de confianza/ más amplio será el intervalo.


# 2.B.

# CONTRASTE DE HIPÓTESIS PARA LA DIFERENCIA DE MEDIAS CON VARIANZAS DESCONOCIDAS PERO IGUALES.

def ch_igualdad_medpob(data1,data2,n_significacion):
    """
    Contrasde de hipótesis para la diferencia de medias poblacionales con varianzas desconocidas pero iguales; para muestras aleatorias simples independientes distribuidas normalmente.
    
    Input: Dos muestras y nivel de significación del test.
    """
                      
    data2 = data2.iloc[:,0]
    data1 = data1.iloc[:,0]

    n, m = len(data1), len(data2)   # Tamaños muestrales

    mean1, mean2 = np.mean(data1), np.mean(data2) # Medias muestrales
    
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1) # Varianzas de las muestras

    se = np.sqrt(((n * var1 + m * var2)*((1/n)+(1/m)))/(n + m - 2)) # Término de desviación estándar

    df = n + m - 2 # Grados de libertad

    t_crit = t.ppf(1 - n_significacion/2, df) # Valores críticos t (+-t_crit es un valor simétrico respecto a 0)

    # Intervalo de confianza
    dif = mean1 - mean2

    T= np.absolute(dif)/se #Estadístico T

    print(f"El valor del estadístico T es: {T}")
    
    if T > t_crit:
        print(f"Como T = {abs(T):.4f} > valor crítico t = {t_crit:.4f}, se rechaza la hipótesis nula:  al {(1-n_significacion)*100:.0f}% de confianza podemos aceptar que las medias poblacionales sean diferentes.")
    else:
        print(f"Como T = {abs(T):.4f} ≤ valor crítico t = {t_crit:.4f}, no se rechaza la hipótesis nula:  al {(1-n_significacion)*100:.0f}% de confianza podemos aceptar que las medias poblacionales sean iguales.")

ch_igualdad_medpob(mptemprano, mptardio, 0.1)