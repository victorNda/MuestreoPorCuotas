import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# DATOS ACTUALIZADOS
# =========================

estudiantes = [
    7000, 8000, 7500, 9000, 8500,
    7000, 12000, 8000, 7500, 9000,
    7000, 8000, 8500, 7500, 7000,
    9000, 12000, 8000, 7500, 8500
]

trabajadores = [
    12000, 15000, 18000, 20000, 22000,
    15000, 12000, 25000, 18000, 20000,
    15000, 22000, 18000, 12000, 30000,
    25000, 20000, 15000, 18000, 22000
]

adultos = [
    4000, 5000, 4000, 6000, 5000,
    4000, 7000, 5000, 4000, 6000,
    5000, 4000, 7000, 5000, 4000,
    6000, 5000, 4000, 7000, 5000
]

# TODOS LOS DATOS
todos = estudiantes + trabajadores + adultos


# =========================
# FUNCION DE ANALISIS
# =========================

def analizar_datos(nombre, datos):

    print("\n===================================")
    print("GRUPO:", nombre)
    print("===================================")

    datos_np = np.array(datos)

    # =========================
    # MEDIA
    # =========================

    media = np.mean(datos_np)
    print("Media:", media)

    # =========================
    # MEDIANA
    # =========================

    mediana = np.median(datos_np)
    print("Mediana:", mediana)

    # =========================
    # MODA
    # =========================

    valores_unicos, conteos = np.unique(datos_np, return_counts=True)

    max_conteo = np.max(conteos)

    moda = valores_unicos[conteos == max_conteo]

    print("Moda:", moda)

    # =========================
    # MEDIDAS DE DISPERSION
    # =========================

    varianza = np.var(datos_np, ddof=1)
    desviacion = np.std(datos_np, ddof=1)

    coeficiente_variacion = (desviacion / media) * 100

    error_estandar = desviacion / np.sqrt(len(datos_np))

    print("\nVarianza:", varianza)
    print("Desviacion Estandar:", desviacion)
    print("Coeficiente de Variacion:", coeficiente_variacion)
    print("Error Estandar:", error_estandar)

    # =========================
    # ESTADISTICAS DE POSICION
    # =========================

    q1 = np.percentile(datos_np, 25)
    q3 = np.percentile(datos_np, 75)

    print("\nQ1:", q1)
    print("Q3:", q3)

    # =========================
    # ASIMETRIA DE PEARSON
    # =========================

    asimetria = (3 * (media - mediana)) / desviacion

    print("Asimetria de Pearson:", asimetria)

    # =========================
    # TABLA DE FRECUENCIAS
    # =========================

    df = pd.DataFrame({'Datos': datos_np})

    tabla = df['Datos'].value_counts().sort_index()

    valores = tabla.index
    fa = tabla.values

    faa = np.cumsum(fa)

    fr = fa / len(datos_np)

    fra = np.cumsum(fr)

    tabla_final = pd.DataFrame({
        'Valor': valores,
        'FA': fa,
        'FAA': faa,
        'FR': fr,
        'FRA': fra
    })

    print("\nTABLA DE FRECUENCIAS")
    print(tabla_final)

    # POSICIONES PARA GRAFICAS
    x = np.arange(len(valores))

    # ==================================================
    # GRAFICA FRECUENCIA ABSOLUTA
    # ==================================================

    fig1, ax1 = plt.subplots(figsize=(10,5))

    ax1.bar(x, fa)

    ax1.set_xticks(x)
    ax1.set_xticklabels(valores)

    # LINEA DE MEDIA
    indice_media = np.argmin(np.abs(valores - media))

    ax1.axvline(
        indice_media,
        linestyle='--',
        linewidth=2,
        label='Media'
    )

    # DESVIACION ESTANDAR
    limite_inferior = media - desviacion
    limite_superior = media + desviacion

    indice_inf = np.argmin(np.abs(valores - limite_inferior))
    indice_sup = np.argmin(np.abs(valores - limite_superior))

    ax1.axvline(
        indice_inf,
        linestyle=':',
        linewidth=2,
        label='-1 Desv.'
    )

    ax1.axvline(
        indice_sup,
        linestyle=':',
        linewidth=2,
        label='+1 Desv.'
    )

    ax1.set_title(f'Frecuencia Absoluta - {nombre}')
    ax1.set_xlabel('Valores')
    ax1.set_ylabel('FA')

    ax1.legend()

    ax1.grid(True)

    fig1.tight_layout()

    plt.show()

    # ==================================================
    # GRAFICA FRECUENCIA ABSOLUTA ACUMULADA
    # ==================================================

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(x, faa, marker='o')

    ax2.set_xticks(x)
    ax2.set_xticklabels(valores)

    ax2.set_title(f'Frecuencia Absoluta Acumulada - {nombre}')
    ax2.set_xlabel('Valores')
    ax2.set_ylabel('FAA')

    ax2.grid(True)

    fig2.tight_layout()

    plt.show()

    # ==================================================
    # GRAFICA FRECUENCIA RELATIVA
    # ==================================================

    fig3, ax3 = plt.subplots(figsize=(10,5))

    ax3.bar(x, fr)

    ax3.set_xticks(x)
    ax3.set_xticklabels(valores)

    # LINEA DE MEDIA
    ax3.axvline(
        indice_media,
        linestyle='--',
        linewidth=2,
        label='Media'
    )

    # DESVIACION ESTANDAR
    ax3.axvline(
        indice_inf,
        linestyle=':',
        linewidth=2,
        label='-1 Desv.'
    )

    ax3.axvline(
        indice_sup,
        linestyle=':',
        linewidth=2,
        label='+1 Desv.'
    )

    ax3.set_title(f'Frecuencia Relativa - {nombre}')
    ax3.set_xlabel('Valores')
    ax3.set_ylabel('FR')

    ax3.legend()

    ax3.grid(True)

    fig3.tight_layout()

    plt.show()

    # ==================================================
    # GRAFICA FRECUENCIA RELATIVA ACUMULADA
    # ==================================================

    fig4, ax4 = plt.subplots(figsize=(10,5))

    ax4.plot(x, fra, marker='o')

    ax4.set_xticks(x)
    ax4.set_xticklabels(valores)

    ax4.set_title(f'Frecuencia Relativa Acumulada - {nombre}')
    ax4.set_xlabel('Valores')
    ax4.set_ylabel('FRA')

    ax4.grid(True)

    fig4.tight_layout()

    plt.show()


# =========================
# ANALISIS POR CUOTAS
# =========================

analizar_datos("Estudiantes", estudiantes)

analizar_datos("Trabajadores", trabajadores)

analizar_datos("Adultos Mayores", adultos)

# =========================
# ANALISIS TOTAL
# =========================

analizar_datos("TODOS LOS DATOS", todos)