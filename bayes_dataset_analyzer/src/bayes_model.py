import pandas as pd
import numpy as np

def cargar_datos(ruta="data/raw/peliculas.csv"):
    """Carga el dataset de películas desde la ruta especificada."""
    try:
        df = pd.read_csv(ruta)
        return df
    except Exception as e:
        return None

def calcular_probabilidad_a_priori(df, columna_objetivo, valor_objetivo):
    """Calcula P(Y = valor_objetivo)"""
    total = len(df)
    if total == 0:
        return 0.0
    conteo = len(df[df[columna_objetivo] == valor_objetivo])
    return conteo / total

def calcular_probabilidad_condicional(df, columna_feature, valor_feature, columna_objetivo, valor_objetivo, k=1):
    """
    Calcula P(Feature = valor | Objetivo = valor_objetivo) usando 
    Suavizado de Laplace (k) para evitar divisiones entre cero.
    """
    df_filtrado = df[df[columna_objetivo] == valor_objetivo]
    total_clase = len(df_filtrado)
    
    if total_clase == 0:
        return 0.0
    
    # Coincidencias con la característica
    coincidencias = len(df_filtrado[df_filtrado[columna_feature] == valor_feature])
    
    # Número de categorías únicas de esa feature (para Laplace)
    categorias_unicas = df[columna_feature].nunique()
    
    # Aplicando suavizado de Laplace
    probabilidad = (coincidencias + k) / (total_clase + (k * categorias_unicas))
    return probabilidad

def predecir_naive_bayes(df, caracteristicas_usuario):
    """
    Clasificador Naive Bayes propio para calcular la probabilidad 
    de obtener una Calificación Alta ("Sí" o "No").
    """
    col_obj = "Calificacion_Alta"
    
    # Probabilidades a priori
    p_si = calcular_probabilidad_a_priori(df, col_obj, "Sí")
    p_no = calcular_probabilidad_a_priori(df, col_obj, "No")
    
    # Numeradores de Bayes (P(X|Sí) * P(Sí) y P(X|No) * P(No))
    num_si = p_si
    num_no = p_no
    
    detalles = {}
    
    for columna, valor in caracteristicas_usuario.items():
        prob_feat_dado_si = calcular_probabilidad_condicional(df, columna, valor, col_obj, "Sí")
        prob_feat_dado_no = calcular_probabilidad_condicional(df, columna, valor, col_obj, "No")
        
        num_si *= prob_feat_dado_si
        num_no *= prob_feat_dado_no
        
        detalles[columna] = {
            "P(Feat|Sí)": prob_feat_dado_si,
            "P(Feat|No)": prob_feat_dado_no
        }
        
    # Evitar división entre cero en la evidencia total
    evidencia_total = num_si + num_no
    if evidencia_total == 0:
        return "Indeterminado", 0.5, detalles
        
    prob_final_si = num_si / evidencia_total
    prob_final_no = num_no / evidencia_total
    
    if prob_final_si >= prob_final_no:
        return "Sí", prob_final_si, detalles
    else:
        return "No", prob_final_no, detalles