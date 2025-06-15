#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ssn
from sklearn.impute import KNNImputer
import numpy as np


# In[172]:


#Exploracion de datos
sales = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\Dataset _ Coolers AC\sales.csv')
warnings = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\Dataset _ Coolers AC\warnings.csv')
coolers = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\Dataset _ Coolers AC\coolers.csv')


# In[211]:


# Asegúrate de que calday sea string para separar
coolers_fecha = coolers.copy()
coolers_fecha['calday'] = coolers_fecha['calday'].astype(str)

# Crear nuevas columnas
coolers_fecha['year'] = coolers_fecha['calday'].str[:4]
coolers_fecha['month'] = coolers_fecha['calday'].str[4:6]
coolers_fecha['day'] = coolers_fecha['calday'].str[6:8]

# Si prefieres que sean enteros:
coolers_fecha[['year', 'month', 'day']] = coolers_fecha[['year', 'month', 'day']].astype(int)


# In[212]:


coolers_fecha.to_csv('coolers_fecha.csv', header=True)


# In[174]:


#Se asigna columna de si presenta warning = 1 || no presenta = 0
coolers_warnings = coolers
coolers_warnings['warning'] = coolers['cooler_id'].isin(warnings['cooler_id']).astype(int)


# In[175]:


# Media por grupo
media = coolers.groupby('warning').mean(numeric_only=True)

# Mediana por grupo
mediana = coolers.groupby('warning').median(numeric_only=True)

# Moda por grupo (puede devolver varias filas si hay empates)
moda = coolers.groupby('warning').agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)


# In[176]:


media


# In[177]:


#visualizacion de datos
coolers_warnings.boxplot(column='door_opens', by='warning', grid=False)


# In[178]:


coolers_warnings.boxplot(column='power', by='warning', grid=False)


# In[179]:


coolers_warnings.boxplot(column='min_voltage', by='warning', grid=False)


# In[180]:


coolers_warnings.boxplot(column='max_voltage', by='warning', grid=False)


# In[181]:


#SUBSET DE LOS QUE NO TIENEN FALLAS PARA MAPEAR DISTRIBUCION
# Paso 1: seleccionar 53 cooler_id únicos donde warning == 0
coolers_sin_warning = coolers[coolers['warning'] == 0]
cooler_ids_sample = coolers_sin_warning['cooler_id'].drop_duplicates().sample(n=53, random_state=42)

# Paso 2: seleccionar todas las filas con esos cooler_id
subset_sinwarnings = coolers[coolers['cooler_id'].isin(cooler_ids_sample)].copy()

subset_nowarning_completo = coolers[coolers['warning'] == 0].copy()

#CON WARNINGS
subset_warnings = coolers[coolers['warning'] == 1].copy()


# In[182]:


subset_warnings['cooler_id'].head()


# In[183]:


import matplotlib.pyplot as plt
features =  ['door_opens', 'open_time',	'compressor',	'power',	'on_time',	'min_voltage',	'max_voltage',	'temperature']  
# Convertir calday a datetime
subset_warnings['calday'] = pd.to_datetime(subset_warnings['calday'].astype(str), format='%Y%m%d')

# Extraer año y mes
subset_warnings['year'] = subset_warnings['calday'].dt.year
subset_warnings['month'] = subset_warnings['calday'].dt.month

# Agrupar por año y mes, calcular la media
monthly_means = subset_warnings.groupby(['year', 'month'])[features].mean().reset_index()

# Graficar cada variable por separado
for feature in features:
    plt.figure(figsize=(10, 5))
    
    for year in sorted(monthly_means['year'].unique()):
        subset = monthly_means[monthly_means['year'] == year]
        plt.plot(subset['month'], subset[feature], marker='o', label=f'Año {year}')
    
    plt.title(f'Comportamiento mensual de {feature}')
    plt.xlabel('Mes')
    plt.ylabel(feature)
    plt.xticks(range(1, 13))
    plt.legend(title='Año')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# In[184]:


# PLOTEA DATOS POR COOLER ID
def plot_cooler_features_by_month(df, cooler_id):
    """
    Genera gráficas por mes y año para cada feature de un cooler específico.

    :param df: DataFrame con los datos
    :param cooler_id: ID del cooler a analizar
    """
    # Filtrar por cooler_id
    df_cooler = df[df['cooler_id'] == cooler_id].copy()
    
    if df_cooler.empty:
        print(f"No se encontraron datos para cooler_id: {cooler_id}")
        return
    
    # Convertir calday a datetime manejando formatos mixtos
    df_cooler['calday'] = pd.to_datetime(df_cooler['calday'], errors='coerce', infer_datetime_format=True)
    
    # Verifica si hay fechas no convertidas
    if df_cooler['calday'].isnull().any():
        print("Advertencia: algunas fechas no se pudieron convertir y se omitirán.")
        df_cooler = df_cooler.dropna(subset=['calday'])
    
    # Extraer año y mes
    df_cooler['year'] = df_cooler['calday'].dt.year
    df_cooler['month'] = df_cooler['calday'].dt.month
    
    # Agrupar por año y mes
    monthly_means = df_cooler.groupby(['year', 'month']).mean(numeric_only=True).reset_index()
    
    # Lista de features a graficar
    features = ['door_opens', 'open_time', 'compressor', 'power', 
                'on_time', 'min_voltage', 'max_voltage', 'temperature']
    
    # Graficar cada feature
    for feature in features:
        plt.figure(figsize=(10, 5))
        
        for year in sorted(monthly_means['year'].unique()):
            subset = monthly_means[monthly_means['year'] == year]
            plt.plot(subset['month'], subset[feature], marker='o', label=f'Año {year}')
        
        plt.title(f'Comportamiento mensual de {feature} (Cooler {cooler_id})')
        plt.xlabel('Mes')
        plt.ylabel(feature)
        plt.xticks(range(1, 13))
        plt.legend()
        plt.grid(True)
        plt.show()


# In[185]:


plot_cooler_features_by_month(subset_warnings, '8d3260b5f7e49fff02b4743037a52406b19279d9cd0144785f653afe0f4622b8')


# In[186]:


#WARNING POR DIA
def plot_cooler_features_by_day(df, cooler_id):
    """
    Genera gráficas por día del mes y año para cada feature de un cooler específico.

    :param df: DataFrame con los datos
    :param cooler_id: ID del cooler a analizar
    """
    # Filtrar por cooler_id
    df_cooler = df[df['cooler_id'] == cooler_id].copy()
    
    if df_cooler.empty:
        print(f"No se encontraron datos para cooler_id: {cooler_id}")
        return
    
    
    # Verifica si hay fechas no convertidas
    if df_cooler['calday'].isnull().any():
        print("Advertencia: algunas fechas no se pudieron convertir y se omitirán.")
        df_cooler = df_cooler.dropna(subset=['calday'])
    
    # Extraer año, mes y día
    df_cooler['year'] = df_cooler['calday'].dt.year
    df_cooler['month'] = df_cooler['calday'].dt.month
    df_cooler['day'] = df_cooler['calday'].dt.day
    
    # Agrupar por año, mes y día
    daily_means = df_cooler.groupby(['year', 'month', 'day']).mean(numeric_only=True).reset_index()
    
    # Lista de features a graficar
    features = ['door_opens', 'open_time', 'compressor', 'power', 
                'on_time', 'min_voltage', 'max_voltage', 'temperature']
    
    # Graficar cada feature
    for feature in features:
        plt.figure(figsize=(12, 6))
        
        # Graficar para cada año y mes
        for (year, month), group in daily_means.groupby(['year', 'month']):
            plt.plot(group['day'], group[feature], marker='o', label=f'{year}-{month:02d}')
        
        plt.title(f'Comportamiento diario de {feature} (Cooler {cooler_id})')
        plt.xlabel('Día del mes')
        plt.ylabel(feature)
        plt.xticks(range(1, 32))
        plt.legend(title='Año-Mes')
        plt.grid(True)
        plt.show()


# In[187]:


plot_cooler_features_by_day(subset_warnings, '8d3260b5f7e49fff02b4743037a52406b19279d9cd0144785f653afe0f4622b8')


# In[188]:


# Asegúrate de que calday esté en datetime
subset_warnings['calday'] = pd.to_datetime(subset_warnings['calday'])

# Extrae año y mes
subset_warnings['year'] = subset_warnings['calday'].dt.year
subset_warnings['month'] = subset_warnings['calday'].dt.month

# Crear un DataFrame con todos los meses esperados por cooler_id y año
cooler_periods = subset_warnings[['cooler_id', 'year']].drop_duplicates()

# Expandir con todos los meses
cooler_periods = cooler_periods.assign(key=1).merge(
    pd.DataFrame({'month': range(1, 13), 'key': 1}),
    on='key'
).drop('key', axis=1)

# Obtener combinaciones realmente presentes en los datos
present_months = subset_warnings[['cooler_id', 'year', 'month']].drop_duplicates()

# Detectar combinaciones que faltan
missing_months = pd.merge(
    cooler_periods, 
    present_months, 
    on=['cooler_id', 'year', 'month'], 
    how='left', 
    indicator=True
).query('_merge == "left_only"').drop('_merge', axis=1)

missing_months.head(10)  # Muestra las primeras 10 filas


# In[189]:


missing_months_filtrado = missing_months[missing_months['year'] != 2025]

faltantes_por_cooler = missing_months_filtrado.groupby('cooler_id').size().reset_index(name='meses_faltantes')
faltantes_por_cooler = faltantes_por_cooler.sort_values(by='meses_faltantes', ascending=False)
faltantes_por_cooler


# In[190]:


'''# imputacion de datos con KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
scaler = StandardScaler()
sinwarnings_scaled = scaler.fit_transform(subset_sinwarnings[features])
conwarnings_scaled = scaler.fit_transform(subset_warnings[features])

def imputador(df, variables):
    """
    Imputa valores faltantes en columnas numéricas usando KNN,
    preservando columnas no imputadas como 'cooler_id'.

    Parámetros:
    - df: DataFrame original
    - variables: lista de columnas numéricas a imputar

    Retorna:
    - DataFrame con las columnas imputadas y las demás columnas intactas.
    """
    # Copia del DataFrame
    df_copy = df.copy()

    # Asegurar que todas las variables están presentes
    missing = [v for v in variables if v not in df_copy.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el DataFrame: {missing}")

    # Aplicar imputación solo a las variables indicadas
    imputer = KNNImputer(n_neighbors=3)
    X_imputed = imputer.fit_transform(df_copy[variables])

    # Reasignar los valores imputados
    df_copy[variables] = X_imputed

    return df_copy

# Reconstruye DataFrame a partir del array y columnas
sinwarnings_scaled_df = pd.DataFrame(sinwarnings_scaled, columns=features)
# Ahora aplica tu función
sinwarnings_imputed = imputador(sinwarnings_scaled_df, features)'''


# In[191]:


def generar_target_por_columna(df, base_col, k, cooler_col='cooler_id', date_col='calday',
                                window=30, threshold='iqr', target_col='Target'):

    df = df.sort_values([cooler_col, date_col]).copy()

    # Detectar umbral atípico
    if threshold == 'iqr':
        Q1 = df[base_col].quantile(0.25)
        Q3 = df[base_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
    elif isinstance(threshold, (int, float)):
        lower_bound = -np.inf
        upper_bound = threshold
    else:
        raise ValueError("threshold debe ser 'iqr' o un número")

    # Marcar días con valor atípico
    df['anomalía'] = ((df[base_col] < lower_bound) | (df[base_col] > upper_bound)).astype(int)

    # Inicializar columna de target
    df[target_col] = 0

    # Expandir anomalías hacia los días anteriores
    for cooler in df[cooler_col].unique():
        idx = df[df[cooler_col] == cooler].index
        flags = df.loc[idx, 'anomalía'].values

        target_flags = np.zeros(len(flags), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1:
                start = max(0, i - window)
                target_flags[start:i+1] = 1  # incluye el día actual y días anteriores
        df.loc[idx, target_col] = target_flags

    df.drop(columns=['anomalía'], inplace=True)
    return df


# In[192]:


subset_warnings.columns


# # GENERAR TARGET METRIC

# In[193]:


def generar_target_por_anomalia(df, base_cols, cooler_col='cooler_id', date_col='calday',
                                  window=30, target_col='Target'):
    """
    Marca como 1 el target si ambas columnas base tienen valor 0 en un día dado,
    y propaga ese target a los siguientes 'window' días para cada cooler.
    
    Parámetros:
        df: DataFrame de entrada.
        base_cols: Lista con los nombres de las dos columnas a evaluar.
        cooler_col: Columna con el ID del cooler.
        date_col: Columna con la fecha (tipo datetime).
        window: Número de días para extender el target después de una anomalía.
        target_col: Nombre de la columna de salida.
    """

    if len(base_cols) != 2:
        raise ValueError("base_cols debe contener exactamente dos nombres de columna")

    col1, col2 = base_cols

    df = df.sort_values([cooler_col, date_col]).copy()

    # Detectar días donde ambas columnas valen 0
    df['anomalía'] = ((df[col1] == 0) & (df[col2] == 0)).astype(int)

    # Inicializar columna target
    df[target_col] = 0

    # Expandir anomalías a los siguientes N días
    for cooler in df[cooler_col].unique():
        idx = df[df[cooler_col] == cooler].index
        flags = df.loc[idx, 'anomalía'].values

        target_flags = np.zeros(len(flags), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1:
                target_flags[i:i + window + 1] = 1
        df.loc[idx, target_col] = target_flags

    df.drop(columns=['anomalía'], inplace=True)
    return df


# In[194]:


subset_warnings_targetpuertas = subset_warnings
subset_warnings_targetpuertas['Target'] = None
subset_warnings_targetpuertas = generar_target_por_anomalia(subset_warnings, base_cols=['door_opens', 'open_time'])


# In[195]:


print(subset_warnings_targetpuertas['Target'].value_counts())


# In[196]:


subset_warnings_targetpuertas[subset_warnings_targetpuertas['Target'] == 1]['cooler_id'].nunique()


# In[197]:


subset_warnings_targetpuertas['Motive'] = 'puertas_no_abren'


# # Target de warning por columna

# In[198]:


def generar_target_por_columna_dias_futuros(df, base_col, k, cooler_col='cooler_id', date_col='calday',
                                window=30, threshold='iqr', target_col='Target'):


    df = df.sort_values([cooler_col, date_col]).copy()

    # Detectar umbral atípico
    if threshold == 'iqr':
        Q1 = df[base_col].quantile(0.25)
        Q3 = df[base_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k* IQR
        upper_bound = Q3 + k* IQR
    elif isinstance(threshold, (int, float)):
        lower_bound = -np.inf
        upper_bound = threshold
    else:
        raise ValueError("threshold debe ser 'iqr' o un número")

    # Marcar días con valor atípico
    df['anomalía'] = ((df[base_col] < lower_bound) | (df[base_col] > upper_bound)).astype(int)

    # Inicializar columna de target
    df[target_col] = 0

    # Expandir anomalías a los siguientes N días
    for cooler in df[cooler_col].unique():
        idx = df[df[cooler_col] == cooler].index
        flags = df.loc[idx, 'anomalía'].values

        # Marcar los siguientes N días como 1
        target_flags = np.zeros(len(flags), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1:
                target_flags[i:i+window+1] = 1  # incluye el día actual
        df.loc[idx, target_col] = target_flags

    df.drop(columns=['anomalía'], inplace=True)
    return df


# falla con compresor

# In[199]:


subset_warnings_targetcompresor = generar_target_por_columna(subset_warnings, base_col='compressor',k=0.5)
subset_warnings_targetcompresor['Motive'] = 'compressor'
subset_warnings_targetcompresor[subset_warnings_targetcompresor['Target'] == 1]['cooler_id'].nunique()


# In[200]:


compressor = subset_warnings['compressor']
Q1 = compressor.quantile(0.25)
Q3 = compressor.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 0.5 * IQR
upper = Q3 + 0.5 * IQR
print(f"Rango permitido: {lower} a {upper}")
print(f"Valores fuera del rango: {(compressor < lower).sum()} debajo, {(compressor > upper).sum()} arriba")


# falla con temperatura

# In[201]:


subset_warnings_targettemp = generar_target_por_columna(subset_warnings, base_col='temperature', k=1.5)
subset_warnings_targettemp['Motive'] = 'temperature'
subset_warnings_targettemp[subset_warnings_targetcompresor['Target'] == 1]['cooler_id'].nunique()


# In[202]:


compressor = subset_warnings['temperature']
Q1 = compressor.quantile(0.25)
Q3 = compressor.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"Rango permitido: {lower} a {upper}")
print(f"Valores fuera del rango: {(compressor < lower).sum()} debajo, {(compressor > upper).sum()} arriba")


# uniendo fallas

# In[203]:


# Luego: unir el resultado con el tercero
fallas_concat = pd.concat(
    [subset_warnings_targettemp, subset_warnings_targetcompresor],
    axis=0,  # vertical
    ignore_index=True
)


# In[204]:


fallas_concat.nunique()


# In[205]:


fallas_concat.to_csv("Fallas_con_target.csv", header=True,  encoding='utf-8' )


# In[206]:


subset_sinwarnings['Motive'] = 'None'
subset_sinwarnings['Target'] = 0
subset_sinwarnings.to_csv("subset_sinwarnings.csv", header=True,  encoding='utf-8')


# In[207]:


subset_nowarning_completo['Motive'] = 'None'
subset_nowarning_completo['Target'] = 0
subset_nowarning_completo.to_csv("nowarning_complete.csv", header=True,  encoding='utf-8')


# In[208]:


# datasets sin incluir febrero
def sin_feb(df, ruta_salida):

    # Asegurarse de que calday esté en formato datetime
    df = df.copy()
    df['calday'] = pd.to_datetime(df['calday'].astype(str), format='%Y%m%d', errors='coerce')

    # Filtrar todo excepto febrero (mes 2)
    df_sin_febrero = df[df['calday'].dt.month != 2]

    # Guardar a CSV
    df_sin_febrero.to_csv(ruta_salida, index=False)

    print(f"Archivo guardado sin registros de febrero en: {ruta_salida}")
    return df_sin_febrero

sin_feb(fallas_concat, "fallas_sin_feb.csv")
sin_feb(subset_sinwarnings, "sinwarnings_sinfeb.csv")


# In[209]:


# datasets SOLO  febrero
def con_feb(df, ruta_salida):

    # Asegurarse de que calday esté en formato datetime
    df = df.copy()
    df['calday'] = pd.to_datetime(df['calday'].astype(str), format='%Y%m%d', errors='coerce')

    # Filtrar todo excepto febrero (mes 2)
    df_sin_febrero = df[df['calday'].dt.month == 2]

    # Guardar a CSV
    df_sin_febrero.to_csv(ruta_salida, index=False)

    print(f"Archivo guardado sin registros de febrero en: {ruta_salida}")
    return df_sin_febrero

con_feb(fallas_concat, "fallas_feb.csv")
con_feb(subset_sinwarnings, "sinwarnings_feb.csv")

