#!/usr/bin/env python
# coding: utf-8

# In[44]:


import joblib
import pandas as pd
import numpy as np


# # PREDICCION ABRIL

# In[54]:


# nuevos datos
abril = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\datos_marzo.csv')


# In[56]:


# Cargar modelo y scaler
model = joblib.load('modelo_sinimputar.pkl')

# Preprocesamiento (ejemplo)
# Guardar el ID en una variable separada
ids = abril[['cooler_id']].copy()

nuevos = abril[['door_opens', 'open_time', 'compressor', 'power',
       'on_time', 'min_voltage', 'max_voltage', 'temperature',
       'warning', 'year', 'month']]  # Asegúrate de que solo estén los features que entrenaste

# Predecir
pred = model.predict(nuevos)
proba = model.predict_proba(nuevos)[:, 1]

# Unir resultados con el ID
resultado = abril.copy()
resultado['pred'] = pred
resultado['proba'] = proba


# In[53]:


resultado.to_csv("predicciones_abril.csv")


# # Prediccion MAYO

# In[57]:


mayo = resultado.copy()
ids = mayo[['cooler_id']].copy()

nuevos_mayo = mayo[['door_opens', 'open_time', 'compressor', 'power',
       'on_time', 'min_voltage', 'max_voltage', 'temperature',
       'warning', 'year', 'month']]  # Asegúrate de que solo estén los features que entrenaste

# Predecir
pred_mayo = model.predict(nuevos_mayo)
proba_mayo = model.predict_proba(nuevos_mayo)[:, 1]

# Unir resultados con el ID
resultado_mayo = mayo.copy()
resultado_mayo['pred'] = pred_mayo
resultado_mayo['proba'] = proba


# In[58]:


# Ahora tienes predicciones con el cooler_id preservado
print(resultado_mayo[['cooler_id', 'pred', 'proba']].head())


# teniendo la probabilidad diaria, puedo usar probabilidad complementaria para encontrar la proba mensual

# In[64]:


# Calcular probabilidad mensual por cooler_id
proba_mensual = resultado_mayo.groupby('cooler_id')['proba'].apply(lambda p: 1 - np.prod(1 - p))

# Convertir a DataFrame si deseas exportarlo o analizarlo
proba_mensual_df = proba_mensual.reset_index(name='proba_mensual')
proba_mensual_df['proba_mensual'] = np.floor(proba_mensual_df['proba_mensual'] * 1e4) / 1e4

# Mostrar resultado
print(proba_mensual_df.head())

proba_mensual_df.to_csv('probabilidad.csv')

