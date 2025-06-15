#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as ssn
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
import joblib


# In[22]:


df_warnings = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\Datasets_Entrenamiento\fallas_sin_feb.csv')
df_nowarnings = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\Datasets_Entrenamiento\sinwarnings_sinfeb.csv')

#df_nowarnings = pd.read_csv(r'C:\Users\REPO\Tec2025\HackForHer\nowarning_complete.csv')

df_warnings = df_warnings.drop(columns=['Unnamed: 0'], errors='ignore')  # elimina si existe
df_warnings_positivos = df_warnings[df_warnings['Target'] == 1].copy()


# In[23]:


df_unidos = pd.concat(
    [df_warnings, df_nowarnings],
    axis=0,  # vertical
    ignore_index=True
)
df_unidos.drop(columns=['cooler_id'], errors='ignore')


# In[24]:


X = df_unidos.drop(columns='Target')
y = df_unidos['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42 )


# In[25]:


# Proporciones por clases
print("\nProporción en y_train:")
print(y_train.value_counts(normalize=True))

print("\nProporción en y_test:")
print(y_test.value_counts(normalize=True))


# In[26]:


X_train = X_train.drop(columns=X_train.select_dtypes(include='object').columns, errors='ignore').copy()
X_test = X_test.drop(columns=X_test.select_dtypes(include='object').columns, errors='ignore').copy()


# In[27]:


model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
'''pred_train = model.predict(X_train)'''
prob = model.predict_proba(X_test)[:, 1]


# In[28]:


print(prob)


# In[29]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
print(cm)


# In[30]:


'''cm_train = confusion_matrix(y_train, pred_train)
print(cm_train)'''


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()


# In[32]:


from sklearn.metrics import classification_report

print(classification_report(y_test, pred))


# In[33]:


fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)

# Gráfico
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random model')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Modelo de Clasificación')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# In[34]:


'''fpr_train, tpr_train, thresholds_train = roc_curve(y_train, pred_train)
roc_auc_train = auc(fpr_train, tpr_train)
# Gráfico
plt.figure()
plt.plot(fpr_train, tpr_train, label=f'ROC curve (AUC = {roc_auc_train:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random model')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Modelo de Clasificación')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
'''


# In[35]:


# Usamos AUC como métrica
X = X.drop(columns=X.select_dtypes(include='object').columns, errors='ignore').copy()
auc_scores = cross_val_score(model, X, y, 
                             cv=10, 
                             scoring='roc_auc')  # usa 'accuracy', 'recall', etc. si prefieres otra métrica

print(f"AUC promedio: {auc_scores.mean():.3f}")
print(f"AUC por fold: {auc_scores}")


# In[36]:


joblib.dump(model, 'modelo_sinimputar.pkl')  # Puedes cambiar el nombre del archivo

