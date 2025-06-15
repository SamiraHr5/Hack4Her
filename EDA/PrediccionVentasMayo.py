#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
coolers = pd.read_csv('/Users/marielalvarez/Downloads/h4h/Hack4Her/datos/Dataset _ Coolers AC/coolers.csv')
calendar = pd.read_csv('/Users/marielalvarez/Downloads/h4h/Hack4Her/datos/Dataset _ Coolers AC/calendar.csv')
sales = pd.read_csv('/Users/marielalvarez/Downloads/h4h/Hack4Her/datos/Dataset _ Coolers AC/sales.csv')
warnings = pd.read_csv('/Users/marielalvarez/Downloads/h4h/Hack4Her/datos/Dataset _ Coolers AC/warnings.csv')

# # Coolers

# In[111]:


coolers['warning'] = coolers['cooler_id'].isin(warnings['cooler_id']).astype(int)

# In[112]:


coolers

# In[113]:


coolers.cooler_id.nunique()

# In[114]:


import seaborn as sns
sns.histplot(coolers.temperature)

# ### buscando outliers

# In[929]:


coolers.describe()

# ### identificando valores nulos

# In[130]:


coolers.isna().sum()

# In[131]:


coolers[coolers.min_voltage.isna()]

# In[157]:


coolers.calday = coolers.calday.astype(str)

# In[158]:


coolers.calday.value_counts()

# ### separando dataframes entre los que han fallado y los que no

# In[132]:


falla = coolers[coolers.warning == 1]

# In[133]:


funciona = coolers[coolers.warning == 0]

# In[134]:


sns.boxplot(data = falla, x = 'temperature') # celsius

# In[135]:


sns.boxplot(data = funciona, x = 'temperature') # celsius

# ### comportamientos durante el tiempo / entendiendo el dataset

# In[139]:



falla['calday'] = pd.to_datetime(falla['calday'].astype(str), format='%Y%m%d')

falla['year'] = falla['calday'].dt.year
falla['month'] = falla['calday'].dt.month

monthly_means = falla.groupby(['year', 'month']).mean(numeric_only=True).reset_index()

features = ['door_opens', 'open_time', 'compressor', 'power', 
            'on_time', 'min_voltage', 'max_voltage', 'temperature']

for feature in features:
    plt.figure(figsize=(10, 5))
    
    for year in sorted(monthly_means['year'].unique()):
        subset = monthly_means[monthly_means['year'] == year]
        plt.plot(subset['month'], subset[feature], marker='o', label=f'Año {year}')
    
    plt.title(f'Comportamiento mensual de {feature}')
    plt.xlabel('Mes')
    plt.ylabel(f'{feature}')
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True)
    plt.show()


# In[149]:


import pandas as pd
import matplotlib.pyplot as plt

def plot_cooler_features_by_month(df, cooler_id):
    """
    Genera gráficas por mes y año para cada feature de un cooler específico.

    :param df: DataFrame con los datos
    :param cooler_id: ID del cooler a analizar
    """
    df_cooler = df[df['cooler_id'] == cooler_id].copy()
    
    if df_cooler.empty:
        print(f"No se encontraron datos para cooler_id: {cooler_id}")
        return
    
    df_cooler['calday'] = pd.to_datetime(df_cooler['calday'], errors='coerce', infer_datetime_format=True)
    
    if df_cooler['calday'].isnull().any():
        print("Advertencia: algunas fechas no se pudieron convertir y se omitirán.")
        df_cooler = df_cooler.dropna(subset=['calday'])
    
    df_cooler['year'] = df_cooler['calday'].dt.year
    df_cooler['month'] = df_cooler['calday'].dt.month
    
    monthly_means = df_cooler.groupby(['year', 'month']).mean(numeric_only=True).reset_index()
    
    features = ['door_opens', 'open_time', 'compressor', 'power', 
                'on_time', 'min_voltage', 'max_voltage', 'temperature']
    
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


# In[161]:


coolers.columns

# In[156]:


plot_cooler_features_by_month(falla, '51acd7bdf1c17f571df58541440547e2bf6e30e322cd1c9dee9bbb7872c6474d')

# In[166]:



def plot_cooler_features_by_day(df, cooler_id):
    """
    Genera gráficas por día del mes y año para cada feature de un cooler específico.

    :param df: DataFrame con los datos
    :param cooler_id: ID del cooler a analizar
    """
    df_cooler = df[df['cooler_id'] == cooler_id].copy()
    
    if df_cooler.empty:
        print(f"No se encontraron datos para cooler_id: {cooler_id}")
        return
    
    df_cooler['calday'] = pd.to_datetime(df_cooler['calday'], errors='coerce', infer_datetime_format=True)
    
    if df_cooler['calday'].isnull().any():
        print("Advertencia: algunas fechas no se pudieron convertir y se omitirán.")
        df_cooler = df_cooler.dropna(subset=['calday'])
    
    df_cooler['year'] = df_cooler['calday'].dt.year
    df_cooler['month'] = df_cooler['calday'].dt.month
    df_cooler['day'] = df_cooler['calday'].dt.day
    
    daily_means = df_cooler.groupby(['year', 'month', 'day']).mean(numeric_only=True).reset_index()
    
    features = ['door_opens', 'open_time', 'compressor', 'power', 
                'on_time', 'min_voltage', 'max_voltage', 'temperature']
    
    for feature in features:
        plt.figure(figsize=(12, 6))
        
        for (year, month), group in daily_means.groupby(['year', 'month']):
            plt.plot(group['day'], group[feature], marker='o', label=f'{year}-{month:02d}')
        
        plt.title(f'Comportamiento diario de {feature} (Cooler {cooler_id})')
        plt.xlabel('Día del mes')
        plt.ylabel(feature)
        plt.xticks(range(1, 32))
        plt.legend(title='Año-Mes')
        plt.grid(True)
        plt.show()


# In[167]:


plot_cooler_features_by_day(falla, '51acd7bdf1c17f571df58541440547e2bf6e30e322cd1c9dee9bbb7872c6474d')

# # Sales

# In[389]:


sales = pd.read_csv('/Users/marielalvarez/Downloads/h4h/Hack4Her/datos/Dataset _ Coolers AC/sales.csv')


# In[390]:


sales.cooler_id.value_counts()

# In[404]:


sales.amount.describe()

# In[395]:


sales.calmonth.value_counts()

# In[391]:


sales.customer_id.nunique()

# In[401]:


sales_clean = sales.drop(columns = ['customer_id'])

# In[402]:


sales_clean

# In[441]:


sales.cooler_id.unique()

# In[440]:



def plot_cooler_amount_over_time(sales, cooler_id):
    # Filtrar por cooler_id
    cooler_data = sales[sales['cooler_id'] == cooler_id].copy()
    
    # Ordenar por calmonth
    cooler_data = cooler_data.sort_values('calmonth')
    
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(cooler_data['calmonth'].astype(str), cooler_data['amount'], marker='o')
    plt.title(f'Amount over Time for Cooler {cooler_id}')
    plt.xlabel('Calmonth')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[443]:


plot_cooler_amount_over_time(sales,'661cc6a46677515e7801eda42d84b6242778a9d77f6062a9d9bb536f57c55e44')

# # Generación de modelo de regresión secuencial para pronóstico (forecasting)

# In[930]:


import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = sales_clean.rename(columns={'cooler_id':'id', 'date':'ds', 'amount':'y'})
df = df.sort_values(['id', 'ds'])

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit   

WINDOW, HORIZON = 12, 1

X, y, groups = [], [], [] 
for cid, grp in df.groupby('id'):
    series = grp['y'].values
    for i in range(len(series) - WINDOW - HORIZON + 1):
        X.append(series[i:i+WINDOW])
        y.append(series[i+WINDOW:i+WINDOW+HORIZON])
        groups.append(cid)                   

X = np.array(X)[:, :, None]
y = np.array(y)

splitter = GroupShuffleSplit(  
    n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, groups=groups))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# 3) scalers para train y test 
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train.reshape(-1,1)).reshape(X_train.shape)
X_val   = x_scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)

y_scaler = StandardScaler()
y_train  = y_scaler.fit_transform(y_train)
y_val    = y_scaler.transform(y_val)


# In[932]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

def safe_mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100

# crossvalidation
df_bt = df[df['ds'] <= '2024-12-01']

X_bt, y_bt = [], []
for _, grp in df_bt.groupby('id'):
    y_series = grp['y'].values
    for i in range(len(y_series) - WINDOW - HORIZON + 1):
        X_bt.append(y_series[i : i+WINDOW])
        y_bt.append(y_series[i+WINDOW : i+WINDOW+HORIZON])

X_bt = np.array(X_bt)[:, :, None]
y_bt = np.array(y_bt)

X_bt_flat = X_bt.reshape(-1, 1)
X_bt_scaled = scaler.transform(X_bt_flat).reshape(X_bt.shape)
y_bt_scaled = scaler.transform(y_bt)



# lstm hyperparameter tunning

# In[465]:


import keras_tuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

def build_model(hp):
    """
    hp : HyperParameters object (define el espacio de búsqueda)
    devuelve un modelo Keras compilado
    """
    model = Sequential()

    # --- nº de capas LSTM ---
    n_layers = hp.Int('n_layers', 1, 3, default=1)
    for i in range(n_layers):
        units = hp.Choice(f'units_{i}', [16, 32, 64, 128], default=32)
        return_sequences = i < n_layers - 1
        model.add(
            LSTM(units,
                 return_sequences=return_sequences,
                 input_shape=(WINDOW, 1) if i == 0 else None)
        )
        # dropout opcional
        drop = hp.Float(f'dropout_{i}', 0.0, 0.4, step=0.1, default=0.0)
        if drop > 0:
            model.add(Dropout(drop))

    model.add(Dense(HORIZON))

    # --- optimizador y LR ---
    lr = hp.Float('lr', 1e-4, 5e-3, sampling='log', default=1e-3)
    opt_choice = hp.Choice('optimizer', ['adam', 'rmsprop'], default='adam')
    optimizer = Adam(learning_rate=lr) if opt_choice == 'adam' else RMSprop(learning_rate=lr)

    model.compile(loss='mae', optimizer=optimizer)
    return model
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',         
    max_epochs=50,
    factor=3,                     
    directory='lstm_tuning',
    project_name='cooler_sales'
)

from tensorflow.keras.callbacks import EarlyStopping
es_cb = EarlyStopping(patience=6, restore_best_weights=True)


# In[466]:


tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=[es_cb],
    verbose=2
)

# Mejor combinación encontrada
best_hp   = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]
print('\nMejor configuración:')
for p in best_hp.values:
    print(f"{p}: {best_hp.get(p)}")


# In[472]:


best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBest hyperparameter configuration found:")
for param in best_hp.values.keys():
    print(f"{param}: {best_hp.get(param)}")


# In[473]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

best_model = Sequential()

best_model.add(LSTM(64, input_shape=(WINDOW, 1)))
best_model.add(Dropout(0.2))

best_model.add(Dense(HORIZON))

optimizer = Adam(learning_rate=0.0023361463812106792)

best_model.compile(loss='mae', optimizer=optimizer)

# entrena best model encontrado anteriormente
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,                 
    batch_size=256,
    verbose=2
)


# In[475]:


y_best_mod_pred_scaled = best_model.predict(X_bt_scaled)
y_best_mod_pred = scaler.inverse_transform(y_bt_pred_scaled)

mae  = mean_absolute_error(y_bt, y_best_mod_pred)
rmse = math.sqrt(mean_squared_error(y_bt, y_best_mod_pred))
mape = safe_mape(y_bt.flatten(), y_best_mod_pred.flatten())
print(f"MAE {mae:.1f} | RMSE {rmse:.1f} | MAPE {mape:.1f}%")


# # probando bi-LSTM

# In[491]:


# -------- BUILD ----------
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

HORIZON = 1     
WINDOW  = 12    

bi_model = Sequential([
    Bidirectional(
        LSTM(64, return_sequences=False),  
        input_shape=(WINDOW, 1)
    ),
    Dropout(0.20),
    Dense(HORIZON)                      
])

bi_model.compile(
    loss='mae',
    optimizer=Adam(learning_rate=0.0023361463812106792)
)

bi_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=256, verbose=2
)

# -------- BACK-TEST ----------
y_bt_pred_scaled = bi_model.predict(X_bt_scaled)        

mae  = mean_absolute_error(y_bt.ravel(), y_bt_pred.ravel())
rmse = mean_squared_error(y_bt.ravel(), y_bt_pred.ravel(), squared=False)
mape = safe_mape(y_bt.ravel(), y_bt_pred.ravel())
print(mae, rmse, mape)


# In[493]:


mae  = mean_absolute_error(y_bt.ravel(), y_bt_pred.ravel())
rmse = math.sqrt(mean_squared_error(y_bt.ravel(), y_bt_pred.ravel()))
mape = safe_mape(y_bt.ravel(), y_bt_pred.ravel())
print(mae, rmse, mape)


# ### MODELO Sequential Bi-LSTM

# In[498]:


may_forecasts = []

for cid, grp in df.groupby('id'):
    last_12 = grp.tail(WINDOW)['y'].values
    if len(last_12) < WINDOW:
        continue  
    X_input = x_scaler.transform(last_12.reshape(-1,1)).reshape(1, WINDOW, 1)
    y_pred_scaled = bi_model.predict(X_input, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)[0,0]
    may_forecasts.append({'id': cid, 'forecast_may2025': y_pred})

may_forecasts_df = pd.DataFrame(may_forecasts)


# In[500]:


may_forecasts_df.forecast_may2025.describe()

# # analisis en sales causado por fallas

# In[755]:


final = pd.read_csv('Fallas_con_target.csv')

# In[756]:


final = final.drop(columns = ['Unnamed: 0'])

# In[757]:


final['calmonth'] = pd.to_datetime(final['year'].astype(str) + '-' + final['month'].astype(str).str.zfill(2) + '-01')

agg_df = final.groupby(['cooler_id', 'calmonth']).agg(
    {**{col: 'median' for col in final.select_dtypes(include=['number']).columns if col != 'Target'},
     'Target': 'max'}
).reset_index()

agg_df.head()

# In[758]:


len(agg_df)

# In[759]:


agg_df.isna().sum()

# In[760]:


agg_df.calmonth.unique()

# In[761]:


agg_df['calmonth'] = agg_df['calmonth'].astype(str)

# In[762]:


sales['calmonth'] = sales['calmonth'].astype(str)

# In[829]:


sales_completas = sales.merge(agg_df, on=['cooler_id', 'calmonth'], how='left')

sales_completas.head()


# In[830]:


sales_completas.isna().sum()

# In[633]:


merged_df.columns

# In[634]:


not_null = merged_df[~merged_df.compressor.isna()]

# In[635]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sales_vs_failures(df, cooler_id, window=3):
    """
    Para un cooler, muestra:
      A. Línea doble Amount (ventas) y Target (fallas) por mes
      B. Rolling window de Amount vs suma de fallas
      C. Scatter Amount vs Target con ajuste de regresión
    Devuelve un diccionario con estadísticas básicas.
    """
    # --- Filtrado y orden ---
    data = (df[df['cooler_id'] == cooler_id]
              .sort_values('calmonth')
              .reset_index(drop=True))
    
    if data.empty:
        raise ValueError(f"No se encontraron datos para cooler {cooler_id}")

    # --- Resumen numérico ---
    corr = data['amount'].corr(data['Target'])
    stats = {
        'cooler_id': cooler_id,
        'observations': len(data),
        'total_sales': data['amount'].sum(),
        'total_failures': data['Target'].sum(),
        'corr_amount_target': corr
    }

    # --- A) Línea doble ---
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(data['calmonth'], data['amount'],  marker='o', label='Amount')
    ax2.bar(data['calmonth'], data['Target'], alpha=0.3, color='crimson', label='Target (failures)')
    ax1.set_ylabel('Amount')
    ax2.set_ylabel('Target')
    ax1.set_title(f"Ventas vs. Fallas – Cooler {cooler_id}")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.show()

    # --- B) Rolling window (ventas media / fallas acumuladas) ---
    roll_amt   = data['amount'].rolling(window).mean()
    roll_fail  = data['Target'].rolling(window).sum()
    plt.figure(figsize=(10, 3))
    plt.plot(data['calmonth'], roll_amt, label=f'Amount MA{window}')
    plt.plot(data['calmonth'], roll_fail, label=f'Failures Σ{window}', color='crimson')
    plt.title(f"Media móvil vs. Suma móvil ({window} meses)")
    plt.legend(); plt.ylabel('Valor'); plt.show()

    # --- C) Scatter con ajuste ---
    plt.figure(figsize=(5, 4))
    sns.regplot(x='Target', y='amount', data=data, scatter_kws={'alpha':0.7})
    plt.title("Relación puntual ventas-fallas")
    plt.xlabel('Fallas (Target)'); plt.ylabel('Ventas (Amount)'); plt.tight_layout()
    plt.show()

    return stats


# In[636]:


not_null[not_null.Target != 0].sample().cooler_id.unique()

# In[734]:


not_null.cooler_id.nunique()

# In[739]:


not_null.groupby(['Target'])['amount'].describe()

# In[637]:


analyze_sales_vs_failures(not_null, cooler_id='c82638a5642ffad7911fa48ee311b2ddef42da6d56b7158513fd979ad6c37435')

# In[663]:


may_forecasts_df

# In[664]:


may_forecasts_df.to_csv('sales_predicted_may.csv')

# In[665]:


may_forecasts_df = may_forecasts_df.rename(columns={
    'id': 'cooler_id',
    'forecast_may2025': 'amount'
})

# In[669]:


may_forecasts_df['calmonth'] = '2025-05-01'

# In[680]:


total_sales = pd.concat([may_forecasts_df, sales[['cooler_id', 'amount', 'calmonth']]])

# In[683]:


total_sales.to_csv('sales_totales.csv')

# In[657]:


import pickle, joblib
bi_model.save("models/bi_lstm.h5")           # modelo
joblib.dump(x_scaler, "models/x_scaler.pkl") # scaler de entrada
joblib.dump(y_scaler, "models/y_scaler.pkl") # scaler de salida

# # Análisis de ventas en Mayo con coolers que fallarán

# In[701]:


proba = pd.read_csv('probabilidad_mayo.csv')

# In[819]:


proba = proba.drop(columns= ['Unnamed: 0'])

# In[821]:


proba.to_csv('proba_mayo.csv',index=False)

# In[727]:


total_sales['calmonth'] = pd.to_datetime(total_sales['calmonth'])

mayo_df = total_sales[
    (total_sales['calmonth'].dt.year == 2025) &
    (total_sales['calmonth'].dt.month == 5)
].copy()

otros_df = total_sales[
    ~(
        (total_sales['calmonth'].dt.year == 2025) &
        (total_sales['calmonth'].dt.month == 5)
    )
]

agg_df = (
    otros_df.groupby('cooler_id')['amount']
    .agg(media_ventas='mean', mediana_ventas='median')
    .reset_index()
)

mayo_df = mayo_df.merge(agg_df, on='cooler_id', how='left')


# In[740]:


resultados = mayo_df.merge(proba, on='cooler_id', how='inner')

high_risk_df = resultados[resultados['proba_mensual'] > 0.7].copy()

high_risk_df['diff_median_amount'] =  high_risk_df['mediana_ventas'] - high_risk_df['amount']
high_risk_sample = high_risk_df.sample(30)
plt.figure(figsize=(8,4))
plt.bar(high_risk_sample['cooler_id'], high_risk_sample['diff_median_amount'])
plt.xlabel('Cooler ID')
plt.ylabel('Mediana sin mayo - Amount mayo')
plt.title('Diferencia mediana histórica vs. amount mayo (coolers con proba > 0.5)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# In[731]:


high_risk_df['diff_median_amount'].describe()

# #### el beneficio potencial debe estar ligado a lo que se espera vender en mayo

# In[803]:


sales.amount.describe()

# In[746]:


results.beneficio_esperado.describe()

# In[856]:


summary = (
    merged
      .groupby(['cooler_id', 'Target'])['amount']
      .median()
      .unstack(fill_value=0)               
      .rename(columns={0: 'mean_amount_funciona',
                       1: 'mean_amount_falla'})
      .reset_index()
)
summary

# In[896]:


summary_fallas = summary[summary.mean_amount_falla != 0]

# In[897]:


summary_funciona = summary[summary.mean_amount_falla == 0]

# In[899]:


summary_funciona.describe()

# In[900]:


summary_fallas.describe()

# In[894]:


summary_fallas[summary_fallas.mean_amount_funciona > 500]

# In[902]:


summary_fallas['pct_error'] = (
    (summary_fallas['mean_amount_falla'] - summary_fallas['mean_amount_funciona'])
    / summary_fallas['mean_amount_funciona']
) 

# In[918]:


summary_fallas[summary_fallas.pct_error < 0].pct_error.describe()

# In[935]:


summary_fallas[summary_fallas.pct_error < 0].pct_error.hist()

# # beneficio esperado final

# In[934]:


mayo_df['beneficio_potencial'] = (
    mayo_df['media_ventas'] - mayo_df['amount']
).clip(lower=0)

results = mayo_df.merge(proba, on='cooler_id', how='inner')

results['beneficio_potencial'] = results['amount'] * 0.2
prob_recuperacion = 0.97

results['beneficio_esperado'] = (
    results['beneficio_potencial'] *
    results['proba_mensual'] * prob_recuperacion
)

total_beneficio = results['beneficio_esperado'].sum()
print(f"Beneficio esperado total: {total_beneficio:,.2f}")


# In[ ]:



