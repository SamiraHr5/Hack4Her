# Predicción de Fallas en Coolers – Arca Continental

Este repositorio contiene la solución completa para **predecir las fallas de coolers** de Arca Continental y estimar las ventas proyectadas, combinando análisis exploratorio, modelado de clasificación y un sistema de despliegue de punta a punta.

# ENTREGABLES (csv, .py, presentacion pitch):
Tambien se pueden encontrar en la siguiente [branch](https://github.com/SamiraHr5/Hack4Her/tree/entregables). 
- Pitch: https://github.com/SamiraHr5/Hack4Her/blob/main/Smart-Coolers-Nest-Pitch.pdf
- CSV: https://github.com/SamiraHr5/Hack4Her/blob/main/output.csv
- Predicción de fallas: https://github.com/SamiraHr5/Hack4Her/tree/main/Modelo_Clasificacion_Github
- Predicción de ventas: https://github.com/SamiraHr5/Hack4Her/blob/main/EDA/PrediccionVentasMayo.py

<div style="text-align: center;">
  <img width="634" alt="Screenshot 2025-06-15 at 12 33 10" src="https://github.com/user-attachments/assets/c908bae3-039d-486e-8ae8-1a9319d46d3a" />
</div>

<img width="1079" alt="Screenshot 2025-06-15 at 11 36 01" src="https://github.com/user-attachments/assets/504e4dee-c8c8-453e-9b61-e96e8b4a69b2" />

---

## Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Estructura del Repositorio](#estructura-del-repositorio)
3. [Requisitos](#requisitos)
4. [Entrenamiento y Re‐entrenamiento](#entrenamiento-y-reentrenamiento)
5. [Inferencia y Predicción](#inferencia-y-predicción)
6. [Licencia](#licencia)

---

## Visión General

- **Clasificación de Fallas (Random Forest):**  
  El modelo principal (`Modelo_Clasificacion_Github/`) utiliza un **Random Forest Classifier** entrenado sobre históricos de telemetría y tickets de servicio para anticipar **qué coolers fallarán en mayo**.  
- **Pronóstico de Ventas (Bi‑LSTM):**  
  En la carpeta `EDA/` se incluye un cuaderno con un **Bi‑LSTM secuencial** que proyecta ventas diarias, permitiendo estimar el impacto económico de los coolers defectuosos.  
- **Aplicación de Despliegue (MERN):**  
  El directorio `arca_coolers/` contiene una **API REST** con **Node.js + Express + MongoDB** y un **frontend React** para visualizar predicciones, alertas y métricas de negocio.

---

## Estructura del Repositorio

```
.
├── EDA/
│   ├── PrediccionVentasMayo.ipynb
│   └── PrediccionVentasMayo.py
├── Modelo_Clasificacion_Github/
│   ├── Archivos_Generados/
│   ├── Data_processing
|   ├── Datasets_Entrenamiento
│   ├── Datasets_Validacion
│   └── modelo_clasificacion
├── arca_coolers/
│   ├── backend/
│   │   ├── src/
│   │   └── package.json
│   └── frontend/
│       ├── src/
│       ├── public/
│       └── package.json
└── README.md
```

---

## Requisitos

| Componente        | Versión mínima | Notas                                |
|-------------------|---------------|--------------------------------------|
| Python            | 3.9           | `conda` o `pyenv` recomendado        |
| Node.js / npm     | 18.x          | Para backend y frontend              |
| MongoDB           | 6.x           | Atlas o instancia local              |
| pip packages      | ver `requirements.txt` | Incluye `scikit‑learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow` (*para Bi‑LSTM*) |
| Yarn (opcional)   | latest        | Alternativa a `npm`                  |

Instala dependencias de Python:
```bash
conda env create -f environment.yml
conda activate arca-coolers
```

Instala dependencias de Node:
```bash
cd arca_coolers/backend
npm install

cd ../frontend
npm install
```

---

## Entrenamiento y Re‑entrenamiento

El script `train_random_forest.py` expone hiperparámetros tales como:
- `n_estimators`
- `max_depth`
- `class_weight`
- `random_state`

Para re‑entrenar con nuevos datos:
```bash
python Modelo_Clasificacion_Github/train_random_forest.py   --data data/processed/train.parquet   --n_estimators 400   --max_depth 25   --save-model rf_cooler_failure_YYYYMM.pkl
```

---

## Inferencia y Predicción

```bash
python Modelo_Clasificacion_Github/inference.py   --model rf_cooler_failure.pkl   --input data/processed/2025-05.parquet   --output predictions/pred_may_2025.csv
```

El script devuelve:
- **probabilidad de falla** (`p_fail`)
- **clasificación binaria** (1 = Falla, 0 = OK)

---

## Despliegue (Backend + Frontend)

### Backend (Node.js + Express) / MONGO DB como base de datos

1. Copia `.env.example` a `.env` y define:  
   ```env
   MONGO_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/coolers
   PORT=5000
   ```
2. Inicia el servidor:
   ```bash
   cd arca_coolers/backend
   npm run dev
   ```
3. Rutas principales:  
   | Método | Ruta              | Descripción               |
   |--------|-------------------|---------------------------|
   | GET    | `/api/predictions`| Lista de predicciones     |
   | POST   | `/api/predictions`| Carga nuevas predicciones |

### Frontend (React + Vite)

```bash
cd arca_coolers/frontend
npm start
```
La interfaz muestra:
- Tablas Heatmap de fallas proyectadas.  
- Gráficos de ventas vs. fallas.  
- Filtros por región y estatus de mantenimiento.

---




## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta `LICENSE` para más información.
