# 🌍 AirPulse AI

### Intelligent Global Air Quality Monitoring, Prediction & Classification

AirPulse AI is an interactive machine learning application for monitoring, analyzing, predicting, and classifying air quality across multiple countries and cities.

The application combines data preprocessing, exploratory data analysis, machine learning, interactive visualization, and Streamlit application development into a single platform.

---

## 🚀 Features

- 📊 Interactive global air-quality dashboard
- 🌍 Country and city-level AQI analysis
- 🔎 Dynamic filtering by country, city, and month
- 📈 Interactive Plotly visualizations
- 🤖 AQI prediction using XGBoost Regressor
- 🏷️ AQI classification using XGBoost Classifier
- 📋 Model performance evaluation
- 🔥 Confusion matrix and classification report
- 📂 Dataset preview and information
- ℹ️ Dedicated project information page
- ⚙️ Reusable preprocessing pipeline with saved encoders and scaler
- 💾 Serialized machine learning models using Joblib

---

## 📌 Project Highlights

- Analyzed **18,000 air-quality records** across multiple countries and cities.
- Built an end-to-end machine learning pipeline for AQI prediction and classification.
- Developed an interactive Streamlit application for environmental analytics.
- Implemented reusable preprocessing objects including country/city encoders and feature scaling.
- Integrated trained XGBoost models into the Streamlit application for real-time predictions.
- Evaluated models using regression and classification performance metrics.

---

## 🧠 Machine Learning

### AQI Regression

The project uses an **XGBoost Regressor** to predict numerical AQI values from environmental and location-based features.

#### Evaluation Results

| Metric | Score |
|---|---:|
| MAE | 0.6286 |
| RMSE | 1.3618 |
| R² | 0.9969 |

### AQI Classification

The project uses an **XGBoost Classifier** to classify AQI into four categories:

- 🟢 Good
- 🟡 Moderate
- 🟠 Unhealthy
- 🔴 Hazardous

**Classification Accuracy: 99.58%**

The classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 📊 Dataset

The project uses a global air-quality dataset containing **18,000 records** across multiple countries and cities.

### Dataset Features

| Feature | Description |
|---|---|
| `country` | Country where the measurement was recorded |
| `city` | City where the measurement was recorded |
| `latitude` | Geographic latitude |
| `longitude` | Geographic longitude |
| `pm25` | PM2.5 concentration |
| `pm10` | PM10 concentration |
| `no2` | Nitrogen dioxide concentration |
| `so2` | Sulfur dioxide concentration |
| `o3` | Ozone concentration |
| `co` | Carbon monoxide concentration |
| `temperature` | Temperature measurement |
| `humidity` | Humidity measurement |
| `wind_speed` | Wind speed measurement |
| `hour` | Extracted hour from timestamp |
| `day` | Extracted day from timestamp |
| `month` | Extracted month from timestamp |
| `aqi` | Air Quality Index |

---

## ⚙️ Data Processing

The preprocessing pipeline includes:

1. Timestamp conversion
2. Time-based feature extraction
3. Country encoding
4. City encoding
5. Feature scaling
6. Train-test splitting
7. AQI category generation
8. Preparation of model-ready feature sets

The preprocessing objects are saved and reused by the application to ensure that prediction inputs follow the same transformation process used during model training.

---

## 🤖 Model Pipeline

The overall machine learning workflow consists of:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Encoding & Scaling
     │
     ▼
Train-Test Split
     │
     ├───────────────┐
     ▼               ▼
Regression      Classification
     │               │
     ▼               ▼
XGBoost          XGBoost
Regressor        Classifier
     │               │
     ▼               ▼
Predicted AQI   AQI Category
```

---

## 📈 Application Pages

### 🏠 Home

Provides an overview of AirPulse AI along with dataset statistics and application information.

### 📊 Dashboard

Provides interactive air-quality analytics through:

- Country filtering
- City filtering
- Month filtering
- AQI distribution
- Top polluted cities
- Country-wise AQI analysis
- KPI metrics

### 🤖 AQI Prediction

Allows users to enter environmental conditions and obtain a predicted AQI value using the trained XGBoost regression model.

### 🏷️ AQI Classification

Classifies the predicted/entered environmental conditions into one of four AQI categories:

```text
Good
Moderate
Unhealthy
Hazardous
```

### 📈 Model Performance

Displays:

- Regression metrics
- Classification metrics
- Confusion matrix
- Classification report

### ℹ️ About

Provides information about:

- Project overview
- Key features
- Machine learning models
- Technologies used
- Project highlights
- Project goals

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Visualization

- Plotly

### Web Application

- Streamlit

### Model Serialization

- Joblib

### Development & Experimentation

- Jupyter Notebook

---

## 📁 Project Structure

```text
AirPulse AI/
│
├── dashboard/
│   ├── assets/
│   │
│   ├── pages/
│   │   ├── about.py
│   │   ├── classification.py
│   │   ├── dashboard.py
│   │   ├── home.py
│   │   ├── performance.py
│   │   └── prediction.py
│   │
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── dashboard_data.csv
│   ├── globalAirQuality.csv
│   └── processed_air_quality.csv
│
├── models/
│   ├── city_encoder.pkl
│   ├── country_encoder.pkl
│   ├── scaler.pkl
│   ├── xgboost_classifier.pkl
│   └── xgboost_regressor.pkl
│
├── notebooks/
│   ├── advanced_modeling.ipynb
│   ├── classification.ipynb
│   ├── eda.ipynb
│   ├── modeling.ipynb
│   └── preprocessing.ipynb
│
├── outputs/
├── reports/
│
├── generate_dashboard_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Run Locally

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Navigate into the project directory:

```bash
cd "AirPulse AI"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

Navigate to the dashboard directory:

```bash
cd dashboard
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Required Dependencies

The application uses the following Python packages:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
joblib
```

These dependencies are listed in `requirements.txt`.

---

## 📊 Model Results

### Regression

The XGBoost regression model achieved:

```text
MAE  : 0.6286
RMSE : 1.3618
R²   : 0.9969
```

### Classification

The XGBoost classification model achieved:

```text
Accuracy : 99.58%
```

The classification pipeline additionally evaluates precision, recall, F1-score, confusion matrix, and classification report.

---

## 🔮 Future Scope

Potential future improvements include:

- 🌐 Real-time air-quality API integration
- 📡 Live AQI monitoring
- 🔄 Automated model retraining
- 📅 Historical AQI trend analysis
- 📈 Advanced AQI forecasting
- 🚨 Automated pollution alerts
- 📍 Location-based air-quality monitoring
- ☁️ Cloud deployment
- 🔍 Advanced anomaly detection
- 🧠 Experimentation with additional forecasting and machine learning models

---

## 🎯 Project Goal

The goal of AirPulse AI is to provide an integrated platform for exploring air-quality data and applying machine learning techniques to AQI prediction and classification through an accessible interactive interface.

---

## 👨‍💻 Project Information

**Project:** AirPulse AI  
**Domain:** Data Science & Machine Learning  
**Application:** Air Quality Monitoring & Prediction  
**Framework:** Streamlit  
**Primary ML Models:** XGBoost Regressor & XGBoost Classifier