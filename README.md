<div align="center">

# Stock Price Prediction System

</div>

<div align="center">
  <img src="https://res.cloudinary.com/dckoipgrs/image/upload/v1760970818/AIML_wduphf.png" alt="Apple Stock Prediction Banner" width="100%"/>
</div>

---

##  Jupyter Notebook
<div align="center">
  <img src="https://res.cloudinary.com/dckoipgrs/image/upload/v1760968107/Screenshot_2025-10-20_at_19.14.48_sdwrza.png" alt="Jupyter Notebook Preview" width="100%"/>
</div>

---

##  Streamlit App Interface with Real-Time Data Updates
<div align="center">
  <br/>
   <strong> ⚡ Fetch real-time market data with Yahoo Finance API</strong>
  <img src="https://res.cloudinary.com/dckoipgrs/image/upload/v1760968107/Screenshot_2025-10-20_at_19.15.33_or1luv.png" alt="Streamlit App Interface with RealTime Data Update" width="100%"/>
  <br/><br/>
 
</div>

---

##  Streamlit App Interface Prediction
<div align="center">
  <br/>
   <strong> ⚡ Fetch real-time market data with Yahoo Finance API</strong>
  <img src="https://res.cloudinary.com/dckoipgrs/image/upload/v1761241018/Screenshot_2025-10-23_at_23.05.45_crlfmx.png" alt="Streamlit App Interface with RealTime Data Update" width="100%"/>
  <br/><br/>
 
</div>

---


<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)]()
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?style=for-the-badge&logo=jupyter)]()
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?style=for-the-badge&logo=streamlit)]()


</div>

---

## Overview

This academic project develops a **machine learning–based prediction system** to estimate **Apple Inc.’s next-day closing stock price** using 44+ years of historical trading data.  
The system integrates **Jupyter Notebook** for model experimentation and **Streamlit** for an interactive user interface.

---

## Project Goals

- Apply machine learning to real-world financial data.  
- Build a complete data pipeline: preprocessing → training → evaluation → deployment.  
- Enable user interaction and scenario testing via a web app interface.  
- Demonstrate professional workflow integration using **Python**, **scikit-learn**, and **Streamlit**.  

---

## Tech Stack

### Core Technologies
- **Python 3.x**
- **pandas**, **numpy** – Data preprocessing
- **scikit-learn**, **XGBoost** – Model development
- **matplotlib**, **seaborn** – Visualization
- **Streamlit** – Interactive app deployment

### Tools
- **Jupyter Notebook** – Model training and evaluation (`Apple.ipynb`)
- **pickle** – Model serialization (`stock_model.pkl`, `scaler.pkl`)
- **Streamlit App** – Deployed interface (`streamlit_app.py`)

---

## Project Structure
```
├── Apple.ipynb          # Jupyter Notebook for ML pipeline
├── streamlit_app.py     # Streamlit web app
├── apple_stock.csv      # Historical dataset (1980–2025)
├── stock_model.pkl      # Trained Linear Regression model
├── scaler.pkl           # Feature scaler
├── Report.pdf           # Full academic report
├── LICENSE
```


---

## Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/apple-stock-prediction.git
cd apple-stock-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 4. Explore in Jupyter Notebook
Launch `Apple.ipynb` to review model development, training process, and evaluation.

---

## Dataset Reference
- **Source:** [Kaggle – Apple Stock Data 2025](https://www.kaggle.com/)
- **Period:** 1980–2025 (44+ years)
- **Market Cap Reference:** Apple Inc. – $3.681 Trillion (January 2025)

---

## ⚠️ Academic Notice
This repository is part of an academic research project created for **educational use only**.

- It is **not intended** for commercial trading or financial advice.
- All datasets and results are used solely for research and demonstration purposes.
