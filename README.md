# Advanced Log Analyzer 2.0

A powerful, modern web-based log analysis platform featuring **VAE-based anomaly detection** and **NeuralProphet traffic forecasting**. Built with FastAPI, Polars, React, and Vite.

---

## 🚀 Features

### 🔍 Anomaly Detection
- **Variational Autoencoder (VAE)** for global anomaly detection
- **HDBSCAN Clustering** on VAE latent space for pattern grouping
- **Hybrid Explainability**: Z-score for all + SHAP for top 100 anomalies
- Detects DDoS attacks, scanning behavior, suspicious patterns
- Root cause analysis across all anomalies

### 📈 Traffic Forecasting
- **NeuralProphet** for time series forecasting
- Multi-horizon predictions (24-hour and 7-day)
- Confidence intervals and uncertainty estimates
- Automatic seasonality detection (hourly, daily, weekly)

### ⚡ Performance
- **Polars** for blazing-fast data processing (10x faster than pandas)
- Handles **474K+ rows** efficiently
- Parallel processing with lazy evaluation
- Optimized VAE training (batch_size=512, 4-6 min on CPU)

### 📊 Rich Analytics
- 50+ engineered features from raw logs
- Interactive dashboards with Plotly visualizations
- Real-time progress tracking
- Comprehensive statistics

---

## 🏗️ Architecture

```
Advanced-Log-Analyzer/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   ├── config.py          # Configuration
│   │   ├── core/              # Core analysis modules
│   │   │   ├── data_processor.py      # Polars-based data processing
│   │   │   ├── feature_engineer.py    # 50+ feature extraction
│   │   │   ├── vae_model.py           # VAE anomaly detection
│   │   │   ├── anomaly_detector.py    # VAE + HDBSCAN pipeline
│   │   │   ├── explainer.py           # Z-score + SHAP explainability
│   │   │   └── forecaster.py          # NeuralProphet forecasting
│   │   └── api/               # API layer
│   │       ├── routes.py      # API endpoints
│   │       └── schemas.py     # Pydantic models
│   └── requirements.txt
├── frontend/                   # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx            # Main dashboard
│   │   ├── main.jsx           # Entry point
│   │   └── index.css          # TailwindCSS styles
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 📦 Installation

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **4GB+ RAM** (8GB recommended for large files)

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install
```

---

## 🚀 Running the Application

### Start Backend (FastAPI)

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run with uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: **http://localhost:8000**
- API docs: **http://localhost:8000/docs**

### Start Frontend (React + Vite)

```bash
cd frontend

# Development mode
npm run dev
```

Frontend will be available at: **http://localhost:5173**

---

## 📖 Usage

### 1. Upload Log File
- Navigate to **http://localhost:5173**
- Click "Choose File" and select your log file
- Supported formats: `.log`, `.txt`, `.csv`, `.tsv`
- Expected format: **Tab-separated** with columns:
  ```
  date  time  cs-ip  cs-method  cs-uri  sc-status  sc-bytes  time-taken  cs(Referer)  cs(User-Agent)
  ```

### 2. Analyze
- Click "Analyze" button
- Processing time: 2-5 minutes for 474K rows
- Real-time progress updates

### 3. View Results
- **Basic Statistics**: Total requests, unique IPs, error rate, response time
- **Anomaly Detection**: Top anomalies with explanations
- **Root Cause Analysis**: Most common anomaly patterns
- **Traffic Forecast**: 24-hour and 7-day predictions with confidence intervals

---

## 🔧 Configuration

Edit `backend/app/config.py` to customize:

```python
# VAE Configuration
VAE_LATENT_DIM = 16          # Latent space dimension
VAE_HIDDEN_DIM = 64          # Hidden layer size
VAE_BATCH_SIZE = 512         # Batch size (optimal for 474K rows)
VAE_EPOCHS = 20              # Training epochs
VAE_LEARNING_RATE = 1e-3     # Learning rate

# Anomaly Detection
ANOMALY_THRESHOLD_PCT = 0.02  # Top 2% flagged as anomalies

# HDBSCAN Clustering
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 20

# Explainability
TOP_N_SHAP_EXPLANATIONS = 100  # SHAP for top 100 anomalies
N_FEATURES_TO_EXPLAIN = 5      # Top 5 features per anomaly

# Forecasting
FORECAST_HORIZONS = [24, 168]  # 24h and 7d (168h)
```

---

## 🧪 Example Data Format

```tsv
date        time        cs-ip          cs-method  cs-uri                                      sc-status  sc-bytes  time-taken  cs(Referer)                cs(User-Agent)
28-11-2018  04:30:42    61.1.64.51     GET        /www.isro.gov.in/about-isro/...            200        10117     1           https://www.google.co.in/  Mozilla/5.0...
28-11-2018  04:30:43    61.1.64.51     GET        /www.isro.gov.in/career?...                200        13281     22          https://www.google.co.in/  Mozilla/5.0...
```

---


**Built with ❤️ for data engineers, security analysts, and DevOps teams**
