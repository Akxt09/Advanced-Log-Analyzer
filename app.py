from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime, timedelta
from plotly.utils import PlotlyJSONEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
import logging
from functools import wraps
import time
import hashlib
from collections import Counter, defaultdict
import re
from scipy import stats
import seaborn as sns
from textblob import TextBlob
import geoip2.database
import geoip2.errors
from user_agents import parse
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# ────────────────────────────────────────────────────────────────────────────────
# Advanced App Configuration
# ────────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'log', 'txt', 'csv', 'tsv'}

# ────────────────────────────────────────────────────────────────────────────────
# Performance Monitoring Decorator
# ────────────────────────────────────────────────────────────────────────────────
def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# ────────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────────────────────
def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def to_native(o):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(o, dict):
        return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_native(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if pd.isna(o):
        return None
    return o

def parse_user_agent(ua_string):
    """Enhanced user agent parsing"""
    try:
        ua = parse(ua_string)
        return {
            'browser': ua.browser.family,
            'browser_version': ua.browser.version_string,
            'os': ua.os.family,
            'device': ua.device.family,
            'is_mobile': ua.is_mobile,
            'is_bot': ua.is_bot
        }
    except:
        return {'browser': 'Unknown', 'os': 'Unknown', 'device': 'Unknown', 
                'is_mobile': False, 'is_bot': False}

def extract_ip_features(ip):
    """Extract advanced features from IP addresses"""
    try:
        # Basic IP validation and feature extraction
        parts = ip.split('.')
        if len(parts) == 4:
            return {
                'is_private': (parts[0] in ['10', '192', '172']),
                'first_octet': int(parts[0]),
                'subnet_class': 'A' if int(parts[0]) < 128 else 'B' if int(parts[0]) < 192 else 'C'
            }
    except:
        pass
    return {'is_private': False, 'first_octet': 0, 'subnet_class': 'Unknown'}

# ────────────────────────────────────────────────────────────────────────────────
# Advanced Data Processing Pipeline
# ────────────────────────────────────────────────────────────────────────────────
class AdvancedLogAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
    @monitor_performance
    def parse_log_file(self, path):
        """Enhanced log file parsing with multiple format support"""
        try:
            # Try multiple common log formats
            formats = [
                # Apache Common Log Format
                {'sep': '\t', 'names': ['date', 'time', 'ip', 'method', 'uri', 'status', 'bytes', 'time-taken', 'Referer', 'User-Agent', 'Cookie']},
                # Extended format
                {'sep': ' ', 'names': ['ip', 'identity', 'user', 'timestamp', 'method', 'uri', 'protocol', 'status', 'bytes', 'referer', 'user_agent']},
                # Custom format with more fields
                {'sep': '\t', 'names': ['timestamp', 'ip', 'method', 'uri', 'status', 'bytes', 'response_time', 'referer', 'user_agent']}
            ]
            
            df = None
            for fmt in formats:
                try:
                    df = pd.read_csv(path, sep=fmt['sep'], header=None, names=fmt['names'], 
                                   on_bad_lines='skip', encoding='utf-8', low_memory=False)
                    if len(df) > 0:
                        break
                except:
                    continue
            
            if df is None or len(df) == 0:
                raise ValueError("Could not parse log file with any known format")
            
            return self._standardize_columns(df)
            
        except Exception as e:
            logger.error(f"Error parsing log file: {str(e)}")
            raise
    
    def _standardize_columns(self, df):
        """Standardize column names and create datetime"""
        # Handle different datetime formats
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            # Try to infer datetime from first column
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        
        # Standardize column names
        column_mapping = {
            'User-Agent': 'user_agent',
            'Referer': 'referer',
            'time-taken': 'response_time'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Remove rows with invalid datetime
        df = df[df['datetime'].notna()]
        
        # Ensure essential columns exist
        if 'ip' not in df.columns:
            raise ValueError("IP address column not found")
        if 'status' not in df.columns:
            raise ValueError("Status code column not found")
            
        return df
    
    @monitor_performance
    def extract_advanced_features(self, df):
        """Extract comprehensive features from log data"""
        logger.info("Extracting advanced features...")
        
        # Basic temporal features
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Status code analysis
        df['status'] = pd.to_numeric(df['status'], errors='coerce')
        df['is_error'] = (df['status'] >= 400).astype(int)
        df['is_server_error'] = (df['status'] >= 500).astype(int)
        df['is_client_error'] = ((df['status'] >= 400) & (df['status'] < 500)).astype(int)
        df['is_success'] = (df['status'] < 400).astype(int)
        
        # Bytes analysis
        if 'bytes' in df.columns:
            df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0)
            df['bytes_log'] = np.log1p(df['bytes'])
        
        # Response time analysis
        if 'response_time' in df.columns:
            df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce').fillna(0)
            df['response_time_log'] = np.log1p(df['response_time'])
        
        # URI analysis
        if 'uri' in df.columns:
            df['uri_length'] = df['uri'].str.len().fillna(0)
            df['has_query_params'] = df['uri'].str.contains(r'\?', na=False).astype(int)
            df['uri_depth'] = df['uri'].str.count('/').fillna(0)
            df['file_extension'] = df['uri'].str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna('none')
        
        # User agent analysis
        if 'user_agent' in df.columns:
            ua_features = df['user_agent'].apply(parse_user_agent)
            df['browser'] = ua_features.apply(lambda x: x['browser'])
            df['os'] = ua_features.apply(lambda x: x['os'])
            df['is_mobile'] = ua_features.apply(lambda x: x['is_mobile']).astype(int)
            df['is_bot'] = ua_features.apply(lambda x: x['is_bot']).astype(int)
        
        # IP analysis
        ip_features = df['ip'].apply(extract_ip_features)
        df['is_private_ip'] = ip_features.apply(lambda x: x['is_private']).astype(int)
        df['ip_first_octet'] = ip_features.apply(lambda x: x['first_octet'])
        
        return df
    
    @monitor_performance
    def perform_advanced_clustering(self, df):
        """Advanced clustering with multiple algorithms"""
        logger.info("Performing advanced clustering analysis...")
        
        # Aggregate features by IP
        ip_features = df.groupby('ip').agg({
            'hour': ['mean', 'std', 'min', 'max'],
            'dayofweek': ['mean', 'std'],
            'is_error': ['mean', 'sum'],
            'is_success': 'sum',
            'bytes': ['mean', 'sum'] if 'bytes' in df.columns else ['count'],
            'response_time': ['mean', 'max'] if 'response_time' in df.columns else ['count'],
            'uri_length': 'mean' if 'uri_length' in df.columns else 'size',
            'is_mobile': 'mean' if 'is_mobile' in df.columns else 'size',
            'is_bot': 'mean' if 'is_bot' in df.columns else 'size'
        }).fillna(0)
        
        # Flatten column names
        ip_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                              for col in ip_features.columns.values]
        
        if len(ip_features) < 4:
            return self._create_minimal_cluster_viz(ip_features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(ip_features)
        
        # Determine optimal number of clusters
        optimal_clusters = self._find_optimal_clusters(X_scaled)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        ip_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Apply DBSCAN for outlier detection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        ip_features['outlier_cluster'] = dbscan.fit_predict(X_scaled)
        
        # Create comprehensive visualization
        return self._create_advanced_cluster_viz(ip_features, X_scaled)
    
    def _find_optimal_clusters(self, X, max_clusters=8):
        """Find optimal number of clusters using elbow method and silhouette score"""
        if len(X) < 4:
            return 2
        
        max_clusters = min(max_clusters, len(X) - 1)
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        return silhouette_scores.index(max(silhouette_scores)) + 2
    
    def _create_advanced_cluster_viz(self, ip_features, X_scaled):
        """Create advanced cluster visualizations"""
        # PCA for visualization
        pca_2d = PCA(n_components=2)
        X_pca = pca_2d.fit_transform(X_scaled)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Distribution', 'PCA Visualization', 
                          'Cluster Characteristics', 'Outlier Detection'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Cluster distribution
        cluster_counts = ip_features['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=[f'Cluster {i}' for i in cluster_counts.index], 
                   y=cluster_counts.values, name='Distribution'),
            row=1, col=1
        )
        
        # PCA scatter
        colors = px.colors.qualitative.Set1
        for cluster in sorted(ip_features['cluster'].unique()):
            mask = ip_features['cluster'] == cluster
            fig.add_trace(
                go.Scatter(x=X_pca[mask, 0], y=X_pca[mask, 1],
                          mode='markers', name=f'Cluster {cluster}',
                          marker=dict(color=colors[cluster % len(colors)])),
                row=1, col=2
            )
        
        # Cluster characteristics heatmap
        cluster_summary = ip_features.groupby('cluster').mean()
        numeric_cols = cluster_summary.select_dtypes(include=[np.number]).columns[:8]
        
        fig.add_trace(
            go.Heatmap(z=cluster_summary[numeric_cols].values,
                      x=numeric_cols, y=[f'Cluster {i}' for i in cluster_summary.index],
                      colorscale='Viridis', name='Characteristics'),
            row=2, col=1
        )
        
        # Outlier detection (DBSCAN results)
        outlier_colors = ['red' if x == -1 else 'blue' for x in ip_features['outlier_cluster']]
        fig.add_trace(
            go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                      mode='markers', name='Outliers',
                      marker=dict(color=outlier_colors)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Advanced Clustering Analysis")
        return fig, ip_features.reset_index()
    
    def _create_minimal_cluster_viz(self, ip_features):
        """Create minimal visualization when insufficient data"""
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for clustering analysis<br>Need at least 4 unique IPs",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title='Clustering Analysis')
        return fig, ip_features.reset_index() if not ip_features.empty else pd.DataFrame()
    
    @monitor_performance
    def perform_anomaly_detection(self, df):
        """Advanced anomaly detection with multiple algorithms"""
        logger.info("Performing advanced anomaly detection...")
        
        # Create multiple time series aggregations
        time_series = {
            'hourly': df.set_index('datetime').resample('H').agg({
                'ip': 'count',
                'is_error': 'sum',
                'bytes': 'sum' if 'bytes' in df.columns else 'count',
                'response_time': 'mean' if 'response_time' in df.columns else 'count'
            }).rename(columns={'ip': 'requests'}),
            
            'daily': df.set_index('datetime').resample('D').agg({
                'ip': 'count',
                'is_error': 'sum',
                'bytes': 'sum' if 'bytes' in df.columns else 'count'
            }).rename(columns={'ip': 'requests'})
        }
        
        # Focus on hourly data for main analysis
        hourly = time_series['hourly'].fillna(0)
        
        if len(hourly) < 2:
            return self._create_minimal_anomaly_viz()
        
        # Multiple anomaly detection methods
        anomalies = {}
        
        # 1. Isolation Forest
        if hourly['requests'].nunique() > 1:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies['isolation_forest'] = iso_forest.fit_predict(hourly[['requests']])
        
        # 2. Statistical anomaly detection (Z-score)
        z_scores = np.abs(stats.zscore(hourly['requests']))
        anomalies['statistical'] = (z_scores > 3).astype(int) * -1 + 1
        
        # 3. Moving average based detection
        window = min(24, len(hourly) // 4)
        if window > 1:
            rolling_mean = hourly['requests'].rolling(window=window).mean()
            rolling_std = hourly['requests'].rolling(window=window).std()
            anomalies['moving_avg'] = ((np.abs(hourly['requests'] - rolling_mean) > 
                                      2 * rolling_std).astype(int) * -1 + 1)
        
        # Combine anomaly detection results
        hourly['anomaly_score'] = 0
        for method, scores in anomalies.items():
            hourly['anomaly_score'] += (scores == -1).astype(int)
        
        hourly['is_anomaly'] = hourly['anomaly_score'] > 0
        
        return self._create_advanced_anomaly_viz(hourly)
    
    def _create_advanced_anomaly_viz(self, hourly):
        """Create comprehensive anomaly detection visualization"""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Request Volume with Anomalies', 
                          'Error Rate Over Time',
                          'Response Patterns'),
            vertical_spacing=0.08
        )
        
        # Main requests line
        fig.add_trace(
            go.Scatter(x=hourly.index, y=hourly['requests'],
                      mode='lines', name='Requests',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        
        # Anomalies
        anomaly_data = hourly[hourly['is_anomaly']]
        if not anomaly_data.empty:
            fig.add_trace(
                go.Scatter(x=anomaly_data.index, y=anomaly_data['requests'],
                          mode='markers', name='Anomalies',
                          marker=dict(color='red', size=10, symbol='x')),
                row=1, col=1
            )
        
        # Moving average
        window = min(24, len(hourly) // 4)
        if window > 1:
            fig.add_trace(
                go.Scatter(x=hourly.index, y=hourly['requests'].rolling(window=window).mean(),
                          mode='lines', name='Moving Average',
                          line=dict(color='orange', dash='dash')),
                row=1, col=1
            )
        
        # Error rate
        if 'is_error' in hourly.columns:
            error_rate = hourly['is_error'] / hourly['requests'] * 100
            error_rate = error_rate.fillna(0)
            fig.add_trace(
                go.Scatter(x=hourly.index, y=error_rate,
                          mode='lines', name='Error Rate %',
                          line=dict(color='red', width=2)),
                row=2, col=1
            )
        
        # Response time patterns (if available)
        if 'response_time' in hourly.columns:
            fig.add_trace(
                go.Scatter(x=hourly.index, y=hourly['response_time'],
                          mode='lines', name='Avg Response Time',
                          line=dict(color='green', width=2)),
                row=3, col=1
            )
        
        fig.update_layout(height=800, title_text="Advanced Anomaly Detection Analysis")
        
        return fig, {
            'total_anomalies': int(hourly['is_anomaly'].sum()),
            'anomaly_rate': float(hourly['is_anomaly'].mean() * 100),
            'max_requests': int(hourly['requests'].max()),
            'avg_requests': float(hourly['requests'].mean())
        }
    
    def _create_minimal_anomaly_viz(self):
        """Create minimal visualization when insufficient data"""
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for anomaly detection<br>Need at least 2 time periods",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title='Anomaly Detection Analysis')
        return fig, {'total_anomalies': 0, 'anomaly_rate': 0, 'max_requests': 0, 'avg_requests': 0}
    
    @monitor_performance
    def perform_advanced_forecasting(self, df):
        """Advanced forecasting with multiple models and confidence intervals"""
        logger.info("Performing advanced forecasting...")
        
        # Create time series
        hourly = df.set_index('datetime').resample('H').size().reset_index(name='requests')
        
        if len(hourly) < 48:  # Need at least 48 hours for meaningful forecast
            return self._create_minimal_forecast_viz()
        
        try:
            # Prepare data for Prophet
            prophet_data = hourly.rename(columns={'datetime': 'ds', 'requests': 'y'})
            
            # Create and fit Prophet model with additional components
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            
            # Add custom seasonalities if enough data
            if len(hourly) >= 7 * 24:  # At least a week
                model.add_seasonality(name='hourly', period=1, fourier_order=3)
            
            model.fit(prophet_data)
            
            # Create future dataframe for 48 hours
            future = model.make_future_dataframe(periods=48, freq='H')
            forecast = model.predict(future)
            
            # Create comprehensive forecast visualization
            return self._create_advanced_forecast_viz(hourly, forecast, model)
            
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            return self._create_minimal_forecast_viz()
    
    def _create_advanced_forecast_viz(self, historical, forecast, model):
        """Create comprehensive forecasting visualization"""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('48-Hour Forecast', 'Trend Components', 
                          'Seasonal Patterns', 'Forecast Accuracy'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(x=historical['datetime'], y=historical['requests'],
                      mode='lines', name='Historical',
                      line=dict(color='grey', width=2)),
            row=1, col=1
        )
        
        # Forecast line
        future_start = len(historical)
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                      mode='lines', name='Forecast',
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Confidence intervals
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                      mode='lines', name='Upper Bound',
                      line=dict(color='rgba(31,119,180,0.3)', width=1),
                      fill=None),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                      mode='lines', name='Lower Bound',
                      line=dict(color='rgba(31,119,180,0.3)', width=1),
                      fill='tonexty'),
            row=1, col=1
        )
        
        # Trend component
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['trend'],
                      mode='lines', name='Trend',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # Daily seasonality
        if 'daily' in forecast.columns:
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['daily'],
                          mode='lines', name='Daily Pattern',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Weekly seasonality (if available)
        if 'weekly' in forecast.columns:
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['weekly'],
                          mode='lines', name='Weekly Pattern',
                          line=dict(color='purple', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Advanced Forecasting Analysis")
        
        # Prepare forecast data for frontend
        forecast_data = forecast.tail(48)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        
        return fig, forecast_data
    
    def _create_minimal_forecast_viz(self):
        """Create minimal visualization when forecasting fails"""
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for forecasting<br>Need at least 48 hours of data",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title='48-Hour Request Volume Forecast')
        return fig, []

# ────────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ────────────────────────────────────────────────────────────────────────────────
analyzer = AdvancedLogAnalyzer()

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File too large; upload under 1GB"), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error="Internal server error occurred"), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@monitor_performance
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify(error="No file part"), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(error="No file selected"), 400
        
        if not allowed_file(file.filename):
            return jsonify(error="Invalid file type. Allowed: log, txt, csv, tsv"), 400

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        
        # Process the file
        result = process_advanced_data(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(to_native(result))
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify(error=f"Processing failed: {str(e)}"), 500

@monitor_performance
def process_advanced_data(filepath):
    """Main data processing pipeline"""
    try:
        # Parse log file
        df = analyzer.parse_log_file(filepath)
        logger.info(f"Parsed {len(df)} log entries")
        
        # Extract features
        df = analyzer.extract_advanced_features(df)
        
        # Perform analyses
        cluster_fig, cluster_data = analyzer.perform_advanced_clustering(df)
        anomaly_fig, anomaly_stats = analyzer.perform_anomaly_detection(df)
        forecast_fig, forecast_data = analyzer.perform_advanced_forecasting(df)
        
        # Generate comprehensive statistics
        stats = generate_comprehensive_stats(df)
        
        # Generate security insights
        security_insights = generate_security_insights(df)
        
        # Generate performance insights
        performance_insights = generate_performance_insights(df)
        
        return {
            'success': True,
            'stats': stats,
            'security_insights': security_insights,
            'performance_insights': performance_insights,
            'anomaly_stats': anomaly_stats,
            'graphs': {
                'cluster': json.dumps(cluster_fig, cls=PlotlyJSONEncoder),
                'anomaly': json.dumps(anomaly_fig, cls=PlotlyJSONEncoder),
                'forecast': json.dumps(forecast_fig, cls=PlotlyJSONEncoder)
            },
            'cluster_data': cluster_data.head(20).to_dict(orient='records') if not cluster_data.empty else [],
            'forecast_data': forecast_data[:24]  # Show next 24 hours
        }
        
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        return {'success': False, 'error': str(e)}

def generate_comprehensive_stats(df):
    """Generate comprehensive statistics from the log data"""
    stats = {
        'total_requests': int(len(df)),
        'unique_ips': int(df['ip'].nunique()),
        'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}",
        'duration_hours': float((df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600),
        'requests_per_hour': float(len(df) / max(1, (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600)),
        
        # Error analysis
        'error_rate_pct': float(df['is_error'].mean() * 100),
        'server_error_rate_pct': float(df['is_server_error'].mean() * 100),
        'client_error_rate_pct': float(df['is_client_error'].mean() * 100),
        
        # Status code breakdown
        'status_breakdown': df['status'].value_counts().head(10).to_dict(),
        
        # Traffic patterns
        'peak_hour': int(df.groupby('hour').size().idxmax()),
        'peak_day': int(df.groupby('dayofweek').size().idxmax()),
        'weekend_traffic_pct': float(df['is_weekend'].mean() * 100),
        
        # Top resources
        'top_ips': df['ip'].value_counts().head(10).to_dict() if 'ip' in df.columns else {},
        'top_uris': df['uri'].value_counts().head(10).to_dict() if 'uri' in df.columns else {},
        'top_user_agents': df['user_agent'].value_counts().head(5).to_dict() if 'user_agent' in df.columns else {},
    }
    
    # Add bytes statistics if available
    if 'bytes' in df.columns and df['bytes'].sum() > 0:
        stats.update({
            'total_bytes': int(df['bytes'].sum()),
            'avg_response_size': float(df['bytes'].mean()),
            'total_bandwidth_gb': float(df['bytes'].sum() / (1024**3))
        })
    
    # Add response time statistics if available
    if 'response_time' in df.columns and df['response_time'].sum() > 0:
        stats.update({
            'avg_response_time': float(df['response_time'].mean()),
            'max_response_time': float(df['response_time'].max()),
            'response_time_95th': float(df['response_time'].quantile(0.95))
        })
    
    # Add browser/device statistics if available
    if 'is_mobile' in df.columns:
        stats.update({
            'mobile_traffic_pct': float(df['is_mobile'].mean() * 100),
            'bot_traffic_pct': float(df['is_bot'].mean() * 100) if 'is_bot' in df.columns else 0
        })
    
    return stats

def generate_security_insights(df):
    """Generate security-related insights"""
    insights = {
        'total_threats': 0,
        'threat_level': 'Low',
        'suspicious_ips': [],
        'attack_patterns': [],
        'recommendations': []
    }
    
    try:
        # Identify suspicious IPs (high error rate or unusual patterns)
        ip_stats = df.groupby('ip').agg({
            'is_error': ['count', 'sum', 'mean'],
            'datetime': ['count', 'nunique'],
            'uri': 'nunique'
        }).fillna(0)
        
        # Flatten column names
        ip_stats.columns = ['_'.join(col).strip() for col in ip_stats.columns.values]
        
        # Find suspicious IPs
        suspicious_mask = (
            (ip_stats['is_error_mean'] > 0.5) |  # High error rate
            (ip_stats['datetime_count'] > df['datetime'].nunique() * 0.1) |  # High request volume
            (ip_stats['uri_nunique'] > 100)  # Scanning behavior
        )
        
        suspicious_ips = ip_stats[suspicious_mask].index.tolist()[:10]
        insights['suspicious_ips'] = suspicious_ips
        insights['total_threats'] = len(suspicious_ips)
        
        # Determine threat level
        if len(suspicious_ips) > 10:
            insights['threat_level'] = 'High'
        elif len(suspicious_ips) > 5:
            insights['threat_level'] = 'Medium'
        
        # Common attack patterns
        if 'uri' in df.columns:
            attack_patterns = []
            
            # SQL injection attempts
            sql_patterns = df[df['uri'].str.contains(r'(union|select|insert|delete|drop|script)', 
                                                   case=False, na=False, regex=True)]
            if len(sql_patterns) > 0:
                attack_patterns.append(f"SQL Injection attempts: {len(sql_patterns)} requests")
            
            # XSS attempts
            xss_patterns = df[df['uri'].str.contains(r'(<script|javascript:|onload=)', 
                                                   case=False, na=False, regex=True)]
            if len(xss_patterns) > 0:
                attack_patterns.append(f"XSS attempts: {len(xss_patterns)} requests")
            
            # Directory traversal
            traversal_patterns = df[df['uri'].str.contains(r'(\.\./|\.\.\\)', 
                                                         case=False, na=False, regex=True)]
            if len(traversal_patterns) > 0:
                attack_patterns.append(f"Directory traversal: {len(traversal_patterns)} requests")
            
            insights['attack_patterns'] = attack_patterns
        
        # Generate recommendations
        recommendations = []
        if len(suspicious_ips) > 0:
            recommendations.append("Consider blocking or rate-limiting suspicious IP addresses")
        if df['is_error'].mean() > 0.1:
            recommendations.append("High error rate detected - review server configuration")
        if 'is_bot' in df.columns and df['is_bot'].mean() > 0.3:
            recommendations.append("High bot traffic - implement bot protection measures")
        
        insights['recommendations'] = recommendations
        
    except Exception as e:
        logger.error(f"Security analysis error: {str(e)}")
    
    return insights

def generate_performance_insights(df):
    """Generate performance-related insights"""
    insights = {
        'performance_score': 85,  # Default score
        'bottlenecks': [],
        'optimization_tips': [],
        'resource_usage': {}
    }
    
    try:
        # Response time analysis
        if 'response_time' in df.columns and df['response_time'].sum() > 0:
            avg_response = df['response_time'].mean()
            slow_requests = (df['response_time'] > avg_response * 2).sum()
            
            if avg_response > 2000:  # 2 seconds
                insights['bottlenecks'].append(f"Slow average response time: {avg_response:.0f}ms")
                insights['performance_score'] -= 20
            
            if slow_requests / len(df) > 0.1:
                insights['bottlenecks'].append(f"High number of slow requests: {slow_requests}")
                insights['performance_score'] -= 15
        
        # Error rate analysis
        error_rate = df['is_error'].mean()
        if error_rate > 0.05:  # 5% error rate
            insights['bottlenecks'].append(f"High error rate: {error_rate*100:.1f}%")
            insights['performance_score'] -= 25
        
        # Traffic pattern analysis
        hourly_requests = df.groupby('hour').size()
        peak_ratio = hourly_requests.max() / hourly_requests.mean()
        if peak_ratio > 3:
            insights['bottlenecks'].append("Uneven traffic distribution - consider load balancing")
            insights['performance_score'] -= 10
        
        # Resource usage
        if 'bytes' in df.columns:
            total_bandwidth = df['bytes'].sum() / (1024**3)  # GB
            insights['resource_usage']['bandwidth_gb'] = round(total_bandwidth, 2)
            
            if total_bandwidth > 100:  # 100GB
                insights['optimization_tips'].append("High bandwidth usage - optimize content delivery")
        
        # Generate optimization tips
        if not insights['optimization_tips']:
            if df['is_error'].mean() > 0:
                insights['optimization_tips'].append("Reduce error rates by fixing broken links and server issues")
            if 'response_time' in df.columns:
                insights['optimization_tips'].append("Implement caching to improve response times")
            insights['optimization_tips'].append("Monitor traffic patterns for capacity planning")
        
        # Ensure performance score is within bounds
        insights['performance_score'] = max(0, min(100, insights['performance_score']))
        
    except Exception as e:
        logger.error(f"Performance analysis error: {str(e)}")
    
    return insights

# ────────────────────────────────────────────────────────────────────────────────
# Additional API Endpoints
# ────────────────────────────────────────────────────────────────────────────────
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/api/stats')
def get_stats():
    """Get basic application statistics"""
    return jsonify({
        'uploads_processed': 'N/A',  # Would track in production
        'uptime': 'N/A',
        'memory_usage': 'N/A'
    })

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000, threaded=True)