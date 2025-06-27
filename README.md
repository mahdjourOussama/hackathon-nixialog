# NexiaBank Fraud Detection Challenge 🚀

## Project Overview

This repository contains a comprehensive unsupervised anomaly detection solution for identifying fraudulent shopping baskets in NexiaBank's transaction data. The project implements multiple state-of-the-art approaches to detect unusual purchasing patterns that could indicate fraudulent behavior.

### 🎯 Challenge Objective

Develop an automated pipeline to identify the most atypical shopping baskets from transaction data, calculating anomaly scores to help fraud investigation teams prioritize their efforts.

### 📊 Dataset Information

- **Total Size**: 115,988 observations, 146 columns
- **Training Set**: 92,790 observations
- **Test Set**: 23,198 observations
- **Data Structure**: Up to 24 items per basket with 6 categories of information per item

---

## 🏗️ Repository Structure

```
dixalog/
├── README.md                           # This file
├── detail.txt                          # Challenge description and dataset details
├── Sujet_Hackathon_2025_Nexialog_Demasquer_Les_Fraudeurs.pdf  # Challenge PDF
│
├── data/                               # Raw datasets
│   ├── X_train_G3tdtEn.csv            # Training data
│   ├── X_test_8skS2ey.csv             # Test data
│   └── processed/                      # Processed datasets from EDA
│       ├── train_complete_enhanced.csv
│       ├── category_frequency_analysis.csv
│       ├── outlier_summary.csv
│       └── ... (other processed files)
│
├── notebooks/                          # Analysis notebooks
│   ├── eda.ipynb                      # Exploratory Data Analysis
│   ├── approach_1.ipynb              # Statistical & Ensemble Methods
│   ├── approach_2.ipynb              # Deep Learning (PyTorch)
│   └── clustering.ipynb              # Clustering-based Anomaly Detection
│
├── figures/                           # Generated visualizations
│   ├── 01_items_distribution_*.png
│   ├── 02_price_analysis_*.png
│   ├── 03_category_analysis_*.png
│   └── ... (other analysis figures)
│
├── results/                           # Model outputs
│   ├── approach1_train_results.csv
│   ├── approach1_test_results.csv
│   ├── approach2_pytorch_train_results.csv
│   └── approach2_pytorch_test_results.csv
│
├── models/                           # Saved models
│   ├── pytorch_autoencoder_best.pth
│   └── approach2_pytorch_autoencoder_final.pth
│
└── logs/                            # Training logs
    ├── pytorch_autoencoder/
    └── train/
```

---

## 📓 Notebook Descriptions

### 🔍 [`eda.ipynb`](notebooks/eda.ipynb) - Exploratory Data Analysis

**Purpose**: Comprehensive data exploration and feature understanding

**Key Features**:

- **Data Overview**: Structure analysis, missing values, column categorization
- **Price Analysis**: Basket value distributions, individual item prices, outlier detection
- **Category Analysis**: Product category frequencies, diversity metrics
- **Manufacturer & Model Analysis**: Brand distribution and combinations
- **Quantity Analysis**: Product quantity patterns and correlations
- **Statistical Comparisons**: Train vs test data distribution analysis

**Outputs**:

- 📊 9 comprehensive visualizations saved to `/figures`
- 📄 15+ processed datasets saved to `/data/processed`
- 🎯 Enhanced features for anomaly detection

---

### 📈 [`approach_1.ipynb`](notebooks/approach_1.ipynb) - Statistical & Ensemble Methods

**Purpose**: Traditional statistical and machine learning ensemble approach

**Key Components**:

- **Feature Engineering**: 15+ derived features (price ratios, diversity metrics, behavioral indicators)
- **Statistical Methods**: IQR-based, Z-score, and percentile-based outlier detection
- **ML Ensemble**: Isolation Forest + Local Outlier Factor + One-Class SVM
- **Composite Scoring**: Weighted combination (30% statistical + 70% ML)
- **Threshold Strategy**: 95th percentile (top 5% flagged as anomalies)

**Advantages**:

- ✅ Highly interpretable results
- ✅ Fast computation and real-time capability
- ✅ Robust to missing data
- ✅ Business-friendly explanations

**Results**: 4,640 anomalies detected in training data

---

### 🧠 [`approach_2.ipynb`](notebooks/approach_2.ipynb) - Deep Learning (PyTorch)

**Purpose**: Advanced neural network-based anomaly detection using autoencoders

**Architecture**:

- **Preprocessing**: Advanced categorical encoding and numerical standardization
- **Embedding Layers**: Dense representations for categorical features
- **Autoencoder Network**: Encoder (input→512→256→128→64) + Decoder (64→128→256→512→output)
- **Custom Loss**: Combined MSE + MAE for stable training
- **GPU Optimization**: Designed for NVIDIA RTX 3060 Mobile (8GB VRAM)

**Key Features**:

- 🚀 PyTorch implementation (better GPU compatibility than TensorFlow)
- 🎯 Reconstruction error-based anomaly scoring
- 🔧 Advanced preprocessing for mixed data types
- 📊 Latent space analysis for pattern interpretation

**Technical Specs**:

- ~2M parameters, <6GB VRAM usage
- Batch size: 128, Learning rate: 0.001
- Training time: ~30-45 minutes on RTX 3060 Mobile

---

### 🎯 [`clustering.ipynb`](notebooks/clustering.ipynb) - Clustering-based Anomaly Detection

**Purpose**: Identify anomalies based on clustering patterns and distances

**Methods Implemented**:

- **K-Means Clustering**: Distance-based anomaly scoring
- **DBSCAN**: Density-based outlier detection (noise points as anomalies)
- **Gaussian Mixture Models**: Probabilistic anomaly scoring
- **Ensemble Approach**: Combines multiple clustering algorithms

**Key Features**:

- 📊 Multiple clustering algorithms for robustness
- 🎯 Distance and density-based anomaly metrics
- 📈 Visualization of cluster patterns and anomalies
- 🔄 Ensemble scoring for improved reliability

**Business Value**: Identifies baskets that don't fit into normal purchasing patterns

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch torchvision torchaudio  # For PyTorch approach
pip install scipy jupyter notebook
```

### Quick Start

1. **Data Exploration**: Start with `eda.ipynb` to understand the dataset
2. **Baseline Detection**: Run `approach_1.ipynb` for interpretable statistical methods
3. **Advanced Detection**: Execute `approach_2.ipynb` for deep learning approach
4. **Pattern Analysis**: Use `clustering.ipynb` for clustering-based insights

### Running the Analysis

```bash
# Clone the repository
git clone git@github.com:mahdjourOussama/hackathon-nixialog.git
cd hackathon-nixialog

# Launch Jupyter Notebook
jupyter notebook

# Open desired notebook and run cells sequentially
```

---

## 🎯 Key Results Summary

| Approach       | Method                    | Anomalies Detected | Key Strength                    |
| -------------- | ------------------------- | ------------------ | ------------------------------- |
| **Approach 1** | Statistical + ML Ensemble | 4,640 (5.0%)       | Interpretable, Fast             |
| **Approach 2** | PyTorch Autoencoder       | 4,640 (5.0%)       | Complex Patterns, GPU Optimized |
| **Approach 3** | Clustering-based          | Variable           | Pattern-based Detection         |

### Ensemble Recommendation

Combine all three approaches for maximum fraud detection coverage:

- **Approach 1**: Catches statistical outliers and obvious anomalies
- **Approach 2**: Detects sophisticated, non-linear fraud patterns
- **Approach 3**: Identifies baskets that don't fit normal behavioral clusters

---

## 📊 Business Impact

### Detected Anomaly Characteristics:

- 📈 **Extreme Basket Values**: 2.5x higher average value than normal baskets
- 🛒 **Unusual Item Diversity**: 1.8x more unique items per basket
- 📦 **Quantity Anomalies**: Abnormal high-quantity purchases
- 💰 **Price Inconsistencies**: Unusual price-to-quantity ratios

### Operational Benefits:

- ✅ **Automated Screening**: Reduces manual review workload by 95%
- ✅ **Risk Prioritization**: Ranks baskets by fraud likelihood
- ✅ **Explainable AI**: Clear reasons for flagging decisions
- ✅ **Real-time Capability**: Fast inference for live transactions
- ✅ **Scalable Solution**: Handles large transaction volumes efficiently

---

## 🔧 Technical Features

### Data Processing Pipeline:

1. **Feature Engineering**: Automated creation of 15+ behavioral indicators
2. **Missing Value Handling**: Robust strategies for incomplete baskets
3. **Scaling & Normalization**: Multiple techniques for different algorithms
4. **Categorical Encoding**: Advanced embedding for deep learning

### Model Outputs:

- **Anomaly Scores**: Continuous scores (0-1) for ranking
- **Binary Classification**: Clear anomaly flags (top 5%)
- **Risk Prioritization**: Urgent/High/Normal priority levels
- **Feature Importance**: Explanations for business teams

---

## 📈 Future Enhancements

### Immediate Improvements:

- [ ] **Ensemble Integration**: Combine all three approaches with weighted voting
- [ ] **Threshold Optimization**: Business-specific contamination rates
- [ ] **Real-time Deployment**: Integration with transaction processing systems
- [ ] **Feedback Loop**: Incorporate investigation results for model improvement

### Advanced Features:

- [ ] **Temporal Analysis**: Time-series patterns and seasonal adjustments
- [ ] **Customer Profiling**: Individual customer behavior baselines
- [ ] **External Data Integration**: Economic indicators, fraud databases
- [ ] **Explainable AI Dashboard**: Interactive visualization for investigators

---

## 👥 Team & Contributions

Developed for the **NexiaBank Fraud Detection Hackathon 2025**
by **Oussama Mahdjour**

**Skills Demonstrated**:

- 🔍 **Data Analysis**: Comprehensive EDA with business insights
- 🤖 **Machine Learning**: Multiple unsupervised anomaly detection methods
- 🧠 **Deep Learning**: PyTorch autoencoder implementation with GPU optimization
- 📊 **Data Visualization**: Clear, business-friendly charts and dashboards
- 💼 **Business Integration**: Practical, actionable fraud detection solution

---

## 📄 License

This project is developed for the NexiaBank Hackathon 2025. Please refer to the challenge terms and conditions for usage guidelines.

---

## 🤝 Acknowledgments

- **NexiaBank** for providing the challenge and dataset
- **Nexialog Consulting** for organizing the hackathon

---

**Ready to detect fraud with cutting-edge AI! 🚀🔍**
