# ✈️ Aerospace Software Defect Prediction - Interactive Dashboard

An interactive Streamlit web application showcasing machine learning techniques for predicting software defects in aerospace modules using NASA datasets.

## 🎯 Project Highlights

- **99.70% Accuracy** achieved with Neural Network model
- **52% Feature Reduction** (21 → 10 features) using advanced selection methods
- **SMOTEENN** successfully balanced highly imbalanced dataset (9:1 → 0.7:1)
- **109 Defects** detected in 419 test samples with high confidence
- **Interactive Visualizations** for complete project exploration

## 📊 Features

### Dashboard Pages

1. **🏠 Home** - Project overview and key achievements
2. **⚖️ Data Preprocessing** - SMOTEENN class imbalance handling with interactive visualizations
3. **🔍 Feature Selection** - Multi-method feature selection (21 → 10 features)
4. **📊 Model Performance** - Top 3 model comparison with interactive charts
5. **🐛 Defect Analysis** - Detected defects exploration and analysis
6. **🚀 Live Prediction** - Real-time defect prediction interface
7. **📁 CM1 Dataset** - Labeled NASA dataset analysis

### Interactive Elements

- **Plotly Charts** - Hover, zoom, and pan on all visualizations
- **Real-time Predictions** - Test models with custom feature values
- **Dynamic Sliders** - Adjust parameters and see instant updates
- **Filterable Tables** - Search and sort through detected defects
- **Download Options** - Export predictions and results
- **Educational Tooltips** - Learn about techniques and metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (or navigate to project directory)
   ```bash
   cd Software_Defect_Prediction_aerospace
   ```

2. **Navigate to the Streamlit app directory**
   ```bash
   cd streamlit_app
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or if using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

3. **Explore the dashboard**
   - Use the sidebar to navigate between pages
   - Interact with visualizations, sliders, and filters
   - Try the live prediction feature with custom inputs

## 📁 Project Structure

```
streamlit_app/
│
├── app.py                          # Main application entry point
├── config.py                       # Configuration and settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── pages/                          # Streamlit pages
│   ├── 1_🏠_Home.py
│   ├── 2_⚖️_Data_Preprocessing.py
│   ├── 3_🔍_Feature_Selection.py
│   ├── 4_📊_Model_Performance.py
│   ├── 5_🐛_Defect_Analysis.py
│   ├── 6_🚀_Live_Prediction.py
│   └── 7_📁_CM1_Dataset.py
│
└── utils/                          # Utility modules
    ├── __init__.py
    ├── data_loader.py              # Data loading functions
    ├── visualizations.py           # Plotly visualization functions
    └── smoteenn_handler.py         # SMOTEENN analysis utilities
```

## 📊 Dataset Information

### Primary Dataset (Without Labels)
- **Training**: 1,676 samples
- **Testing**: 419 samples
- **Features**: 21 software complexity metrics
- **Source**: NASA aerospace module datasets from Kaggle

### CM1 Dataset (Labeled)
- **Samples**: 498 (723 after SMOTEENN)
- **Defect Rate**: 9.84% (original)
- **Purpose**: Validation and comparison
- **Source**: NASA CM1 aerospace project

### Features (10 Selected from 21)
1. HALSTEAD_LEVEL
2. ESSENTIAL_COMPLEXITY
3. HALSTEAD_ERROR_EST
4. LOC_BLANK
5. LOC_COMMENTS
6. LOC_EXECUTABLE
7. NUM_UNIQUE_OPERATORS
8. HALSTEAD_EFFORT
9. HALSTEAD_VOLUME
10. NUM_OPERATORS

## 🤖 Model Performance

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| **Neural Network** | **99.70%** | **99.40%** | **1.0000** |
| Random Forest | 99.40% | 98.80% | 0.9998 |
| Gradient Boosting | 99.11% | 98.20% | 0.9995 |

## 🔬 Methodology

### 1. Data Preprocessing
- StandardScaler normalization (Z-score)
- SMOTEENN for class imbalance (498 → 723 samples)
- Stratified train-test split

### 2. Feature Selection
- ANOVA F-value
- Mutual Information
- Random Forest Importance
- Extra Trees Importance
- Recursive Feature Elimination (RFE)
- Autoencoder-based selection

### 3. Model Training
- 10 ML algorithms evaluated
- 5-fold cross-validation
- Hyperparameter tuning
- Ensemble voting for top models

## 💡 Key Technologies

- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive data visualizations
- **Scikit-learn** - Machine learning models and preprocessing
- **Imbalanced-learn** - SMOTEENN implementation
- **Pandas & NumPy** - Data manipulation
- **Python 3.8+** - Core programming language

## 🎨 Dashboard Features

### Interactive Visualizations
- **Pie Charts** - Class distribution before/after SMOTEENN
- **Bar Charts** - Model performance comparison
- **Heatmaps** - Feature correlations
- **ROC Curves** - Model discrimination ability
- **Gauges** - Prediction confidence scores
- **Sankey Diagrams** - Data flow through SMOTEENN

### Real-time Interactions
- Adjust feature values with sliders
- Change prediction thresholds dynamically
- Compare multiple feature selection methods
- Filter and search defective samples
- Export results to CSV

## 📝 Usage Examples

### Making Predictions

1. Navigate to the **🚀 Live Prediction** page
2. Adjust feature values using sliders or input boxes
3. Click "Predict" to see results
4. View prediction probability and confidence
5. Try preset examples (defective/non-defective)

### Exploring Defects

1. Go to **🐛 Defect Analysis** page
2. Browse detected defects table
3. Filter by feature values or prediction confidence
4. View feature distributions by defect status
5. Examine synthetic code examples

### Comparing Models

1. Visit **📊 Model Performance** page
2. View interactive comparison table
3. Click on models to see detailed metrics
4. Explore confusion matrices
5. Analyze ROC curves

## 🔧 Troubleshooting

### Common Issues

**1. Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**2. Module not found errors**
```bash
pip install -r requirements.txt --upgrade
```

**3. Data files not found**
- Ensure you're running from the `streamlit_app` directory
- Check that parent directory contains the data files
- Verify paths in `config.py` match your file structure

**4. Visualization not rendering**
- Clear browser cache
- Try a different browser (Chrome/Firefox recommended)
- Check JavaScript is enabled

### Performance Optimization

- **Large datasets**: The app uses Streamlit's `@st.cache_data` decorator for efficient loading
- **Slow rendering**: Reduce the number of data points displayed in tables
- **Memory issues**: Close other browser tabs or restart the Streamlit server

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

## 🤝 Contributing

This is a project demonstration. For educational purposes, feel free to:
- Explore the code
- Modify visualizations
- Try different models
- Add new features

## 📄 License

This project is for educational and demonstration purposes.

## 👏 Acknowledgments

- **NASA** - For providing the aerospace module datasets
- **Kaggle** - For hosting the datasets
- **Streamlit** - For the amazing web app framework
- **Scikit-learn & Imbalanced-learn communities** - For ML tools

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review Streamlit documentation
3. Inspect browser console for errors
4. Verify data file paths in config.py

---

**Built with** ❤️ **using Streamlit, Python, and Machine Learning**

*Aerospace Software Defect Prediction - Making software safer, one prediction at a time!* ✈️
