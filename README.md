# HR Analytics: Employee Attrition Prediction

## 📋 Overview

This project applies machine learning techniques to predict employee attrition and identify key factors contributing to turnover in an organization. Using HR analytics data, the model helps HR departments proactively address retention issues and develop targeted strategies to improve employee satisfaction.

## 🎯 Key Features

- **Data Exploration**: Comprehensive analysis of HR data to identify patterns and relationships
- **Predictive Modeling**: Implementation of Random Forest classifier for attrition prediction
- **Feature Importance**: Identification of key factors that contribute to employee turnover
- **Performance Metrics**: Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- **Visualization**: Intuitive charts and graphs to communicate insights

## 🛠️ Technologies Used

- **Python**: Main programming language
- **Scikit-learn**: For building and evaluating machine learning models
- **Pandas/NumPy**: For data manipulation and numerical operations
- **Matplotlib/Seaborn**: For data visualization
- **Jupyter Notebooks**: For exploratory data analysis and demonstration

## 📊 Dataset

The dataset contains HR information including:

- Satisfaction level
- Last evaluation score
- Number of projects
- Average monthly hours
- Time spent at the company
- Work accidents
- Promotions in the last 5 years
- Department
- Salary
- Whether the employee left the company (target variable)

## 🔍 Key Insights

From the exploratory data analysis and modeling, we discovered:

1. **Satisfaction Impact**: Employee satisfaction level is the most influential factor in predicting turnover
2. **Workload Effect**: Employees with high workloads (number of projects, working hours) combined with low satisfaction are at highest risk of leaving
3. **Departmental Differences**: Sales, technical, and support roles have the highest turnover rates
4. **Promotion Consideration**: Lack of promotion for 5+ years significantly increases attrition risk

## 📈 Model Performance

The Random Forest model achieved:
- Accuracy: 96.7%
- Precision: 96.4%
- Recall: 92.7%
- F1-Score: 94.5%
- ROC-AUC: 95.8%

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version
```

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/hr-analytics-attrition.git
cd hr-analytics-attrition

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the main prediction script
python src/hr_attrition_prediction.py

# Or explore the notebooks
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 📁 Project Structure

```
hr-analytics-attrition/
│
├── data/
│   └── raw/                     # Original, immutable data
│       └── A2_Data_Succession.csv
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   └── hr_attrition_prediction.py
│
├── results/
│   └── figures/                 # Generated graphs and visualization
│
├── requirements.txt             # Dependencies
├── setup.py                     # Make the project pip installable
├── README.md                    # Project description
└── LICENSE                      # MIT License
```

## 📝 Future Improvements

- Implement hyperparameter tuning to optimize model performance
- Explore more advanced feature engineering techniques
- Develop a web dashboard for HR professionals to use the model interactively
- Incorporate additional HR datasets for better generalization

## 🙏 Acknowledgements

- Data provided by the Data Science course at KUL
- Inspired by real-world HR challenges in talent retention
