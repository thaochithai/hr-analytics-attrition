# HR Analytics: Employee Attrition Prediction
This project applies machine learning techniques to predict employee attrition and identify key factors contributing to turnover in an organization. Using HR analytics data, the model helps HR departments proactively address retention issues and develop targeted strategies to improve employee satisfaction.

## 📊 Dataset

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

From the exploratory data analysis and modeling, I discovered:

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

## 📝 Future Improvements

- Implement hyperparameter tuning to optimize model performance
- Explore more advanced feature engineering techniques
- Incorporate additional HR datasets for better generalization

## 🙏 Acknowledgements

- Data provided by the Data Science course at KUL
- Inspired by real-world HR challenges in talent retention
