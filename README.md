# HR Analytics: Employee Attrition Prediction
This project applies machine learning techniques to predict employee attrition and identify key factors contributing to turnover in an organization. Using HR analytics data, the model helps HR departments proactively address retention issues and develop targeted strategies to improve employee satisfaction.

## 📊 Datas Exploration
  
**Numerical Variables**
![pairplot](https://github.com/user-attachments/assets/24f98fda-4fa4-4bd9-be29-ddb3ec79f442) ![Correlation](https://github.com/user-attachments/assets/92785fdc-f9e8-40f2-894d-f2da7fa86c55)

**Key insights**
1. **Satisfaction Impact**: Employee satisfaction level is the most influential factor in predicting turnover
2. **Workload Effect**: Employees with high workloads (number of projects, working hours) combined with low satisfaction are at highest risk of leaving
3. **Departmental Differences**: Sales, technical, and support roles have the highest turnover rates
4. **Promotion Consideration**: Lack of promotion for 5+ years significantly increases attrition risk

## 📈 Models and Evaluation
**Models**
- Logistic Regression
- Random Forest
- Gradient Boosting
  
**Evaluation**
- Train-Test Split: A 70% training and 30% testing split was used.
- Evaluation Setup: Three predictive models proposed were tested. The models were compared based on out of sample accuracy to assess overall performance and precision, recall, F1-score, and ROC-AUC to evaluate their ability to predict the negative class (employees likely to leave). 

## 🚀 Performance
![Figure_2](https://github.com/user-attachments/assets/82402b2b-1873-47b5-91b8-874ac51e6f53)
-	RF and GB achieved high performance, RF has the highest accuracy and robust performance across metrics. Using features importance scores > 0.01 further optimized RF, increasing accuracy to 0.986,
-	LR has low performance and is not good in handling non-linear interactions and complex relationships, making it less effective than RF and GB.

## 📝 Future Improvements

- Implement hyperparameter tuning to optimize model performance
- Explore more advanced feature engineering techniques
- Incorporate additional HR datasets for better generalization

## 🙏 Acknowledgements

- Data provided by the Data Science course at KUL
- Inspired by real-world HR challenges in talent retention
