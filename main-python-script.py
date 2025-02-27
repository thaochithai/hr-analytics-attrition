from statistics import correlation
import pandas as pd
import numpy as np
from matplotlib.pyplot import colorbar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

#-----Model comparing ----

#1/Load and explore
#load file
file_path_training = file_path
data = pd.read_csv(file_path_training)
print(data.info())
print(data.describe())
print(data.isnull().sum())

#pairplot

data_categorical = data.drop(columns=['department','salary','work_accident','promotion_last_5years','number_project'])
sns.pairplot(data=data_categorical,hue='left',kind='reg',diag_kind='kde',palette="Greens",corner=True)


# Drop non-numeric columns to focus on correlation between numeric variables
data_cor = data.drop(columns=['department', 'salary'])

# Compute the correlation matrix
corrmat = data_cor.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))  # Slightly larger figure for better readability

# Create a heatmap with improved parameters
sns.heatmap(
    corrmat,
    vmax=0.8,  # Maximum correlation value to color-scale
    annot=True,  # Annotate cells with correlation coefficients
    cmap="Greens",  # Use Greens color map
    square=True,  # Make cells square-shaped
    cbar_kws={"shrink": 0.8}  # Shrink the color bar slightly for better aesthetics
)

# Display the heatmap
plt.title("Correlation Matrix Heatmap")  # Add a title for clarity
plt.show()


#2/Preprocessing
#encode categorical not binary yet
data_encoded = pd.get_dummies(data, columns=['department', 'salary'], drop_first=True)


#spliting for training and testing 30:70
X = data_encoded.drop(columns=['left'])
y = data_encoded['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Scale features for clustering and logistic regression
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#3/Modeling
#Logistic Regression
log_reg = LogisticRegression(max_iter=5000, solver='liblinear', random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

#Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

#4/Evaluate models

#metrics
models = {
    'Logistic Regression': y_pred_log_reg,
    'Logistic Regression (Clustered)': y_pred_log_reg_clustered,
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb
}

metrics = []
for model_name, y_pred in models.items():
    metrics.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC': roc_auc_score(y_test, y_pred)
    })
metrics_df = pd.DataFrame(metrics)
print("\nModel Performance Metrics\n")
print(metrics_df)

#visualization
plt.figure(figsize=(12, 6))
sns.heatmap(metrics_df.set_index('Model'), annot=True, fmt=".3f", cmap="Greens", cbar=True, linewidths=0.5)
plt.title('Model Performance Metrics')
plt.show()
