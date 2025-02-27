import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)

class HRAttritionModel:
    """
    A class for predicting employee attrition using HR analytics data.
    
    This model analyzes factors that contribute to employee turnover and
    provides predictions on which employees are likely to leave.
    """
    
    def __init__(self, data_path=None, data=None):
        """
        Initialize the HR Attrition Model.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing HR data
        data : pandas.DataFrame, optional
            DataFrame containing HR data (alternative to data_path)
        """
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or data must be provided")
            
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None
        self.predictions = None
        self.probabilities = None
    
    def explore_data(self):
        """Perform basic data exploration and return summary statistics."""
        print("Data Information:")
        print(self.data.info())
        
        print("\nDescriptive Statistics:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        return {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict()
        }
    
    def visualize_correlations(self):
        """Create and display correlation heatmap for numerical features."""
        # Drop non-numeric columns to focus on correlation between numeric variables
        data_cor = self.data.drop(columns=['department', 'salary'])

        # Compute the correlation matrix
        corrmat = data_cor.corr()

        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))

        # Create a heatmap
        sns.heatmap(
            corrmat,
            vmax=0.8,
            annot=True,
            cmap="Greens",
            square=True,
            cbar_kws={"shrink": 0.8}
        )

        plt.title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.savefig("results/figures/correlation_heatmap.png")
        plt.show()
        
        return corrmat
    
    def visualize_pairplot(self):
        """Create and display pairplot for data visualization."""
        data_categorical = self.data.drop(
            columns=['department', 'salary', 'work_accident', 
                     'promotion_last_5years', 'number_project']
        )
        
        g = sns.pairplot(
            data=data_categorical,
            hue='left',
            kind='reg',
            diag_kind='kde',
            palette="Greens",
            corner=True
        )
        
        plt.savefig("results/figures/feature_pairplot.png")
        plt.show()
        
        return g
    
    def preprocess_data(self, test_size=0.3, random_state=42):
        """
        Preprocess the data for modeling by splitting into train/test sets.
        
        Parameters:
        -----------
        test_size : float, default=0.3
            Proportion of the dataset to include in the test split
        random_state : int, default=42
            Controls the shuffling for reproducible output
            
        Returns:
        --------
        dict
            Dictionary containing split datasets
        """
        # Split data
        X = self.data.drop(columns=['left'])
        y = self.data['left']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test
        }
    
    def build_pipeline(self, random_state=42):
        """
        Build a pipeline with preprocessing steps and a Random Forest classifier.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        sklearn.pipeline.Pipeline
            The complete preprocessing and modeling pipeline
        """
        # Define preprocessing for numerical features
        numerical_features = [
            'satisfaction_level', 'last_evaluation', 'number_project', 
            'average_montly_hours', 'time_spend_company', 'work_accident', 
            'promotion_last_5years'
        ]
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])
        
        # Define preprocessing for categorical features
        categorical_features = ['department', 'salary']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create full pipeline with Random Forest classifier
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])
        
        return self.pipeline
    
    def train_model(self):
        """
        Train the model using the pipeline.
        
        Returns:
        --------
        sklearn.pipeline.Pipeline
            The trained pipeline
        """
        if self.pipeline is None:
            self.build_pipeline()
            
        if self.X_train is None:
            raise ValueError("Data must be preprocessed before training the model")
            
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline
    
    def evaluate_model(self):
        """
        Evaluate the trained model and compute performance metrics.
        
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Generate predictions
        self.predictions = self.pipeline.predict(self.X_test)
        self.probabilities = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, self.predictions),
            'Precision': precision_score(self.y_test, self.predictions),
            'Recall': recall_score(self.y_test, self.predictions),
            'F1-Score': f1_score(self.y_test, self.predictions),
            'ROC AUC': roc_auc_score(self.y_test, self.probabilities)
        }
        
        # Print metrics
        print("\nModel Performance Metrics\n")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))
        
        return metrics
    
    def visualize_metrics(self, metrics):
        """
        Create a bar chart visualization of model performance metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing performance metrics
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the metrics visualization
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            palette="Greens_d"
        )
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/figures/model_performance_metrics.png")
        plt.show()
        
        return plt.gcf()
    
    def feature_importance(self):
        """
        Display feature importance for the Random Forest model.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance values
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before extracting feature importance")
        
        # Get random forest model from pipeline
        rf_model = self.pipeline.named_steps['classifier']
        
        # Get preprocessor from pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Get feature names from preprocessor
        feature_names = []
        
        # Get numerical feature names (these pass through unchanged)
        numerical_features = preprocessor.transformers_[0][2]
        feature_names.extend(numerical_features)
        
        # Get categorical feature names (these are expanded by one-hot encoding)
        categorical_features = preprocessor.transformers_[1][2]
        # Get the OneHotEncoder for categorical features
        ohe = preprocessor.transformers_[1][1].named_steps['onehot']
        
        # Get the categories from the OneHotEncoder
        if hasattr(ohe, 'categories_'):
            # For fitted encoder
            for i, category in enumerate(ohe.categories_):
                feature_names.extend([f"{categorical_features[i]}_{c}" for c in category])
        else:
            # For unfitted encoder, just use placeholders
            for feature in categorical_features:
                feature_names.append(f"{feature}_categories")
        
        # Get feature importances
        feature_importances = rf_model.feature_importances_
        
        # If feature names length doesn't match importances length, use generic names
        if len(feature_names) != len(feature_importances):
            feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
        
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importances)],
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='Greens_r')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig("results/figures/feature_importance.png")
        plt.show()
        
        return importance_df
    
    def confusion_matrix_plot(self):
        """
        Plot confusion matrix for the model.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the confusion matrix
        """
        if self.predictions is None:
            raise ValueError("Model must be evaluated before plotting confusion matrix")
        
        cm = confusion_matrix(self.y_test, self.predictions)
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])
        disp.plot(cmap='Greens', values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("results/figures/confusion_matrix.png")
        plt.show()
        
        return plt.gcf()
    
    def roc_curve_plot(self):
        """
        Plot ROC curve for the model.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the ROC curve
        """
        if self.probabilities is None:
            raise ValueError("Model must be evaluated before plotting ROC curve")
        
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_predictions(
            self.y_test, 
            self.probabilities,
            name="Random Forest",
            alpha=0.8
        )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/figures/roc_curve.png")
        plt.show()
        
        return plt.gcf()
    
    def get_prediction_results(self, n_samples=10):
        """
        Get a sample of prediction results with actual values and probabilities.
        
        Parameters:
        -----------
        n_samples : int, default=10
            Number of samples to include in the results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sample predictions with actual values and probabilities
        """
        if self.predictions is None or self.probabilities is None:
            raise ValueError("Model must be evaluated before getting prediction results")
            
        # Create results DataFrame
        results_df = self.X_test.copy()
        results_df['Actual'] = self.y_test.reset_index(drop=True)
        results_df['Predicted'] = self.predictions
        results_df['Probability'] = self.probabilities
        
        # Return sample or full results
        if n_samples is None:
            return results_df
        else:
            return results_df.head(n_samples)
    
    def save_model_report(self, output_path="results/model_report.html"):
        """
        Generate and save a comprehensive model report.
        
        Parameters:
        -----------
        output_path : str, default="results/model_report.html"
            Path to save the HTML report
            
        Returns:
        --------
        str
            Path to the saved report
        """
        import os
        import datetime
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Calculate metrics if not already done
        if self.predictions is None:
            metrics = self.evaluate_model()
        else:
            metrics = {
                'Accuracy': accuracy_score(self.y_test, self.predictions),
                'Precision': precision_score(self.y_test, self.predictions),
                'Recall': recall_score(self.y_test, self.predictions),
                'F1-Score': f1_score(self.y_test, self.predictions),
                'ROC AUC': roc_auc_score(self.y_test, self.probabilities)
            }
        
        # Get feature importance
        try:
            importance_df = self.feature_importance()
            top_features = importance_df.head(5)[['Feature', 'Importance']].to_dict('records')
        except:
            top_features = [{'Feature': 'N/A', 'Importance': 0}]
        
        # Generate report content
        report = f"""
        <html>
        <head>
            <title>HR Attrition Model Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c7c3e; }}
                h2 {{ color: #2c7c3e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .timestamp {{ color: gray; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <h1>HR Attrition Model Report</h1>
            <p class="timestamp">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Data Overview</h2>
            <p>Dataset shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns</p>
            
            <h2>Model Performance</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add model metrics to the report
        for metric, value in metrics.items():
            report += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{value:.4f}</td>
                </tr>
            """
        
        report += """
            </table>
            
            <h2>Top 5 Important Features</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
        """
        
        # Add top features to the report
        for feature in top_features:
            report += f"""
                <tr>
                    <td>{feature['Feature']}</td>
                    <td>{feature['Importance']:.4f}</td>
                </tr>
            """
        
        report += """
            </table>
            
            <h2>Conclusions</h2>
            <ul>
                <li>The Random Forest model showed strong performance in predicting employee attrition.</li>
                <li>The most important features for predicting attrition are satisfaction level, time spent at the company, and number of projects.</li>
                <li>Employees with high workload (number of projects, average monthly hours) combined with low satisfaction are at highest risk of leaving.</li>
            </ul>
            
            <h2>Model Images</h2>
            <p>Model performance visualizations, confusion matrix, and ROC curve can be found in the results/figures directory.</p>
        </body>
        </html>
        """
        
        # Save the report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    import os
    
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Initialize the model with the data
    hr_model = HRAttritionModel("data/raw/A2_Data_Succession.csv")
    
    # Explore and visualize data
    hr_model.explore_data()
    hr_model.visualize_correlations()
    hr_model.visualize_pairplot()
    
    # Preprocess data
    hr_model.preprocess_data()
    
    # Build, train and evaluate model
    hr_model.build_pipeline()
    hr_model.train_model()
    metrics = hr_model.evaluate_model()
    
    # Visualize results
    hr_model.visualize_metrics(metrics)
    hr_model.feature_importance()
    hr_model.confusion_matrix_plot()
    hr_model.roc_curve_plot()
    
    # Get prediction results
    results = hr_model.get_prediction_results(n_samples=10)
    print("\nSample Predictions:")
    print(results)
    
    # Generate and save report
    hr_model.save_model_report()
