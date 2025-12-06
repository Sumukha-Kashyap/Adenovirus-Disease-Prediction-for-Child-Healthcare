import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Machine Learning Modules
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class AdenovirusPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.encoders = {}

    def load_and_clean_data(self):
        print(">>> Loading Dataset...")
        self.data = pd.read_csv(self.file_path)
        
        # Standardize column names (strip whitespace)
        self.data.columns = self.data.columns.str.strip()
        
        print(f"Data Loaded: {self.data.shape[0]} rows, {self.data.columns.size} columns")
        return self.data.head()

    def preprocess_data(self):
        print(">>> Preprocessing Data...")
        
        # Encode Categorical Variables
        # Note: Storing encoders to handle new input data later (Production ready)
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.encoders[col] = le
            
        # correlation matrix visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.show()

        # Splitting
        X = self.data.drop('Adenoviruses', axis=1)
        y = self.data['Adenoviruses']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        print("Data Split and Encoded successfully.")

    def train_and_evaluate_models(self):
        print(">>> Training Models with Cross-Validation...")
        
        # Dictionary of classifiers (Fixed Regressors to Classifiers)
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=10),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(kernel='linear', probability=True)
        }

        results_df = []

        for name, clf in classifiers.items():
            # Using a pipeline to scale data before feeding to model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            
            # Training
            pipeline.fit(self.X_train, self.y_train)
            
            # Prediction
            y_pred = pipeline.predict(self.X_test)
            
            # Metrics
            acc = accuracy_score(self.y_test, y_pred) * 100
            f1 = f1_score(self.y_test, y_pred)
            
            self.models[name] = pipeline
            results_df.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})

        results = pd.DataFrame(results_df).sort_values(by='Accuracy', ascending=False)
        print("\nModel Performance Metrics:")
        print(results)
        
        # Select best model automatically
        best_model_name = results.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        print(f"\n>>> Best Model Selected: {best_model_name}")
        
        return results

    def visualize_performance(self):
        if not self.best_model:
            return
        
        y_pred = self.best_model.predict(self.X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix ({self.best_model.steps[-1][0]})")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Feature Importance (if applicable to the model)
        classifier = self.best_model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feature_names = self.X_train.columns
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(self.X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(self.X_train.shape[1]), feature_names[indices], rotation=90)
            plt.tight_layout()
            plt.show()

    def make_prediction(self, input_data):
        """
        Input data should be a list of values matching the feature columns.
        """
        print(f"\n>>> Processing New Patient Data: {input_data}")
        prediction = self.best_model.predict([input_data])
        probability = self.best_model.predict_proba([input_data])[0][1] # Prob of positive class

        status = "Positive (+ve)" if prediction[0] == 1 else "Negative (-ve)"
        confidence = probability if prediction[0] == 1 else 1 - probability
        
        print(f"Prediction: Adenovirus {status}")
        print(f"Confidence Level: {confidence*100:.2f}%")
        return prediction[0]

    def save_best_model(self):
        if self.best_model:
            joblib.dump(self.best_model, 'adenovirus_model.pkl')
            print(">>> Model serialized and saved to 'adenovirus_model.pkl'")

# --- Execution Block ---

if __name__ == "__main__":
    # Initialize Project
    # Replace with your actual path
    project = AdenovirusPredictor('Adenoviruses_Dataset.csv') 
    
    # Run Pipeline
    project.load_and_clean_data()
    project.preprocess_data()
    project.train_and_evaluate_models()
    project.visualize_performance()
    
    # Advanced Prediction (Simulating Real Inputs)
    # Note: Ensure inputs match the exact order of X columns
    sample_patient_1 = [1, 1, 1, 1, 1, 1, 0, 0] 
    sample_patient_2 = [0, 0, 1, 0, 0, 1, 0, 0]
    
    project.make_prediction(sample_patient_1)
    project.make_prediction(sample_patient_2)
    
    # Save for deployment
    project.save_best_model()