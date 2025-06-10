import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def preprocess_inputs(df):
    df = df.copy()



    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)


    df = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'], prefix=['EDU', 'MAR'])


    y = df['default payment next month'].copy()
    X = df.drop('default payment next month', axis=1).copy()


    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, scaler


file_path = "default of credit card clients.xls"
data = pd.read_excel(file_path, header=1, engine="xlrd")
X, y, scaler = preprocess_inputs(data)

data

data.info()


plt.figure(figsize=(18, 15))

corr = data.corr()
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.title("Correlation Heatmap")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)


models = {
    LogisticRegression(max_iter=1000): "Logistic Regression",
    SVC(kernel='linear', probability=True): "Support Vector Machine",
    MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,)): "Neural Network"
}



for model in models.keys():
    model.fit(X_train, y_train)

predictions = {name: model.predict(X_test) for model, name in models.items()}


results_df = pd.DataFrame({'Actual': y_test, **predictions})


for model, name in models.items():
    if hasattr(model, 'predict_proba'):
        results_df[f'{name}_Probability'] = model.predict_proba(X_test)[:, 1]



print("\nFirst 15 Predictions:")
print(results_df.head(15))

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6)
plt.title('PCA Projection (2D) of Credit Clients')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()



results_df.to_csv('default_predictions.csv', index=False)
print("\nPredictions saved to 'default_predictions.csv'")



accuracies = {name: accuracy_score(y_test, pred) for name, pred in predictions.items()}
best_model_name = max(accuracies, key=accuracies.get)
best_model = [m for m, name in models.items() if name == best_model_name][0]

print(f"\nBest Model: {best_model_name} with accuracy {accuracies[best_model_name]:.2%}")



def predict_new_data(model, scaler, new_data):
    """Predict for new customer data"""
    new_data = new_data.reindex(columns=X.columns, fill_value=0)
    scaled_data = pd.DataFrame(scaler.transform(new_data), columns=X.columns)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] if hasattr(model, 'predict_proba') else None
    return prediction, probability


new_customer = pd.DataFrame({
    'LIMIT_BAL': [2000],
    'SEX': [2],
    'AGE': [24],
    'PAY_0': [2],
    'PAY_2': [2],
    'PAY_3': [-1],
    'PAY_4': [-1],
    'PAY_5': [-2],
    'PAY_6': [-2],
    'BILL_AMT1': [3913],
    'BILL_AMT2': [3102],
    'BILL_AMT3': [689],
    'BILL_AMT4': [0],
    'BILL_AMT5': [0],
    'BILL_AMT6': [0],
    'PAY_AMT1': [0],
    'PAY_AMT2': [689],
    'PAY_AMT3': [0],
    'PAY_AMT4': [0],
    'PAY_AMT5': [0],
    'PAY_AMT6': [0],
    'EDU_1': [2],
    'EDU_2': [2],
    'EDU_3': [2],
    'MAR_1': [1],
    'MAR_2': [1],
    'MAR_3': [1]
})



pred, prob = predict_new_data(best_model, scaler, new_customer)

print(f"\nNew Customer Prediction:")
print(f"Will Default: {'Yes' if pred == 1 else 'No'}")
print(f"Probability: {prob:.2%} (if available)")
