import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('ml/dataset.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df['Destination'])
joblib.dump(label_encoder, 'label_encoder1.joblib')
df['Destination'] = label_encoder.transform(df['Destination'])

categorical_columns = ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members']

for column in categorical_columns:
    label_encoder.fit(df[column])
    joblib.dump(label_encoder, f'label_encoder_{column}.joblib')
    df[column] = label_encoder.transform(df[column])


X = df.drop(['Destination'], axis=1)
y = df['Destination']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

joblib.dump(best_model, 'destination_prediction_model_rf_best.joblib')