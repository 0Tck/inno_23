from flask import Flask, render_template, request
#from ml import predict
import pandas as pd
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV

app = Flask(__name__, static_url_path='/static')

"""def  train():
    df = pd.read_csv('ml/datasetf.csv')
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

    joblib.dump(best_model, 'destination_prediction_model_rf_best.joblib')"""

def predict_dest(data_list):
    #train()
    label_encoder_destination = joblib.load('label_encoder1.joblib')
    label_encoders_categorical = {column: joblib.load(f'label_encoder_{column}.joblib') for column in ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members']}
    model = joblib.load('destination_prediction_model_rf_best.joblib')

    #data_list = [80000, 'Family', 5, 'State', 'Summer', 'Train', 'Devotional', 4]

    columns = ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members']

    new_data_df = pd.DataFrame([data_list], columns=columns)

    for column, encoder in label_encoders_categorical.items():
        data_column = new_data_df[column]

        # Check if the column has numeric values
        if pd.api.types.is_numeric_dtype(data_column):
            # If numeric, no need for encoding
            new_data_df[column] = data_column
        else:
            # If string, apply label encoding with handling of unknown labels
            new_data_df[column] = data_column.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    predicted_label_encoded = model.predict(new_data_df)[0]

    predicted_destination = label_encoder_destination.inverse_transform([predicted_label_encoded])[0]

    #print(f"Predicted Destination: {predicted_destination}")

    return predicted_destination

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/suggest', methods=['GET','POST'])
def suggest():
    if request.method == 'POST':
        budget = request.form.get('Budget')
        company = request.form.get('Company')
        days = request.form.get('Days')
        rang = request.form.get('Range')
        season = request.form.get('Season')
        transport = request.form.get('Transport')
        vacation = request.form.get('Vacation-type')
        popu = request.form.get('Member-count')
        l=[budget, company, days, rang, season, transport, vacation, popu]
        #print(l)
        if '' in l:
            return render_template('home.html', result='Fill in all the details before submitting.')
        res = predict_dest(l)  # Adjust this based on your actual prediction logic

        return render_template('home.html', result='You can go to '+res)
    return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True)

