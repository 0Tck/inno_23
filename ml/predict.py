import pandas as pd
import joblib

def predict_dest():
    label_encoder_destination = joblib.load('label_encoder1.joblib')
    label_encoders_categorical = {column: joblib.load(f'label_encoder_{column}.joblib') for column in ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members','type of place']}
    model = joblib.load('destination_prediction_model_rf_best.joblib')

    data_list = [80000, 'Family', 5, 'State', 'Summer', 'Train', 'Devotional', 4, 'hill station']

    columns = ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members','type of place']

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

    print(f"Predicted Destination: {predicted_destination}")

    

if __name__=='__main__':
    predict_dest()
