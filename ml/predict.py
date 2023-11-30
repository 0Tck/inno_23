import pandas as pd
import joblib

label_encoder_destination = joblib.load('label_encoder1.joblib')
label_encoders_categorical = {column: joblib.load(f'label_encoder_{column}.joblib') for column in ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members']}
model = joblib.load('destination_prediction_model_rf_best.joblib')

new_data = {
    'Budget': 40000,
    'Company': 'Solo',
    'No. of Days': 5,
    'Range': 'City',
    'Season': 'Autumn',
    'Mode of Transport': 'Train',
    'Type of Vacation': 'Adventure',
    'No. of Members': 1
}

new_data_df = pd.DataFrame([new_data])

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

