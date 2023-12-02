from flask import Flask, render_template, request, redirect, url_for
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

def  train():
    df = pd.read_csv('ml/datasett.csv')
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Destination'])
    joblib.dump(label_encoder, 'label_encoder1.joblib')
    df['Destination'] = label_encoder.transform(df['Destination'])

    categorical_columns = ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members', 'place-type']

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

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(best_model, 'destination_prediction_model_rf_best.joblib')

def predict_dest(data_list):
    train()
    label_encoder_destination = joblib.load('label_encoder1.joblib')
    label_encoders_categorical = {column: joblib.load(f'label_encoder_{column}.joblib') for column in ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members','place-type']}
    model = joblib.load('destination_prediction_model_rf_best.joblib')

    #data_list = [80000, 'Family', 5, 'State', 'Summer', 'Train', 'Devotional', 4]

    columns = ['Budget', 'Company', 'No. of Days', 'Range', 'Season', 'Mode of Transport', 'Type of Vacation', 'No. of Members','place-type']

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
        plac = request.form.get('place-type')
        l=[budget, company, days, rang, season, transport, vacation, popu, plac]
        #print(l)
        if '' in l:
            return render_template('home.html', result='Fill in all the details before submitting.')
        res = predict_dest(l)  # Adjust this based on your actual prediction logic

        return render_template('home.html', result='You can go to '+res)
    return render_template('home.html')

def back_end(place):
    if place=='Tirupathi':
        desc="""Tirupati, a spiritual haven nestled in the southeastern state of Andhra Pradesh, is synonymous with devotion and rich cultural heritage.
        Renowned for the sacred Sri Venkateswara Temple atop the picturesque Tirumala Hills, the town draws millions of pilgrims annually.
        The temple, dedicated to Lord Venkateswara, is a symbol of architectural splendor and spiritual significance.
        The vibrant atmosphere is heightened during festivals, with the Brahmotsavam being a major highlight,
        showcasing elaborate processions and religious fervor. Tirupati's culture is deeply intertwined with its religious practices,
        and the town's traditional arts, music, and dance further enrich the cultural tapestry,
        offering a unique blend of spirituality and artistic expression."""
        tra_des="""Tirupati, nestled in the southern part of India, is renowned as a spiritual haven and a major pilgrimage destination. The focal point is the sacred Venkateswara Temple atop the Seven Hills, dedicated to Lord Venkateswara. Millions of devotees embark on the journey to seek blessings and fulfill vows. The temple's intricate architecture and the breathtaking views from the hills make the pilgrimage a profound and awe-inspiring experience. Tirupati's spiritual aura extends beyond the temple, with serene spots like Akasa Ganga offering moments of contemplation. The city's cultural richness and warm hospitality further enhance the overall travel experience, making Tirupati a destination where spirituality and tradition converge."""
        food_des="""Tirupati, a spiritual destination in southern India, not only draws pilgrims to its sacred temples but also tantalizes taste buds with its unique culinary offerings. Renowned for its luscious laddus, Tirupati Balaji Temple serves these sweet delicacies as prasadam, considered divine blessings. The local cuisine reflects the rich culinary heritage of Andhra Pradesh, featuring flavorful dishes like Puliyodarai (tamarind rice), Pongal (rice and lentil dish), and spicy Andhra Biryanis. Street markets brim with traditional snacks like Murukku and Boorelu, tempting both locals and visitors alike. In Tirupati, the culinary experience is a delightful journey complementing the spiritual ambiance of the sacred city."""
        lang_des="""Tirupati, nestled in the southern part of India, is renowned for the sacred Tirumala Venkateswara Temple, one of the most visited pilgrimage sites globally. The temple atop the seven hills attracts millions of devotees annually, drawn by the spiritual significance of Lord Venkateswara. Pilgrims climb the steps, or opt for various modes of transport, seeking blessings and participating in rituals. Beyond its religious aura, Tirupati offers a serene escape with lush landscapes and traditional markets, making it a destination that seamlessly blends spiritual devotion with natural beauty."""
        places_des="""Sri Govindarajaswami Temple:  An ancient temple dedicated to Lord Krishna and one of the most important in Tirupati.
                    Kapila Theertham:  A picturesque waterfall and a sacred pond located about 3 kilometers from Tirupati, surrounded by lush greenery.
                    Sri Padmavathi Ammavari Temple:  Dedicated to Goddess Padmavathi, the consort of Lord Venkateswara, this temple is located in Tiruchanur, about 5 kilometers from Tirupati.
                    ISKCON Temple:  The International Society for Krishna Consciousness (ISKCON) has a vibrant temple in Tirupati that attracts devotees and visitors.
                    TTD Gardens:  Sprawling gardens maintained by the Tirumala Tirupati Devasthanams (TTD), offering a serene atmosphere for relaxation.
                    Deer Park:  A natural habitat for deer and other wildlife, providing a tranquil setting for nature enthusiasts.
                    Chandragiri Fort:  About 16 kilometers from Tirupati, this historical fort dates back to the 11th century and offers panoramic views of the surrounding landscape.
                    Silathoranam:  A natural rock formation in Tirumala, believed to be Lord Rama's bow, adding a touch of geological and mythological interest.
                    Akasa Ganga:  A mountain stream and waterfall near the Tirumala temple, known for its religious significance and scenic beauty.
                    Sri Venkateswara Zoological Park:  Located in Tirupati, this zoo is home to a variety of wildlife species, providing an educational and entertaining experience."""               
    return desc, tra_des, food_des, lang_des, places_des

@app.route('/onvisit', methods=['GET','POST'])
def onvisit():
    place = request.args.get('place_search', '')
    if place=='':
        return render_template('home.html', null='fill it')
    desc,trav,food,lang,places=back_end(place)
    return render_template('onvisit.html',place=place,desc=desc,travel=trav,food=food,lang=lang,places=places)
        
@app.route('/search', methods=['GET', 'POST'])
def search():
    return redirect(url_for('onvisit'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/onvisit?place_search=Tirupathi/summary')
def summary():
    budget = request.form.get('Budget')
    return render_template('bill.html', total=budget)

if __name__ == '__main__':
    app.run(debug=True)

