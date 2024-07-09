from flask import Flask, render_template, Response, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import sqlite3
from camera import VideoCamera

app = Flask(__name__)

# Initialize the camera
camera = VideoCamera()
current_mood = None  # Global variable to store the current detected mood

# Step 1: Load and Preprocess the Dataset
dataset_path = '/Applications/general/vscode/Projects/mood rec/song_dataset/data_moods.csv'  # Ensure this path is correct
metadata = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Debug: Print the first few rows of the dataset to ensure it's loaded correctly
print(metadata.head())
print(metadata.columns)

# Step 2: Store Dataset in SQLite Database
conn = sqlite3.connect('mood_rec_database.db')
metadata.to_sql('songs', conn, if_exists='replace', index=False)
conn.close()

# Step 3: Train the Model
try:
    # Define feature columns and target variable
    feature_columns = ['name', 'album', 'artist']  # Replace with your actual feature columns
    X = metadata[feature_columns]
    y = metadata['mood']  # Assuming 'mood' is the target variable

    # One-hot encode categorical features
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # Debug: Print the shapes of X_encoded and y to ensure they are correctly formed
    print(f"X_encoded shape: {X_encoded.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
except ValueError as e:
    print(f"Error during model training: {e}")

# Step 4: Define Label Mapping (if 'mood' column is categorical)
label_mapping = {idx: mood for idx, mood in enumerate(y_train.unique())}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    global current_mood
    while True:
        frame, mood = camera.get_frame()
        if frame is None:
            continue
        current_mood = mood  # Update the current detected mood
        mood_text = mood.encode() if mood else b'unknown'
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' + b'mood: ' + mood_text + b'\r\n')

@app.route('/current_mood', methods=['GET'])
def get_current_mood():
    global current_mood
    return jsonify({'mood': current_mood})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global current_mood
    current_mood = None  # Reset current mood
    camera.start_camera()
    return '', 204

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    camera.stop_camera()
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    # Step 5: Process User Input and Make Predictions
    selected_mood = request.form['mood']

    # Step 6: Query SQLite Database for Recommendations
    conn = sqlite3.connect('mood_rec_database.db')
    query = f"SELECT * FROM songs WHERE mood = '{selected_mood}'"
    recommended_songs = pd.read_sql(query, conn)
    conn.close()

    # Step 7: Prepare Recommendations for JSON Response
    song_recommendations = []
    for index, row in recommended_songs.iterrows():
        song_recommendations.append({
            'name': row['name'],
            'artist': row['artist']
        })

    return jsonify({'songs': song_recommendations})

if __name__ == '__main__':
    app.run(debug=True)
