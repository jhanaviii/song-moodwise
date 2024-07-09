Mood based-Song Recommendation System
This is a Flask-based web application that leverages machine learning to recommend songs based on detected and manually input moods. It integrates a Convolutional Neural Network (CNN) for real-time mood detection from video streams, alongside manual mood selection for personalized recommendations. The CNN model, trained using K-Fold cross-validation on a custom image dataset, achieves high accuracy in classifying emotions. Using SQLite queries, the system dynamically suggests songs corresponding to the detected or selected mood category, enhancing user engagement and music discovery experiences.

Features
1. Real-time and Manual Mood Detection: Utilizes a video stream and allows users to manually input their mood for personalized song recommendations.
2. Model Training: Employs a CNN model trained with K-Fold cross-validation on custom image data to accurately classify emotions.
3. Song Recommendations: Uses SQLite queries to recommend songs based on detected or selected mood categories, providing tailored music suggestions.
4. User Interaction: Enhances user engagement through interactive mood selection and dynamic content updates.

Technologies Used
1. Backend: Python, Flask, SQLite
2. Machine Learning: TensorFlow/Keras, scikit-learn
3. Web Technologies: HTML, CSS, JavaScript
