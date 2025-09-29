import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import joblib 

warnings.filterwarnings("ignore", category=UserWarning)

def train_crop_model():
    """
    This function handles the entire process of loading data, training the model,
    and evaluating its performance.
    """
    print("Starting the model training process...")

    try:
        data = pd.read_csv('Crop_recommendation.csv')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("The 'Crop_recommendation.csv' file was not found.")
        return None, None

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    label = 'label'

    X = data[features]
    y = data[label]
    
    print(f"Features (inputs): {features}")
    print(f"Label (output to predict): {label}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training the Random Forest model... This might take a moment.")
    model.fit(X_train, y_train)
    print("Model training complete!")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    return model, features

def make_prediction(model, feature_names):
    if model is None:
        print("Model is not available for prediction.")
        return

    new_data_list = [[104, 18, 30, 23.6, 60.3, 6.7, 140.9]]
    new_data_df = pd.DataFrame(new_data_list, columns=feature_names)
    
    predicted_crop = model.predict(new_data_df)
    
    print("\n--- New Prediction ---")
    print(f"Input conditions (N, P, K, Temp, Humidity, pH, Rainfall): {new_data_list[0]}")
    print(f"==> AI Recommended Crop: {predicted_crop[0]}")
    print("----------------------\n")
    
    new_data_list_2 = [[83, 45, 60, 28.5, 70.3, 7.0, 150.9]]
    new_data_df_2 = pd.DataFrame(new_data_list_2, columns=feature_names)
    predicted_crop_2 = model.predict(new_data_df_2)

    print("\n--- Another Prediction ---")
    print(f"Input conditions (N, P, K, Temp, Humidity, pH, Rainfall): {new_data_list_2[0]}")
    print(f"==> AI Recommended Crop: {predicted_crop_2[0]}")
    print("----------------------\n")

if __name__ == "__main__":
    trained_model, feature_names = train_crop_model()
    
    if trained_model:
        model_filename = 'crop_model.joblib'
        joblib.dump(trained_model, model_filename)
        print(f"Model saved successfully as '{model_filename}'")

        make_prediction(trained_model, feature_names)