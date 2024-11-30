from xgboost import XGBRegressor
import joblib
from feature_engineering import preprocess_data

def train_model(data_path, model_path):
    # Preprocess data
    X, y, preprocessor = preprocess_data(data_path)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save model and preprocessor
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    print("Model trained and saved!")
    return model, X_test, y_test