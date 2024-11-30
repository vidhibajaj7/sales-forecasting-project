from train_model import train_model
from evaluate_model import evaluate_model

# Define file paths
data_path = 'data/sales_data.csv'  # Path to the dataset
model_path = 'models/sales_forecasting_model.pkl'  # Path to save the trained model

# Step 1: Train the model
print("Starting model training...")
model, X_test, y_test = train_model(data_path, model_path)
print("Model training completed!")

# Step 2: Evaluate the model
print("Starting model evaluation...")
evaluate_model(model_path, X_test, y_test)
print("Model evaluation completed! Visualizations saved in the 'visuals/' folder.")