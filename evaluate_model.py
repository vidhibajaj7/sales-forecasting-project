import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def evaluate_model(model_path, X_test, y_test):
    # Load model
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluation Metrics:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.2f}")

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Sales', marker='o', linestyle='--')
    plt.plot(y_pred, label='Predicted Sales', marker='x', linestyle=':')
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Index')
    plt.ylabel('Total Sales')
    plt.savefig('visuals/actual_vs_predicted.png')
    plt.show()

    sns.scatterplot(x=y_test, y=y_pred)
    plt.title('Actual vs Predicted Sales Scatter Plot')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.savefig('visuals/scatter_plot.png')
    plt.show()

    errors = y_test - y_pred
    sns.histplot(errors, kde=True, color='red')
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig('visuals/error_distribution.png')
    plt.show()