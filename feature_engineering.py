import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Create a total_sales column
    data['total_sales'] = data['price'] * data['units_sold']

    # Define features and target
    features = ['price', 'units_sold', 'store', 'product']
    target = 'total_sales'

    X = data[features]
    y = data[target]

    # Preprocessing pipeline
    numerical_features = ['price', 'units_sold']
    categorical_features = ['store', 'product']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor