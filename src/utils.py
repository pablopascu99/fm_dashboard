import joblib

def load_models():
    """Loads trained models, the encoder, and the scaler."""
    models = {
        "RandomForest": joblib.load("src/ml/ml_models/RandomForest_model.pkl"),
        "SVM": joblib.load("src/ml/ml_models/SVM_model.pkl"),
        "KNN": joblib.load("src/ml/ml_models/KNN_model.pkl"),
        "NaiveBayes": joblib.load("src/ml/ml_models/NaiveBayes_model.pkl")
    }
    encoder = joblib.load("src/ml/ml_models/label_encoder.pkl")
    scaler = joblib.load("src/ml/ml_models/scaler.pkl")
    return models, encoder, scaler

def preprocess_data(df, scaler):
    """Preprocesses data for prediction."""
    df.drop(columns=["timestamp"], inplace=True)  # Remove timestamp
    
    # Load original columns used for training
    original_columns = scaler.feature_names_in_  # Get original names
    
    # Remove extra columns and ensure order
    df = df[original_columns]
    
    # Scale the data
    X_scaled = scaler.transform(df)
    return X_scaled

def standardize_column_names(df, column_mapping):
    """Standardizes column names based on a mapping."""
    new_columns = {}
    for col in df.columns:
        if col in column_mapping:
            new_columns[col] = column_mapping[col][0]  # Use standardized name
        else:
            new_columns[col] = col  # Keep original name if not in mapping
    df.rename(columns=new_columns, inplace=True)
    return df

def predict(models, X, encoder):
    """Generates predictions and transforms them to labels using the encoder."""
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X)
        predictions[name] = encoder.inverse_transform(pred)  # Transform to labels
    return predictions