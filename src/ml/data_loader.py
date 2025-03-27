import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Loads the dataset from a CSV file and removes the 'timestamp' column."""
    df = pd.read_csv(filepath)
    df.drop(columns=["timestamp"], inplace=True)
    return df

def preprocess_data(df):
    """Preprocesses the data: separates features and labels, encodes them, and scales them."""
    X = df.drop(columns=["class"])
    y = df["class"]
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, encoder, scaler