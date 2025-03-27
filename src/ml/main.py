from sklearn.model_selection import train_test_split
from data_loader import load_data, preprocess_data
from model_trainer import train_models, evaluate_models
from model_saver import save_models

def main(filepath):
    """Main pipeline to load data, train models, and save them."""
    # Load and preprocess data
    df = load_data(filepath)
    X, y, encoder, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test, encoder)
    
    # Save models
    save_models(models, encoder, scaler)

if __name__ == "__main__":
    file_path = "../files/all_data_ts.csv"  # Change this to the actual path of your file
    main(file_path)