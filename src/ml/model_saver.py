import os
import joblib

def save_models(models, encoder, scaler, output_dir="ml_models"):
    """Saves trained models, the encoder, and the scaler to files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{output_dir}/{name}_model.pkl")
    joblib.dump(encoder, f"{output_dir}/label_encoder.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"Models successfully saved in the folder '{output_dir}'.")