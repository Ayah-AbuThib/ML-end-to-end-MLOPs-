@custom
if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import List, Dict, Union
import joblib
import numpy as np
import os

def transform_custom(data: List[Dict[str, Union[float, int]]], *args, **kwargs):
    """
    Loads the trained model and makes predictions on the given input data.
    """
    model_path = "default_repo/transformers/random_forest_(tuned).joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)

    # Extract input features dynamically
    input_features = [[entry[key] for key in entry] for entry in data]

    # Make predictions
    predictions = model.predict(np.array(input_features))

    return {"predicted_price": predictions.tolist()}
