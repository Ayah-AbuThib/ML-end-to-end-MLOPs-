if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import List, Dict, Union
import joblib
import numpy as np
import os

@custom
def transform_custom(*args, **kwargs):
    """
    Loads the trained model and makes predictions on the given input data.
    """
    model_path = "default_repo/transformers/random_forest_(tuned).joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    
    default_inputs = [
        {
            'carat': 0.5,
            'x': 4.2,
            'y': 4.3,
            'z': 2.6,
            'cut_encoded': 3,
            'color_encoded': 2,
            'clarity_encoded': 5,
            'depth': 61.2,
            'table': 57.0,
        },
        {
            'carat': 1.0,
            'x': 6.5,
            'y': 6.6,
            'z': 4.0,
            'cut_encoded': 2,
            'color_encoded': 3,
            'clarity_encoded': 4,
            'depth': 62.5,
            'table': 58.0,
        },
    ]
    
    input_features = [[entry[key] for key in entry] for entry in default_inputs]
    
    predictions = model.predict(np.array(input_features))
    
    return predictions.tolist()

@test
def test_output(output, *args) -> None:
    """
    Tests if the output of the inference block is valid.
    """
    assert isinstance(output, list), "Output should be a list of predictions."
    assert all(isinstance(pred, (float, int)) for pred in output), "Each prediction should be a numerical value."
    assert len(output) == len(default_inputs), "Output length should match input length."
