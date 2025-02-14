if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import List, Dict, Union
import joblib
import numpy as np
import os

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


@custom
def transform_custom():


    app = FastAPI()

    # Define input schema
    class DiamondFeatures(BaseModel):
        carat: float
        x: float
        y: float
        z: float
        cut_encoded: int
        color_encoded: int
        clarity_encoded: int
        depth: float
        table: float

    # Load model at startup
    model_path = "default_repo/transformers/random_forest_(tuned).joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    @app.post("/predict")
    def predict(features: List[DiamondFeatures]):
        try:
            input_features = [[
                f.carat, f.x, f.y, f.z, f.cut_encoded,
                f.color_encoded, f.clarity_encoded, f.depth, f.table
            ] for f in features]
            
            predictions = model.predict(np.array(input_features))
            return {"predicted_price": predictions.tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))




@test
def test_output(output, *args) -> None:
    """
    Tests if the output of the inference block is valid.
    """
    assert isinstance(output, list), "Output should be a list of predictions."
    assert all(isinstance(pred, (float, int)) for pred in output), "Each prediction should be a numerical value."
    assert len(output) == len(default_inputs), "Output length should match input length."
