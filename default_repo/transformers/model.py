if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

@transformer
def transform(data, *args, **kwargs):
    df_model_training = data[['carat', 'x', 'y', 'z', 'cut_encoded', 'color_encoded', 'clarity_encoded', 'depth', 'table']]
    df_target = data['price']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(df_model_training, df_target, test_size=0.2, random_state=42)

    # Standardizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models with tuned hyperparameters
    tuned_models = {
        "Ridge Regression (alpha=10)": Ridge(alpha=10),
        "Random Forest (Tuned)": RandomForestRegressor(n_estimators=500, max_depth=50, 
                                                        min_samples_split=8, min_samples_leaf=50, 
                                                        bootstrap=True, oob_score=True, random_state=42)
    }

    # Train and evaluate models
    best_mse = float('inf')
    best_model = None
    best_model_name = None

    for name, model in tuned_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse:.4f}")

        # Save the best model based on MSE
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = name

    # Save the best model
    model_path = f"default_repo/transformers/{best_model_name.replace(' ', '_').lower()}.joblib"
    joblib.dump(best_model, model_path)
    
    print(f"Best model is: {best_model_name}. Saved at: {model_path}")

    return {"best_model_path": model_path, "mse": best_mse}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert 'best_model_path' in output, 'Model path is missing in output'
    assert os.path.exists(output['best_model_path']), 'Saved model file not found'
