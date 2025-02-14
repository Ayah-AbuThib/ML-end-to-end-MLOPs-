if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd
from pandas import DataFrame

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
@transformer
def transform(data:DataFrame, *args, **kwargs) -> DataFrame:

    df_model_training = data.drop(columns=['price'])

    df_target=data['price']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(df_model_training, df_target, test_size=0.2, random_state=42)

    # Standardizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models with default hyperparameters
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest (Default)": RandomForestRegressor(random_state=42)
    }

    best_model = None
    best_mse = float('inf')  # Start with an infinitely high MSE to find the minimum

    # Train and evaluate models, track the best one based on MSE
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse:.4f}")
        
        # Update the best model if this one has a lower MSE
        if mse < best_mse:
            best_mse = mse
            best_model = model

    print(f"Best model is: {best_model} ")

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
