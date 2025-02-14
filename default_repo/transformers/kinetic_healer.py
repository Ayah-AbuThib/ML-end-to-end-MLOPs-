if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
from pandas import DataFrame
 # from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


@transformer
def transform(data: dict, *args, **kwargs) -> DataFrame:
    scale=StandardScaler()
    x_train_scaled=scale.fit_transform(data)
    random_forest = RandomForestRegressor(n_estimators=500,oob_score = True, max_depth=50,min_samples_split=8,min_samples_leaf=50,random_state=42)
    random_forest.fit(x_train_scaled, df_target)

# # Predict on the test set
    yhat = random_forest.predict(x_train_scaled)
    return yhat





@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
