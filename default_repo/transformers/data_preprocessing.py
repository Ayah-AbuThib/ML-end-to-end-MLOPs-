if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.preprocessing import OrdinalEncoder


@transformer
def transform(df_train, *args, **kwargs):
    encoder = OrdinalEncoder()
    data=df_train[['cut']]
    # transform data
    result = encoder.fit_transform(data)
    df_train['cut_encoded']= result.astype(int)
    data=df_train[['color']]
    # define ordinal encoding
    encoder = OrdinalEncoder()
    # transform data
    result = encoder.fit_transform(data)
    df_train['color_encoded']= result.astype(int)
    df_train[['color','color_encoded']].head(20)
    data=df_train[['clarity']]
    # define ordinal encoding
    encoder = OrdinalEncoder()
    # transform data
    result = encoder.fit_transform(data)
    df_train['clarity_encoded']= result.astype(int)

    df_model_training=df_train[['carat','x','y','z','cut_encoded','color_encoded','clarity_encoded','depth','table','price']]
    return df_model_training


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
