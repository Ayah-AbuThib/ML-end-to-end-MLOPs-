if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from mage_ai.api.operations import PostOperation
from mage_ai.data_preparation.models.block import Block

@custom
def transform_custom(self, block_uuid: str, payload: dict,*args, **kwargs):

        block = Block.get(block_uuid)
        return block.execute(payload["inputs"])



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
