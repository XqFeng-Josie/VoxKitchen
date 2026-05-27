def test_operator_api_version_exposed():
    from voxkitchen.operators import OPERATOR_API_VERSION

    assert isinstance(OPERATOR_API_VERSION, int)
    assert OPERATOR_API_VERSION >= 1


def test_operator_api_version_on_base_module():
    from voxkitchen.operators.base import OPERATOR_API_VERSION

    assert isinstance(OPERATOR_API_VERSION, int)
