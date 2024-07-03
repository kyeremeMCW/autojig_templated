"""example pytest module"""
from src.python_template.example import example
def test_example():
    """tests pytest"""
    assert example(True) is True
