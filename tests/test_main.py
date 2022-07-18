import pytest

from src.bi_lstm_ner import addition


@pytest.fixture
def numbers():
    a = 10
    b = 20
    c = 30
    return [a,b,c]

class TestMain:
    def test_addition(self, numbers):
        res = addition(numbers[0], numbers[1])
        assert res == numbers[2]
