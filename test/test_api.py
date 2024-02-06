from fastapi.testclient import TestClient
from main import app

# Create client test
client = TestClient(app)


# Test init client
def test_init_client():
    """
        Test the server is still running
    """
    respone = client.get('/')
    # Test code respone
    assert respone.status_code == 200
    # Test message respone
    assert respone.json()[0] == "I am still alive ヾ(⌐■_■)ノ♪"


def test_predict_happy_case_1():
    """
        Test model predict with happy case <= 50K
    """
    data_test = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 59496,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2407,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    respone = client.post('/predict', json=data_test)
    # Check response code
    assert respone.status_code == 200
    # Check response predict results of the model
    assert respone.json() == " <=50K"


def test_predict_happy_case_2():
    """
        Test model predict with happy case <= 50K
    """
    data_test = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 29933,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    respone = client.post('/predict', json=data_test)
    # Check response code
    assert respone.status_code == 200
    # Check response predict results of the model
    assert respone.json() == " <=50K"


def test_predict_error_case():
    """
        Test model predict with error case
    """
    data_test = {
        "workclass": "Private",
        "fnlgt": 59496,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "capital_gain": 2407,
        "capital_loss": 0,
        "hours_per_week": 40,
    }
    respone = client.post('/predict', json=data_test)
    assert respone.status_code != 200
    assert "age" not in respone.json()
    assert "education_num" not in respone.json()
    assert "race" not in respone.json()
    assert "sex" not in respone.json()
    assert "native_country" not in respone.json()