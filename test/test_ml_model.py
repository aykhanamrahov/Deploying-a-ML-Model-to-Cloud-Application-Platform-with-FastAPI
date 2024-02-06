import pytest
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data, load_data, categorical_features
from starter.starter.ml.model import inference

DATA_PATH = r"data/census.csv"
MODEL_PATH = r"model/"


@pytest.fixture(scope="module")
def data():
    """
        Load data for test
    """
    return load_data(DATA_PATH)


def test_data_input(data):
    """
        Check data shape
    """
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_data_process(data):
    """
        Check the dataset train and test
    """
    train, test = train_test_split(data, random_state=42, test_size=0.2)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    assert len(X_train) + len(X_test) == len(data)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_model_output(data):
    """
        Test the model inference after training
    """
    model = joblib.load(MODEL_PATH + "model.pkl")
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    y_pred = inference(model, X_test)
    assert len(y_test) == len(y_pred)