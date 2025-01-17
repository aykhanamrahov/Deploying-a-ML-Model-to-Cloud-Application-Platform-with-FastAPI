import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from .model import compute_model_metrics

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def load_data(path):
    """
        Function to load data from a file given
        Parameters:
            path: Path to the data file: str
        Returns:
            df: Input data frame for model
    """
    # Load data
    df = pd.read_csv(path)
    return df


def slide_performance(model, data, encoder, lb, categorical_features=categorical_features):
    """
        Function to calculate the performance of the model on slices of the data
        Parameters:
            model: ML model
            data: data need calulating the performance
            encoder: sklearn.preprocessing._encoders.OneHotEncoder
            lb: sklearn.preprocessing._label.LabelBinarizer
            categorical_features: the categorical features list
        Returns:
            file: create the file containing the results of the silde performance
    """
    with open('log_slice_performance.txt', 'w') as f:
        for categorical in categorical_features:
            for value in data[categorical].unique():
                data_preformance = data[data[categorical] == value]
                X, y, _, _ = process_data(data_preformance, categorical_features=categorical_features, label="salary",
                                          training=False, encoder=encoder, lb=lb)
                y_pred = model.predict(X)
                precision, recall, fbeta = compute_model_metrics(y, y_pred)
                line = "Feature {value} of the Categorical {cat}: Precision:{precision} | Recall:{recall} | Fbeta:{fbeta}\n".format(
                    value=value, cat=categorical, precision=precision, recall=recall, fbeta=fbeta)
                f.write(line)
    f.close()
