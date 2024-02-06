# Script to train machine learning model.
# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml.data import process_data, load_data, slide_performance
from ml.model import train_model, compute_model_metrics, inference
import joblib
import os
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
DATA_PATH = dir + "/starter/data/census.csv"
MODEL_SAVE = dir + "/starter/model/"

# Add code to load in the data.
print("Load dataset")
data = load_data(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
print("Train Test Split")
train, test = train_test_split(data, test_size=0.20)

# Process the train dataset with the process_data function.
print("Process the train dataset with the process_data function.")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=categorical_features, label="salary", training=True
)

# Proces the test data with the process_data function.
print("Proces the test data with the process_data function.")
X_test, y_test, _, _ = process_data(
    test, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
)

#Training, References and Evaluation model
print("Starting training model")
model = train_model(X_train, y_train)
print("Training model finished")
print("References the model trained")
y_pred = inference(model, X_test)
print("Evaluation model trained")
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print("Evaluation result: Precision:{precision} | Recall:{recall} | Fbeta:{fbeta}".format(precision = precision, recall = recall, fbeta = fbeta))

# Calculate the performance of the model on slices of the data
print("Calculate the performance of the model on slices of the data.")
slide_performance(model, test, encoder, lb, categorical_features)

# Save a model
print("Save the model")
joblib.dump(model, MODEL_SAVE + 'model.pkl')
joblib.dump(encoder, MODEL_SAVE + 'encoder.pkl')
joblib.dump(lb, MODEL_SAVE + 'lb.pkl')