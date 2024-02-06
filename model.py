from pydantic import BaseModel


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }