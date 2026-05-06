"""
api/schemas.py
--------------
Pydantic models for request / response validation.
FastAPI uses these to:
  - Auto-generate /docs (Swagger UI) and /redoc
  - Validate and coerce incoming JSON
  - Serialize outgoing responses
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class PassengerInput(BaseModel):
    """
    Minimum required: Pclass, Sex, Fare.
    Age, Embarked, SibSp, Parch are optional — missing values are imputed by the pipeline.
    """

    Pclass:   int            = Field(..., ge=1, le=3,
                                     description="Ticket class: 1=First, 2=Second, 3=Third",
                                     json_schema_extra={"example": 3})

    Sex:      str            = Field(...,
                                     description="Passenger sex: 'male' or 'female'",
                                     json_schema_extra={"example": "male"})

    Fare:     float          = Field(..., ge=0,
                                     description="Ticket fare in pre-decimal pounds",
                                     json_schema_extra={"example": 7.25})

    Age:      Optional[float]= Field(None, ge=0, le=120,
                                     description="Age in years. Leave null to use median imputation.",
                                     json_schema_extra={"example": 22.0})

    Embarked: Optional[str]  = Field(None,
                                     description="Port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton",
                                     json_schema_extra={"example": "S"})

    SibSp:    Optional[int]  = Field(None, ge=0, le=10,
                                     description="Number of siblings/spouses aboard",
                                     json_schema_extra={"example": 0})

    Parch:    Optional[int]  = Field(None, ge=0, le=10,
                                     description="Number of parents/children aboard",
                                     json_schema_extra={"example": 0})

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("male", "female"):
            raise ValueError("Sex must be 'male' or 'female'")
        return v

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.upper().strip()
        if v not in ("C", "Q", "S"):
            raise ValueError("Embarked must be 'C', 'Q', or 'S'")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "3rd-class male (low survival chance)",
                    "value": {"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25, "Embarked": "S", "SibSp": 1, "Parch": 0},
                },
                {
                    "summary": "1st-class female (high survival chance)",
                    "value": {"Pclass": 1, "Sex": "female", "Age": 38, "Fare": 71.28, "Embarked": "C", "SibSp": 1, "Parch": 0},
                },
                {
                    "summary": "Minimal input (Age & Embarked omitted)",
                    "value": {"Pclass": 2, "Sex": "female", "Fare": 21.0},
                },
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction:  int   = Field(..., description="0 = Did Not Survive, 1 = Survived")
    probability: float = Field(..., description="Probability of survival (0.0–1.0)")
    label:       str   = Field(..., description="Human-readable result")
    threshold:   float = Field(..., description="Decision threshold applied")
    input:       dict  = Field(..., description="Cleaned input values used for prediction")


class HealthResponse(BaseModel):
    status:      str  = Field(..., description="'ok' or 'degraded'")
    service:     str
    version:     str
    model_ready: bool


class InfoResponse(BaseModel):
    model:            str
    version:          str
    features:         dict
    valid_values:     dict
    training_metrics: dict
