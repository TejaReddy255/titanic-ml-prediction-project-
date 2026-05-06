"""
tests/test_all.py
-----------------
Comprehensive test suite for the Titanic ML v2 system.

Coverage:
  - Feature engineering  (TitanicFeatureEngineer)
  - Preprocessing pipeline  (ColumnTransformer)
  - Input validation  (validate_input)
  - End-to-end prediction  (predict)
  - REST API  (FastAPI test client — requires trained model)

Run:
    python -m pytest tests/ -v
    python tests/test_all.py          # no pytest required
"""

import os
import sys
import json
import math
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features      import TitanicFeatureEngineer, _extract_title
from src.preprocessor  import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.predict       import validate_input

# ── Test runner (no pytest dependency) ────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []

def _test(name: str, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name}")
        print(f"        {e}")


# ── helpers ────────────────────────────────────────────────────────────────────
def _make_raw(**kw) -> pd.DataFrame:
    base = {
        "Pclass": 2, "Sex": "female", "Age": 30.0, "Fare": 21.0,
        "Embarked": "S", "SibSp": 0, "Parch": 0,
        "Name": "Smith, Mrs. Jane",
    }
    base.update(kw)
    return pd.DataFrame([base])


VALID_API = {"Pclass": 3, "Sex": "male", "Age": 22.0, "Fare": 7.25, "Embarked": "S"}


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def test_title_extraction():
    assert _extract_title("Smith, Mr. John")      == "Mr"
    assert _extract_title("Doe, Mrs. Jane")        == "Mrs"
    assert _extract_title("Jones, Miss. Alice")    == "Miss"
    assert _extract_title("Brown, Master. Tommy")  == "Master"
    assert _extract_title("Prof, Dr. Strange")     == "Rare"
    assert _extract_title("No title here")         == "Rare"

def test_family_size_computed():
    eng = TitanicFeatureEngineer()
    df  = _make_raw(SibSp=2, Parch=1)
    out = eng.fit_transform(df)
    assert out["FamilySize"].iloc[0] == 4   # 2 + 1 + 1

def test_is_alone_true():
    eng = TitanicFeatureEngineer()
    out = eng.fit_transform(_make_raw(SibSp=0, Parch=0))
    assert out["IsAlone"].iloc[0] == 1

def test_is_alone_false():
    eng = TitanicFeatureEngineer()
    out = eng.fit_transform(_make_raw(SibSp=1, Parch=0))
    assert out["IsAlone"].iloc[0] == 0

def test_raw_columns_dropped():
    eng = TitanicFeatureEngineer()
    out = eng.fit_transform(_make_raw())
    assert "SibSp"  not in out.columns
    assert "Parch"  not in out.columns
    assert "Name"   not in out.columns

def test_new_columns_added():
    eng = TitanicFeatureEngineer()
    out = eng.fit_transform(_make_raw())
    assert "FamilySize" in out.columns
    assert "IsAlone"    in out.columns
    assert "Title"      in out.columns

def test_title_correct_in_output():
    eng = TitanicFeatureEngineer()
    out = eng.fit_transform(_make_raw(Name="Doe, Mr. John"))
    assert out["Title"].iloc[0] == "Mr"


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def _engineered_df(**kw) -> pd.DataFrame:
    raw = _make_raw(**kw)
    return TitanicFeatureEngineer().fit_transform(raw)

def test_preprocessor_output_shape():
    prep  = build_preprocessor()
    train = pd.concat([_engineered_df(Sex=s, Pclass=p, Embarked=e)
                       for s in ["male","female"]
                       for p in [1,2,3]
                       for e in ["C","Q","S"]])
    prep.fit(train)
    out = prep.transform(_engineered_df())
    assert out.shape[0] == 1
    assert out.shape[1] > 10    # 3 numeric + 1 binary + OHE expansions

def test_preprocessor_no_nans_after_impute():
    prep  = build_preprocessor()
    train = pd.concat([_engineered_df(Sex=s) for s in ["male","female"]])
    prep.fit(train)
    df_missing = _engineered_df()
    df_missing.loc[0, "Age"] = np.nan
    out = prep.transform(df_missing)
    assert not np.isnan(out).any()

def test_preprocessor_unseen_category_safe():
    prep  = build_preprocessor()
    train = pd.concat([_engineered_df(Embarked="S"), _engineered_df(Embarked="C")])
    prep.fit(train)
    novel = _engineered_df(Embarked="Z")    # 'Z' never seen
    out   = prep.transform(novel)           # should not raise
    assert out is not None


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def test_valid_input_passes():
    r = validate_input(VALID_API)
    assert r["Pclass"] == 3
    assert r["Sex"]    == "male"

def test_null_age_allowed():
    r = validate_input({**VALID_API, "Age": None})
    assert r["Age"] is None

def test_null_embarked_allowed():
    r = validate_input({**VALID_API, "Embarked": None})
    assert r["Embarked"] is None

def test_sibsp_defaults_zero():
    r = validate_input({"Pclass": 1, "Sex": "female", "Fare": 50.0})
    assert r["SibSp"] == 0

def test_parch_defaults_zero():
    r = validate_input({"Pclass": 1, "Sex": "female", "Fare": 50.0})
    assert r["Parch"] == 0

def test_name_synthesised_when_absent():
    r = validate_input({"Pclass": 1, "Sex": "female", "Fare": 50.0})
    assert r["Name"] is not None and "," in r["Name"]

def test_invalid_pclass_raises():
    try:
        validate_input({**VALID_API, "Pclass": 5})
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert "Pclass" in str(e)

def test_invalid_sex_raises():
    try:
        validate_input({**VALID_API, "Sex": "alien"})
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert "Sex" in str(e)

def test_invalid_embarked_raises():
    try:
        validate_input({**VALID_API, "Embarked": "X"})
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert "Embarked" in str(e)

def test_negative_fare_raises():
    try:
        validate_input({**VALID_API, "Fare": -1.0})
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert "Fare" in str(e)

def test_age_out_of_range_raises():
    try:
        validate_input({**VALID_API, "Age": 200})
        raise AssertionError("Should have raised")
    except ValueError:
        pass

def test_string_numbers_coerced():
    r = validate_input({**VALID_API, "Age": "22", "Fare": "7.25", "Pclass": "3"})
    assert r["Age"] == 22.0 and r["Pclass"] == 3


# ══════════════════════════════════════════════════════════════════════════════
#  END-TO-END PREDICTION  (model must be trained)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "titanic_pipeline.pkl")
MODEL_PRESENT = os.path.exists(MODEL_PATH)

def test_e2e_prediction_schema():
    if not MODEL_PRESENT:
        print("    [SKIP] no trained model")
        return
    from src.predict import predict
    r = predict(VALID_API)
    assert "prediction"  in r
    assert "probability" in r
    assert "label"       in r
    assert "threshold"   in r
    assert r["prediction"] in (0, 1)
    assert 0 <= r["probability"] <= 1

def test_1st_class_female_higher_than_3rd_male():
    if not MODEL_PRESENT:
        print("    [SKIP] no trained model")
        return
    from src.predict import predict
    fc  = predict({"Pclass": 1, "Sex": "female", "Age": 30, "Fare": 200, "Embarked": "C"})
    low = predict({"Pclass": 3, "Sex": "male",   "Age": 30, "Fare":   7, "Embarked": "S"})
    assert fc["probability"] > low["probability"], \
        f"Expected 1st-class female > 3rd-class male: {fc['probability']} vs {low['probability']}"

def test_null_inputs_handled():
    if not MODEL_PRESENT:
        print("    [SKIP] no trained model")
        return
    from src.predict import predict
    r = predict({"Pclass": 3, "Sex": "male", "Fare": 7.25, "Age": None, "Embarked": None})
    assert r["prediction"] in (0, 1)

def test_family_size_affects_prediction():
    if not MODEL_PRESENT:
        print("    [SKIP] no trained model")
        return
    from src.predict import predict
    alone  = predict({"Pclass":3,"Sex":"male","Fare":8,"SibSp":0,"Parch":0})
    family = predict({"Pclass":3,"Sex":"male","Fare":8,"SibSp":4,"Parch":3})
    # Probabilities should differ (FamilySize has signal)
    assert alone["probability"] != family["probability"]


# ══════════════════════════════════════════════════════════════════════════════
#  API  (FastAPI test client)
# ══════════════════════════════════════════════════════════════════════════════

def _get_client():
    """Return FastAPI TestClient, or None if dependencies unavailable."""
    try:
        from fastapi.testclient import TestClient
        from api.app import app
        return TestClient(app)
    except ImportError:
        return None

def test_api_health():
    client = _get_client()
    if client is None or not MODEL_PRESENT:
        print("    [SKIP] fastapi/httpx not installed or no model")
        return
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_api_predict_200():
    client = _get_client()
    if client is None or not MODEL_PRESENT:
        print("    [SKIP]")
        return
    r = client.post("/predict", json=VALID_API)
    assert r.status_code == 200
    d = r.json()
    assert d["prediction"] in (0, 1)

def test_api_predict_invalid_422():
    client = _get_client()
    if client is None or not MODEL_PRESENT:
        print("    [SKIP]")
        return
    r = client.post("/predict", json={"Pclass": 9, "Sex": "alien", "Fare": -5})
    assert r.status_code == 422

def test_api_docs_available():
    client = _get_client()
    if client is None:
        print("    [SKIP]")
        return
    r = client.get("/docs")
    assert r.status_code == 200

def test_api_info_endpoint():
    client = _get_client()
    if client is None or not MODEL_PRESENT:
        print("    [SKIP]")
        return
    r = client.get("/info")
    assert r.status_code == 200
    assert "training_metrics" in r.json()

def test_api_unknown_returns_404():
    client = _get_client()
    if client is None:
        print("    [SKIP]")
        return
    r = client.get("/nonexistent")
    assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TITANIC ML v2 — TEST SUITE")
    print("=" * 60)

    sections = [
        ("Feature engineering", [
            test_title_extraction, test_family_size_computed,
            test_is_alone_true, test_is_alone_false,
            test_raw_columns_dropped, test_new_columns_added,
            test_title_correct_in_output,
        ]),
        ("Preprocessor", [
            test_preprocessor_output_shape,
            test_preprocessor_no_nans_after_impute,
            test_preprocessor_unseen_category_safe,
        ]),
        ("Input validation", [
            test_valid_input_passes, test_null_age_allowed,
            test_null_embarked_allowed, test_sibsp_defaults_zero,
            test_parch_defaults_zero, test_name_synthesised_when_absent,
            test_invalid_pclass_raises, test_invalid_sex_raises,
            test_invalid_embarked_raises, test_negative_fare_raises,
            test_age_out_of_range_raises, test_string_numbers_coerced,
        ]),
        ("End-to-end prediction", [
            test_e2e_prediction_schema,
            test_1st_class_female_higher_than_3rd_male,
            test_null_inputs_handled,
            test_family_size_affects_prediction,
        ]),
        ("API", [
            test_api_health, test_api_predict_200,
            test_api_predict_invalid_422, test_api_docs_available,
            test_api_info_endpoint, test_api_unknown_returns_404,
        ]),
    ]

    total_pass = total_fail = 0
    for section, tests in sections:
        print(f"\n── {section} {'─'*(45-len(section))}")
        for t in tests:
            _test(t.__name__.replace("test_", ""), t)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{'='*60}")
    print(f"  RESULTS:  {passed} passed  |  {failed} failed  |  {passed+failed} total")
    print(f"{'='*60}")
    if failed:
        print("\nFailed tests:")
        for name, ok, err in _results:
            if not ok:
                print(f"  ✗ {name}: {err}")
