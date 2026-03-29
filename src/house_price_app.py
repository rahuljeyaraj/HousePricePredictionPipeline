import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


TRAIN_PATH = "../data/raw/train.csv"
TEST_PATH = "../data/raw/test.csv"
MODEL_PATH = "../models/house_price_pipeline.pkl"
SUBMISSION_PATH = "../reports/submission.csv"
TARGET_COLUMN = "`SalePrice"


# -----------------------------
# Feature engineering
# -----------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    required_bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if required_bath_cols.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )

    required_sf_cols = {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}
    if required_sf_cols.issubset(df.columns):
        df["TotalSF"] = (
            df["TotalBsmtSF"].fillna(0)
            + df["1stFlrSF"].fillna(0)
            + df["2ndFlrSF"].fillna(0)
        )

    return df


# -----------------------------
# Training utilities
# -----------------------------
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def train_and_save_model(
    train_path: str = TRAIN_PATH,
    model_path: str = MODEL_PATH,
) -> Pipeline:
    train_df = pd.read_csv(train_path)

    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]

    X = add_engineered_features(X)

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    return pipeline


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Train the model first."
        )
    return joblib.load(model_path)


# -----------------------------
# Submission file creation
# -----------------------------
def create_submission(
    test_path: str = TEST_PATH,
    model_path: str = MODEL_PATH,
    submission_path: str = SUBMISSION_PATH,
) -> pd.DataFrame:
    test_df = pd.read_csv(test_path)

    if "Id" not in test_df.columns:
        raise ValueError("Test file must contain an 'Id' column.")

    test_ids = test_df["Id"]
    test_features = add_engineered_features(test_df)

    model = load_model(model_path)
    predictions = model.predict(test_features)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions,
    })

    submission["SalePrice"] = submission["SalePrice"].clip(lower=0)
    submission.to_csv(submission_path, index=False)

    return submission


# -----------------------------
# FastAPI input model
# -----------------------------
class HouseFeatures(BaseModel):
    MSSubClass: Optional[int] = None
    MSZoning: Optional[str] = None
    LotFrontage: Optional[float] = None
    LotArea: Optional[int] = None
    Street: Optional[str] = None
    Alley: Optional[str] = None
    LotShape: Optional[str] = None
    LandContour: Optional[str] = None
    Utilities: Optional[str] = None
    LotConfig: Optional[str] = None
    LandSlope: Optional[str] = None
    Neighborhood: Optional[str] = None
    Condition1: Optional[str] = None
    Condition2: Optional[str] = None
    BldgType: Optional[str] = None
    HouseStyle: Optional[str] = None
    OverallQual: Optional[int] = None
    OverallCond: Optional[int] = None
    YearBuilt: Optional[int] = None
    YearRemodAdd: Optional[int] = None
    RoofStyle: Optional[str] = None
    RoofMatl: Optional[str] = None
    Exterior1st: Optional[str] = None
    Exterior2nd: Optional[str] = None
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: Optional[str] = None
    ExterCond: Optional[str] = None
    Foundation: Optional[str] = None
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: Optional[float] = None
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: Optional[float] = None
    BsmtUnfSF: Optional[float] = None
    TotalBsmtSF: Optional[float] = None
    Heating: Optional[str] = None
    HeatingQC: Optional[str] = None
    CentralAir: Optional[str] = None
    Electrical: Optional[str] = None
    FirstFlrSF: Optional[float] = None
    SecondFlrSF: Optional[float] = None
    LowQualFinSF: Optional[float] = None
    GrLivArea: Optional[float] = None
    BsmtFullBath: Optional[float] = None
    BsmtHalfBath: Optional[float] = None
    FullBath: Optional[int] = None
    HalfBath: Optional[int] = None
    BedroomAbvGr: Optional[int] = None
    KitchenAbvGr: Optional[int] = None
    KitchenQual: Optional[str] = None
    TotRmsAbvGrd: Optional[int] = None
    Functional: Optional[str] = None
    Fireplaces: Optional[int] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: Optional[float] = None
    GarageArea: Optional[float] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: Optional[str] = None
    WoodDeckSF: Optional[int] = None
    OpenPorchSF: Optional[int] = None
    EnclosedPorch: Optional[int] = None
    ThreeSsnPorch: Optional[int] = None
    ScreenPorch: Optional[int] = None
    PoolArea: Optional[int] = None
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: Optional[int] = None
    MoSold: Optional[int] = None
    YrSold: Optional[int] = None
    SaleType: Optional[str] = None
    SaleCondition: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        data = self.model_dump()

        rename_map = {
            "FirstFlrSF": "1stFlrSF",
            "SecondFlrSF": "2ndFlrSF",
            "ThreeSsnPorch": "3SsnPorch",
        }

        normalized = {
            rename_map.get(key, key): value
            for key, value in data.items()
        }

        return pd.DataFrame([normalized])


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="House Price Prediction API")


@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "House Price Prediction API is running"}


@app.post("/train")
def train_model_endpoint() -> Dict[str, str]:
    try:
        train_and_save_model()
        return {"message": f"Model trained and saved to {MODEL_PATH}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
def predict_price(house: HouseFeatures) -> Dict[str, Any]:
    try:
        model = load_model()
        input_df = house.to_dataframe()
        input_df = add_engineered_features(input_df)
        prediction = model.predict(input_df)[0]

        return {"predicted_sale_price": round(float(max(prediction, 0)), 2)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/submission")
def create_submission_endpoint() -> Dict[str, str]:
    try:
        create_submission()
        return {"message": f"Submission file created at {SUBMISSION_PATH}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    pipeline = train_and_save_model()
    submission = create_submission()
    print("Model trained successfully.")
    print(f"Submission file created with {len(submission)} rows.")
