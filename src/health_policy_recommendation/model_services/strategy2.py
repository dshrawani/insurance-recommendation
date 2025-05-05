import os
import pandas as pd

# from catboost import CatBoost
import joblib

from src.health_policy_recommendation.schemas.request_bodies import UserPreferences2


# loaded_model = CatBoost()

cwd = os.getcwd()

# model = loaded_model.load_model(
#     cwd + "/src/health_policy_recommendation/models/insurance_model.cbm"
# )

model = joblib.load(
    cwd + "/src/health_policy_recommendation/models/insurance_model.pkl"
)

model_features = joblib.load(
    cwd + "/src/health_policy_recommendation/models/model_features.pkl"
)

label_enconder = joblib.load(
    cwd + "/src/health_policy_recommendation/models/label_encoder.pkl"
)


# Basic (Static Preferences)


def get_utility_scores_for_user(user_input_df):
    # Example static mapping (normally you'd fetch this from user profile/preferences)

    preferred_region = (
        user_input_df.iloc[0]["preferred_region"] or user_input_df.iloc[0]["region"]
    )

    budget_level = user_input_df.iloc[0]["budget"]  # low / medium / high

    # Define utility rules for companies (this could come from a config or DB)
    company_meta = {
        "Company A": {"region": "northwest", "price": "medium"},
        "Company B": {"region": "southeast", "price": "low"},
        "Company C": {"region": "southwest", "price": "high"},
        "Company D": {"region": "northeast", "price": "medium"},
    }

    utility_scores = {}

    for company, meta in company_meta.items():
        score = 1.0  # base utility

        # Increase utility if region matches
        if meta["region"] == preferred_region:
            score += 0.3

        # Modify based on budget match
        if meta["price"] == budget_level:
            score += 0.2

        utility_scores[company] = score

    return utility_scores


def preprocess_user_input(user_input_df, model_features):
    # One-hot encode like training
    user_input_encoded = pd.get_dummies(
        user_input_df, columns=["sex", "smoker", "region", "budget"], drop_first=True
    )

    # Add missing columns that existed during training
    for col in model_features:
        if col not in user_input_encoded.columns:
            user_input_encoded[col] = 0

    # Reorder to match model input exactly
    user_input_encoded = user_input_encoded[model_features]

    return user_input_encoded


def health_score(row):
    score = 2
    if row["bmi"] < 25:
        score += 1
    elif row["bmi"] >= 30:
        score -= 1

    score += 1 if row["smoker"] == "no" else -1
    return score


def predict(user_input_df, top_n=3):

    user_input_df = pd.DataFrame([user_input_df])

    # Map budget to score
    budget_score = user_input_df["budget"].map({"low": 5, "medium": 3, "high": 1})

    # Compute feedback
    user_input_df["feedback"] = (
        budget_score
        + user_input_df["smoker"].map({"yes": -1, "no": 1})
        + user_input_df["bmi"].apply(lambda x: 1 if x < 25 else -1)
    )

    # Clip to range 0–5
    user_input_df["feedback"] = user_input_df["feedback"].clip(0, 5)

    user_input_df["health_score"] = user_input_df.apply(health_score, axis=1)

    # Preprocess user input to match training features
    user_input_encoded = preprocess_user_input(user_input_df, model_features)

    # Step 1: Predict class probabilities
    probs = model.predict_proba(user_input_encoded)
    class_names = label_enconder.inverse_transform(
        model.classes_
    )  # get original labels

    # Step 2: Get top-N most probable recommendations
    # top_n_probs = sorted(
    #     list(zip(class_names, probs[0])),
    #     key=lambda x: x[1],
    #     reverse=True
    # )[:top_n]

    prob_dict = dict(zip(class_names, probs[0]))  # {'CompanyA': 0.45, ...}

    # Step 3: Sort and get top-N as a dictionary
    top_n_probs = dict(
        sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    # # Step 3: Fetch user-specific utility scores
    # utility_scores = get_utility_scores_for_user(user_input_df)

    # # Step 4: Multiply probabilities with utility scores
    # recommendations = {
    #     company: prob * utility_scores.get(company, 1.0)  # Default utility = 1.0 if not found
    #     for company, prob in top_n_probs.items()
    # }

    # Step 5: Re-rank based on adjusted utility score
    # recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # return recommendations

    return top_n_probs


def model_based_recommendation(user_preferences: UserPreferences2):
    return predict(user_preferences.model_dump(), top_n=3)
