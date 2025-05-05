from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import joblib
import os

from src.health_policy_recommendation.schemas.request_bodies import UserPreferences1

cwd = os.getcwd()

label_encoder = joblib.load(
    cwd
    + "/src/health_policy_recommendation/models/content_based/region_budget_encoder.pkl"
)
company_vectors = pd.read_csv(
    cwd + "/src/health_policy_recommendation/models/content_based/company_vectors.csv",
    index_col=0,
)


def feedback_score(row):
    budget_score = {"low": 5, "medium": 3, "high": 1}.get(row["user_budget"], 0)
    smoker_score = {"yes": -1, "no": 1}.get(row["smoker"], 0)
    bmi_score = 1 if row["bmi"] < 25 else -1

    score = budget_score + smoker_score + bmi_score

    # Clip the score between 0 and 5
    return max(0, min(5, score))


def health_score(row):
    score = 2
    if row["bmi"] < 25:
        score += 1
    elif row["bmi"] >= 30:
        score -= 1

    score += 1 if row["smoker"] == "no" else -1
    return score


def create_user_vector(new_user_input: dict, encoder: OneHotEncoder) -> np.ndarray:
    user_input_df = pd.DataFrame([new_user_input])
    user_input_df["user_health_score"] = user_input_df.apply(health_score, axis=1)

    if "user_feedback" not in user_input_df.columns:
        user_input_df["user_feedback"] = user_input_df.apply(feedback_score, axis=1)

    # Encode categorical features
    encoded_cats = encoder.transform(user_input_df[["user_budget", "user_region"]])

    # Scale numerical features
    scaled_health = user_input_df["user_health_score"] / 4.0
    scaled_feedback = user_input_df["user_feedback"] / 5.0

    # Concatenate encoded and scaled features
    user_vector = np.hstack(
        (
            encoded_cats,
            scaled_health.values.reshape(-1, 1),
            scaled_feedback.values.reshape(-1, 1),
        )
    )

    return user_vector


def recomend(new_user_input, company_vectors, label_encoder):
    # create user vector
    user_vector = create_user_vector(new_user_input, label_encoder)

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, company_vectors.values)

    # Map scores back to company names for readability
    similarity_scores = pd.Series(similarities[0], index=company_vectors.index)

    # # Get the top recommendation
    # recommendation = similarity_scores.idxmax() # Get the index (company name) of the max score
    # highest_score = similarity_scores.max()

    # return recommendation, highest_score

    # Display ranked top 3 recommendations
    recommendation = similarity_scores.sort_values(ascending=False).iloc[:3]
    return recommendation.to_dict()


def content_based_recommendation(user_preferences: UserPreferences1):
    return recomend(user_preferences.model_dump(), company_vectors, label_encoder)
