import pandas as pd

from src.health_policy_recommendation.schemas.request_bodies import UserPreferences3
from src.health_policy_recommendation.model_services import strategy1, strategy2


def fuse_recommendations(similarity_scores, feedback_scores, alpha=0.5):
    """
    Combine content-based similarity scores with utility-based ML predictions.

    Parameters:
        similarity_scores (dict): item_id -> content-based similarity score.
        feedback_scores (dict): item_id -> utility/feedback score from ML model.
        alpha (float): Balance between content-based and ML-based scores.
                       alpha=1: only content-based; alpha=0: only ML-based.

    Returns:
        fused_scores (dict): item_id -> combined fused score.
    """

    # Align the items present in both systems
    item_ids = list(set(similarity_scores) | set(feedback_scores))

    df = pd.DataFrame(
        {
            "item_id": item_ids,
            "similarity_score": [similarity_scores.get(i, 0) for i in item_ids],
            "feedback_score": [feedback_scores.get(i, 0) for i in item_ids],
        }
    )

    # Fuse scores using weighted average
    df["fused_score"] = (
        alpha * df["similarity_score"] + (1 - alpha) * df["feedback_score"]
    )

    # Sort or filter if needed:
    df = df.sort_values(by="fused_score", ascending=False)[:3]

    return dict(zip(df["item_id"], df["fused_score"]))


def combined_recommendation(user_preferences: UserPreferences3):
    user_preferences_1 = user_preferences.user_preferences_1
    user_preferences_2 = user_preferences.user_preferences_2

    # Call the content-based recommendation function
    content_based_result = strategy1.content_based_recommendation(user_preferences_1)

    # Call the model-based recommendation function
    model_based_result = strategy2.model_based_recommendation(user_preferences_2)

    # Combine the results
    return fuse_recommendations(content_based_result, model_based_result, alpha=0.6)
