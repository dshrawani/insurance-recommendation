from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from src.health_policy_recommendation.model_services import (
    strategy1,
    strategy2,
    strategy3,
)
from src.health_policy_recommendation.utils.logger import get_logger
from src.health_policy_recommendation.schemas.request_bodies import (
    UserPreferences1,
    UserPreferences2,
    UserPreferences3,
)


app = FastAPI(
    title="Health Policy Recommendation APIs",
)

log = get_logger(name="health_policy_recommendation", log_file="logs/logs.txt")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/recommendations/content-based")
def content_based_recommendation(user_preferences: UserPreferences1):
    try:
        result = strategy1.content_based_recommendation(user_preferences)
        return {"status": "success", "details": result}
    except Exception as e:
        log.error(msg=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/recommendations/model-based")
def model_based_recommendation(user_preferences: UserPreferences2):
    try:
        result = strategy2.model_based_recommendation(user_preferences)
        return {"status": "success", "details": result}
    except Exception as e:
        log.error(msg=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/recommendations/combined")
def combined_recommendation(user_preferences: UserPreferences3):
    try:
        result = strategy3.combined_recommendation(user_preferences)
        return {"status": "success", "details": result}
    except Exception as e:
        log.error(msg=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
