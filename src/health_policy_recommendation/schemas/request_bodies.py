from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel


class UserBudget(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UserRegion(str, Enum):
    SOUTHWEST = "southwest"
    SOUTHEAST = "southeast"
    NORTHWEST = "northwest"
    NORTHEAST = "northeast"


class UserPreferences1(BaseModel):
    user_budget: UserBudget
    user_region: UserRegion
    bmi: float
    smoker: Literal["yes", "no"]
    user_feedback: Optional[float] = None


class UserPreferences2(BaseModel):
    age: float
    sex: Literal["male", "female"]
    bmi: float
    children: int
    smoker: Literal["yes", "no"]
    region: UserRegion
    budget: UserBudget
    preferred_region: Optional[UserRegion] = None


class UserPreferences3(BaseModel):
    user_preferences_1: UserPreferences1
    user_preferences_2: UserPreferences2
