"""FastAPI service: recommend contact channel for a customer."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.recommend import load_policy, load_reward_model, recommend

app = FastAPI(
    title="Datathon FIAP — Adaptive Offer API",
    description="Recomenda o canal de contato (cellular/telephone) via Thompson Sampling.",
    version="1.0.0",
)

_policy = None
_reward_model = None


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., examples=[30])
    job: str = Field(..., examples=["student"])
    marital: str = Field(..., examples=["single"])
    education: str = Field(..., examples=["professional course"])
    housing: str = Field(..., examples=["yes"])
    loan: str = Field(..., examples=["no"])
    month: str = Field(..., examples=["sep"])
    day_of_week: str = Field(..., examples=["tue"])
    campaign: int = Field(..., examples=[2])
    pdays: int = Field(..., examples=[6])
    previous: int = Field(..., examples=[1])
    poutcome: str = Field(..., examples=["success"])
    emp_var_rate: float = Field(..., alias="emp.var.rate", examples=[-1.1])
    cons_price_idx: float = Field(..., alias="cons.price.idx", examples=[94.199])
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", examples=[-37.5])
    euribor3m: float = Field(..., examples=[0.88])
    nr_employed: float = Field(..., alias="nr.employed", examples=[4963.6])
    never_contacted: int = Field(..., examples=[0])
    previous_success: int = Field(..., examples=[1])
    campaign_bucket: str = Field(..., examples=["few_contacts"])


class RecommendRequest(BaseModel):
    customer: CustomerFeatures
    mode: Literal["exploit", "explore"] = "exploit"
    seed: Optional[int] = None


class RecommendResponse(BaseModel):
    recommended_offer: str
    conversion_probability: float
    conversion_by_arm: dict
    policy_mean_estimates: dict
    algorithm: str
    mode: str
    rationale: str


@app.on_event("startup")
def startup() -> None:
    global _policy, _reward_model
    _policy = load_policy()
    _reward_model = load_reward_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(payload: RecommendRequest) -> dict:
    customer = payload.customer.model_dump(by_alias=True)
    return recommend(
        customer=customer,
        policy=_policy,
        reward_model=_reward_model,
        mode=payload.mode,
        seed=payload.seed,
    )
