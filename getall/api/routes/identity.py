"""Identity Federation API – issue IFT, query, bind."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from getall.identity.federation_service import FederationService, IdentityError
from getall.storage.database import get_session

router = APIRouter()


# ── schemas ──────────────────────────────────────────────────────────────


class IssueIftResponse(BaseModel):
    principal_id: str
    ift: str
    agent_identity_id: str


class BindRequest(BaseModel):
    ift: str
    platform: str
    platform_user_id: str


class ResolveRequest(BaseModel):
    platform: str
    platform_user_id: str


# ── deps ─────────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_session)]


# ── routes ───────────────────────────────────────────────────────────────


@router.post("/issue", response_model=IssueIftResponse)
async def issue_ift(session: SessionDep) -> IssueIftResponse:
    svc = FederationService(session)
    ref = await svc.issue_ift()
    return IssueIftResponse(principal_id=ref.principal_id, ift=ref.ift, agent_identity_id=ref.agent_identity_id)


@router.post("/bind", response_model=IssueIftResponse)
async def bind_ift(body: BindRequest, session: SessionDep) -> IssueIftResponse:
    svc = FederationService(session)
    try:
        ref = await svc.bind_ift_to_platform(body.ift, body.platform, body.platform_user_id)
    except IdentityError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IssueIftResponse(principal_id=ref.principal_id, ift=ref.ift, agent_identity_id=ref.agent_identity_id)


@router.post("/resolve", response_model=IssueIftResponse | None)
async def resolve_identity(body: ResolveRequest, session: SessionDep) -> IssueIftResponse | None:
    svc = FederationService(session)
    ref = await svc.resolve_from_platform(body.platform, body.platform_user_id)
    if ref is None:
        return None
    return IssueIftResponse(principal_id=ref.principal_id, ift=ref.ift, agent_identity_id=ref.agent_identity_id)


@router.get("/by-ift/{ift}", response_model=IssueIftResponse | None)
async def get_by_ift(ift: str, session: SessionDep) -> IssueIftResponse | None:
    svc = FederationService(session)
    ref = await svc.get_by_ift(ift)
    if ref is None:
        return None
    return IssueIftResponse(principal_id=ref.principal_id, ift=ref.ift, agent_identity_id=ref.agent_identity_id)
