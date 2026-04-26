"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..auth import authenticate_user, create_access_token, create_user, get_current_user
from ..models.schema import AuthRequest, AuthResponse, UserResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(request: AuthRequest):
    """Register a user and return an access token."""
    user = create_user(request.username, request.password)
    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.post("/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Log in a user and return an access token."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    public_user = {
        "id": user["id"],
        "username": user["username"],
        "created_at": user.get("created_at"),
    }
    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer", "user": public_user}


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    """Return the current user."""
    return current_user
