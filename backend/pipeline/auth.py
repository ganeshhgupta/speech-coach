"""auth.py — verifies Neon Auth (Managed Better Auth) JWTs. Stateless: the
frontend's cookie-authenticated session mints a short-lived EdDSA JWT via
GET {NEON_AUTH_BASE_URL}/token (see frontend/index.html), and every
protected request here just checks that JWT's signature against Neon's
published JWKS. No callback to Neon per request.
"""
import os
from urllib.parse import urlsplit

import jwt
from fastapi import Header, HTTPException

_jwks_client: "jwt.PyJWKClient | None" = None


def _get_jwks_client() -> "jwt.PyJWKClient":
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(os.environ["NEON_AUTH_JWKS_URL"])
    return _jwks_client


def _issuer_origin() -> str:
    """The JWT's iss/aud claims are the bare origin (scheme://host), while
    NEON_AUTH_BASE_URL includes the auth path (.../neondb/auth) — strip it."""
    parts = urlsplit(os.environ["NEON_AUTH_BASE_URL"])
    return f"{parts.scheme}://{parts.netloc}"


def get_current_user(authorization: str = Header(default=None)) -> str:
    """FastAPI dependency: returns the authenticated user's UUID (the JWT's
    `sub` claim), or raises 401."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token.")
    token = authorization.removeprefix("Bearer ").strip()

    origin = _issuer_origin()
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(token, signing_key.key, algorithms=["EdDSA"], audience=origin, issuer=origin)
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"Invalid or expired token: {e}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(401, "Token missing subject claim.")
    return user_id
