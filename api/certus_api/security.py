"""API-key auth with roles.

Demo-grade on purpose: keys come from settings and map to a role. The dependency shape is what
matters, so swapping in OIDC/ABDM later touches this file only.
"""
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from .config import settings

ROLES = ("technician", "ophthalmologist", "admin")


@dataclass
class Principal:
    key: str
    role: str

    @property
    def actor(self) -> str:
        return f"{self.role}:{self.key[:8]}"


def current_principal(x_api_key: str = Header(default="")) -> Principal:
    role = settings.keys().get(x_api_key)
    if not role:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or unknown X-API-Key")
    return Principal(key=x_api_key, role=role)


def requires(*allowed: str):
    """Route dependency: admin always passes, everyone else must hold one of `allowed`."""

    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if p.role != "admin" and p.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role {p.role} may not do this")
        return p

    return dep
