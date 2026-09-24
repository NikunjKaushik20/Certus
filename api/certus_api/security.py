"""API-key auth with roles.

Demo-grade on purpose: keys come from settings and map to a role. The dependency shape is what
matters, so swapping in OIDC/ABDM later touches this file only.

Roles
-----
- ``technician`` — capture images, register patients, open encounters.
- ``ophthalmologist`` — review referrals, override grades.
- ``admin`` — everything, including site and device management.
"""
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, Header, HTTPException, status

from .config import settings

ROLES: tuple[str, ...] = ("technician", "ophthalmologist", "admin")


@dataclass
class Principal:
    """Authenticated caller identity.

    Attributes:
        key: The raw API key string.
        role: One of :data:`ROLES`.
    """

    key: str
    role: str

    @property
    def actor(self) -> str:
        """Short identifier for audit logs — ``role:key_prefix``."""
        return f"{self.role}:{self.key[:8]}"


def current_principal(x_api_key: str = Header(default="")) -> Principal:
    """FastAPI dependency that resolves the caller from the ``X-API-Key`` header.

    Raises:
        HTTPException: 401 if the key is missing or unknown.
    """
    role = settings.keys().get(x_api_key)
    if not role:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or unknown X-API-Key")
    return Principal(key=x_api_key, role=role)


def requires(*allowed: str) -> Callable:
    """Route dependency factory: admin always passes, everyone else must hold one of *allowed*.

    Usage::

        @router.post("/v1/sites", dependencies=[Depends(requires("admin"))])
        def create_site(...): ...
    """

    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if p.role != "admin" and p.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role {p.role} may not do this")
        return p

    return dep
