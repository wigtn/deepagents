"""``secret://`` references → environment variables.

Assets only carry references; values live in the environment of the process that calls out.
``secret://records/base-url`` reads ``WIGTN_SECRET_RECORDS_BASE_URL``.
"""

from __future__ import annotations

import os
import re
from typing import Optional

SECRET_PREFIX = "secret://"
ENV_PREFIX = "WIGTN_SECRET_"


def secret_env_name(ref: str) -> str:
    """Environment variable that holds the value of a ``secret://`` reference."""
    if not ref.startswith(SECRET_PREFIX):
        raise ValueError(f"secret:// 참조가 아닙니다: {ref}")
    key = re.sub(r"[^A-Za-z0-9]+", "_", ref[len(SECRET_PREFIX) :]).strip("_").upper()
    if not key:
        raise ValueError(f"빈 secret 참조입니다: {ref}")
    return ENV_PREFIX + key


def resolve_secret(ref: str, *, required: bool = True) -> Optional[str]:
    """Value of a reference. Missing and required → ``LookupError`` naming the variable, never the value."""
    name = secret_env_name(ref)
    value = os.environ.get(name, "").strip()
    if not value and required:
        raise LookupError(f"{ref}: 환경 변수 {name}가 비어 있습니다")
    return value or None
