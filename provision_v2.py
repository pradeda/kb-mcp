#!/usr/bin/env python3
"""Idempotently provision the private v2 client allowlist and local tokens."""

from __future__ import annotations

import argparse
import os
import secrets
import tempfile
import re
from pathlib import Path

import yaml


ENV_PATH = Path("/opt/kb/.env")
CONFIG_PATH = Path("/opt/kb/v2-clients.yml")
ROUTER_PATH = Path("/opt/kb/corpus-router.yml")
TOKEN_KEYS = ("KB_V2_TOKEN_MCP_LOCAL", "KB_V2_TOKEN_KB_CLI_LOCAL")
CLIENTS = {
    "mcp-local": {
        "token_env": "KB_V2_TOKEN_MCP_LOCAL",
        "allowed_corpora": ["homelab", "ai"],
        "allowed_scopes": ["homelab", "ai", "both", "auto"],
    },
    "kb-cli-local": {
        "token_env": "KB_V2_TOKEN_KB_CLI_LOCAL",
        "allowed_corpora": ["homelab", "ai"],
        "allowed_scopes": ["homelab", "ai", "both", "auto"],
    },
}
ROUTER_CONFIG = {
    "router_version": "corpus-router-v1-precalibration",
    "accept_thresholds": {"homelab": 0.60, "ai": 0.60},
    "reject_threshold": 0.40,
    "both_margin": 0.05,
    "dead_zone": {"lower": 0.40, "upper": 0.60},
    "candidate_k": 25,
    "max_distance": {"homelab": 0.60, "ai": 0.60},
    "ai_decay": {"mode": "disabled"},
}


def _atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.stat() if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if previous is not None:
            os.chown(temporary, previous.st_uid, previous.st_gid)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_owner(path: Path) -> None:
    if path.stat().st_uid != os.geteuid():
        raise RuntimeError(f"refusing file not owned by runtime user: {path}")


def _load_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read valid YAML from {path}") from exc


def _env_keys(content: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in TOKEN_KEYS:
            if key in found:
                raise RuntimeError(f"duplicate {key} in environment file")
            value = value.strip().strip('"').strip("'")
            if not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", value):
                raise RuntimeError(f"{key} is not a valid strong local token")
            found[key] = value
    return found


def install(
    env_path: Path = ENV_PATH,
    config_path: Path = CONFIG_PATH,
    router_path: Path = ROUTER_PATH,
) -> dict:
    if not env_path.exists():
        raise RuntimeError(f"refusing to create missing base environment file: {env_path}")
    _require_owner(env_path)
    original = env_path.read_text(encoding="utf-8")
    values = _env_keys(original)
    expected = {"clients": CLIENTS}
    rendered = yaml.safe_dump(expected, sort_keys=False)
    if config_path.exists():
        _require_owner(config_path)
        current = _load_yaml(config_path)
        if current != expected:
            raise RuntimeError(f"refusing to overwrite unexpected client allowlist: {config_path}")
    if router_path.exists():
        _require_owner(router_path)
        current_router = _load_yaml(router_path)
        if current_router != ROUTER_CONFIG:
            raise RuntimeError(f"refusing to overwrite unexpected router config: {router_path}")

    # All existing-file conflicts are checked before the first mutation.
    additions = []
    for key in TOKEN_KEYS:
        if key not in values:
            values[key] = secrets.token_hex(32)
            additions.append(f"{key}={values[key]}")
    if additions:
        separator = "" if original.endswith("\n") else "\n"
        _atomic_write(env_path, original + separator + "\n".join(additions) + "\n")
    else:
        os.chmod(env_path, 0o600)

    if config_path.exists():
        os.chmod(config_path, 0o600)
    else:
        _atomic_write(config_path, rendered)
    rendered_router = yaml.safe_dump(ROUTER_CONFIG, sort_keys=False)
    if router_path.exists():
        os.chmod(router_path, 0o600)
    else:
        _atomic_write(router_path, rendered_router)
    return {
        "tokens_created": len(additions),
        "clients": len(CLIENTS),
        "router_version": ROUTER_CONFIG["router_version"],
    }


def check(
    env_path: Path = ENV_PATH,
    config_path: Path = CONFIG_PATH,
    router_path: Path = ROUTER_PATH,
) -> dict:
    for path in (env_path, config_path, router_path):
        _require_owner(path)
    values = _env_keys(env_path.read_text(encoding="utf-8"))
    missing = sorted(set(TOKEN_KEYS) - set(values))
    if missing:
        raise RuntimeError(f"missing v2 token variables: {missing}")
    document = _load_yaml(config_path)
    if document != {"clients": CLIENTS}:
        raise RuntimeError("v2 client allowlist does not match the approved local clients")
    router = _load_yaml(router_path)
    if router != ROUTER_CONFIG:
        raise RuntimeError("router config does not match the approved pre-calibration contract")
    if any(path.stat().st_mode & 0o077 for path in (env_path, config_path, router_path)):
        raise RuntimeError("v2 environment/config permissions are broader than 0600")
    return {
        "tokens": len(values),
        "clients": len(CLIENTS),
        "router_version": router["router_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--install", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = install() if args.install else check()
    print(yaml.safe_dump(result, sort_keys=True).strip())


if __name__ == "__main__":
    main()
