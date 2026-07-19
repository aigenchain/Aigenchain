# src/image_providers.py
"""Image-generation API providers.

A clean, self-contained layer for managing external image-generation
providers (Cloudflare Workers AI, OpenAI-compatible endpoints, ...) through
the Settings UI. This is deliberately separate from the generic "Integrations"
REST store and from the chat ``ModelEndpoint`` table:

  * Integrations = arbitrary third-party REST APIs (RSS, git, home
    automation, notifications). Consumed by the ``api_call`` tool.
  * ModelEndpoint = OpenAI-compatible chat/embedding/image endpoints
    registered for the chat pipeline.
  * Image providers = image-generation services that do NOT speak the
    OpenAI ``/images/generations`` contract (e.g. Cloudflare's
    ``ai/run/{model}``), plus an OpenAI-compatible adapter for migration.

Each provider's config is stored in ``data/image_providers.json`` with secret
fields encrypted at rest (same Fernet scheme as ``secret_storage``). The UI
renders a provider-specific form from ``IMAGE_PROVIDER_DEFS`` so each provider
only shows the fields it actually needs.
"""

import json
import os
import logging
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from core.atomic_io import atomic_write_json
from core.platform_compat import safe_chmod
from src.secret_storage import decrypt, encrypt, is_encrypted
from src.constants import IMAGE_PROVIDERS_FILE

log = logging.getLogger(__name__)

DATA_FILE = IMAGE_PROVIDERS_FILE

# ---------------------------------------------------------------------------
# Provider definitions (drive the dynamic UI form)
# ---------------------------------------------------------------------------
# Each field:
#   key       unique config key
#   label     human label
#   type      "text" | "password"
#   secret    True  -> store encrypted + mask in API responses + render as password
#   required  must be non-empty
#   placeholder / help  optional UI hints
IMAGE_PROVIDER_DEFS: Dict[str, Dict[str, Any]] = {
    "cloudflare": {
        "name": "Cloudflare Workers AI",
        "description": (
            "Generate images via Cloudflare Workers AI using your Account ID, "
            "API Token, and a @cf/... model (e.g. @cf/black-forest-labs/flux-2-klein-9b)."
        ),
        "fields": [
            {
                "key": "account_id",
                "label": "Account ID",
                "type": "text",
                "secret": False,
                "required": True,
                "placeholder": "e2154d9550ef5cec6b5fab9029844e16",
                "help": "Cloudflare account ID (My Account → overview).",
            },
            {
                "key": "api_token",
                "label": "API Token",
                "type": "password",
                "secret": True,
                "required": True,
                "placeholder": "cfut_...",
                "help": "API token with Workers AI / AI Models permission.",
            },
            {
                "key": "model",
                "label": "Model",
                "type": "text",
                "secret": False,
                "required": True,
                "placeholder": "@cf/black-forest-labs/flux-2-klein-9b",
                "help": "Full model ID, including the @cf/ namespace.",
            },
        ],
    },
    "openai_compatible": {
        "name": "OpenAI-compatible",
        "description": (
            "Any endpoint that implements the OpenAI /images/generations contract "
            "(OpenAI, local diffusion servers, gateways, ...)."
        ),
        "fields": [
            {
                "key": "api_base",
                "label": "Base URL",
                "type": "text",
                "secret": False,
                "required": True,
                "placeholder": "https://api.openai.com/v1",
                "help": "Base URL; /images/generations is appended automatically.",
            },
            {
                "key": "api_key",
                "label": "API Key",
                "type": "password",
                "secret": True,
                "required": False,
                "placeholder": "sk-...",
                "help": "Leave blank for keyless local endpoints.",
            },
            {
                "key": "model",
                "label": "Model",
                "type": "text",
                "secret": False,
                "required": True,
                "placeholder": "dall-e-3",
                "help": "Model name the endpoint expects.",
            },
        ],
    },
}

# All config keys that must be encrypted at rest / masked in responses.
_SECRET_KEYS = {
    field["key"]
    for _def in IMAGE_PROVIDER_DEFS.values()
    for field in _def["fields"]
    if field.get("secret")
}


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)


def _mask_value(value: str) -> str:
    if not value:
        return ""
    return f"{str(value)[:4]}****"


def _encrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    copy = dict(config)
    for key in _SECRET_KEYS:
        v = copy.get(key, "")
        if v:
            copy[key] = encrypt(str(v))
    return copy


def _decrypt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    copy = dict(config)
    for key in _SECRET_KEYS:
        v = copy.get(key, "")
        if v:
            copy[key] = decrypt(str(v))
    return copy


def _mask_provider(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Return a UI-safe copy (secrets masked)."""
    safe = dict(provider)
    config = dict(safe.get("config", {}))
    for key in _SECRET_KEYS:
        if config.get(key):
            config[key] = _mask_value(config[key])
    safe["config"] = config
    return safe


def load_providers() -> List[Dict[str, Any]]:
    """Load all providers with secrets decrypted for runtime use."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            providers = json.load(f)
        if not isinstance(providers, list):
            log.error("Invalid image_providers file shape: expected a list")
            return []
        valid = [p for p in providers if isinstance(p, dict)]
        if len(valid) != len(providers):
            log.error("Invalid image_providers rows: ignored non-object entries")
        decrypted = []
        for p in valid:
            copy = dict(p)
            copy["config"] = _decrypt_config(p.get("config", {}))
            decrypted.append(copy)
        return decrypted
    except (json.JSONDecodeError, IOError) as exc:
        log.error("Failed to load image_providers: %s", exc)
        return []


def save_providers(providers: List[Dict[str, Any]]) -> None:
    """Persist providers with secrets encrypted at rest."""
    _ensure_data_dir()
    encrypted = []
    for p in providers:
        copy = dict(p)
        copy["config"] = _encrypt_config(p.get("config", {}))
        encrypted.append(copy)
    atomic_write_json(DATA_FILE, encrypted, indent=2)
    safe_chmod(DATA_FILE, 0o600)


def list_providers_masked() -> List[Dict[str, Any]]:
    return [_mask_provider(p) for p in load_providers()]


def get_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    for p in load_providers():
        if p.get("id") == provider_id:
            return p
    return None


def get_active_provider() -> Optional[Dict[str, Any]]:
    for p in load_providers():
        if p.get("is_active"):
            return p
    return None


def _validate_fields(provider_key: str, config: Dict[str, Any]) -> None:
    """Raise HTTPException(400) if a required field is missing/blank."""
    definition = IMAGE_PROVIDER_DEFS.get(provider_key)
    if not definition:
        raise HTTPException(400, f"Unknown image provider '{provider_key}'")
    for field in definition["fields"]:
        if field.get("required") and not str(config.get(field["key"], "")).strip():
            raise HTTPException(400, f"{field['label']} is required")


def add_provider(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new provider from UI payload.

    Expected payload: { "provider": "cloudflare", "name": "...", "config": {...} }
    """
    provider_key = data.get("provider")
    if provider_key not in IMAGE_PROVIDER_DEFS:
        raise HTTPException(400, f"Unknown image provider '{provider_key}'")
    name = data.get("name", "").strip()
    if not name:
        name = IMAGE_PROVIDER_DEFS[provider_key]["name"]
    config = dict(data.get("config", {}))

    _validate_fields(provider_key, config)

    provider: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "provider": provider_key,
        "name": name,
        "is_active": bool(data.get("is_active", False)),
        "config": config,
    }
    providers = load_providers()
    # Only one active at a time: the first created active wins the flag; later
    # ones must be explicitly activated.
    if provider["is_active"] and any(p.get("is_active") for p in providers):
        provider["is_active"] = False
    providers.append(provider)
    save_providers(providers)
    return provider


def update_provider(provider_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update an existing provider. Returns updated provider or None.

    Secret fields sent as empty/masked are kept unchanged.
    """
    providers = load_providers()
    existing = next((p for p in providers if p.get("id") == provider_id), None)
    if not existing:
        return None

    new_provider_key = data.get("provider", existing.get("provider"))
    if new_provider_key not in IMAGE_PROVIDER_DEFS:
        raise HTTPException(400, f"Unknown image provider '{new_provider_key}'")

    new_config = dict(data.get("config", {}))
    old_config = existing.get("config", {})

    # Preserve unchanged secret fields.
    merged_config: Dict[str, Any] = {}
    for field in IMAGE_PROVIDER_DEFS[new_provider_key]["fields"]:
        key = field["key"]
        incoming = new_config.get(key, "")
        if field.get("secret"):
            if not incoming or "****" in str(incoming):
                merged_config[key] = old_config.get(key, "")
            else:
                merged_config[key] = incoming
        else:
            merged_config[key] = incoming if incoming != "" or key in new_config else old_config.get(key, "")
    _validate_fields(new_provider_key, merged_config)

    existing["provider"] = new_provider_key
    if data.get("name", "").strip():
        existing["name"] = data["name"].strip()
    existing["config"] = merged_config
    save_providers(providers)
    return existing


def delete_provider(provider_id: str) -> bool:
    providers = load_providers()
    original_len = len(providers)
    remaining = [p for p in providers if p.get("id") != provider_id]
    if len(remaining) < original_len:
        save_providers(remaining)
        return True
    return False


def set_active_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    providers = load_providers()
    found = None
    for p in providers:
        if p.get("id") == provider_id:
            p["is_active"] = True
            found = p
        else:
            p["is_active"] = False
    if found is None:
        return None
    save_providers(providers)
    return found


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def _strip_data_uri(b64: str) -> str:
    """Cloudflare returns raw base64; some endpoints return a data: URI."""
    if "," in b64 and b64.strip().lower().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def _generate_cloudflare(config: Dict[str, Any], prompt: str, size: str, timeout: float) -> str:
    account_id = (config.get("account_id") or "").strip()
    api_token = (config.get("api_token") or "").strip()
    model = (config.get("model") or "").strip()
    if not (account_id and api_token and model):
        raise ValueError("Cloudflare provider is missing account_id, api_token, or model")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    auth_header = {"Authorization": f"Bearer {api_token}"}
    json_payload: Dict[str, Any] = {"prompt": prompt, "image_size": size or "1024x1024"}
    # Some Workers AI models (e.g. flux-2 variants) require multipart/form-data
    # rather than a JSON body. Try JSON first; if Cloudflare rejects it asking
    # for multipart, retry as multipart/form-data with the same fields.
    multipart_payload = [("prompt", (None, prompt)), ("image_size", (None, size or "1024x1024"))]

    last_exc = None
    for attempt, (use_multipart,) in enumerate([(False,), (True,)]):
        try:
            with httpx.Client(timeout=timeout) as client:
                if use_multipart:
                    # Drop Content-Type so httpx sets the multipart boundary.
                    headers = dict(auth_header)
                    resp = client.post(url, files=multipart_payload, headers=headers)
                else:
                    headers = dict(auth_header)
                    headers["Content-Type"] = "application/json"
                    resp = client.post(url, json=json_payload, headers=headers)
        except httpx.TimeoutException:
            raise ValueError(f"Cloudflare request timed out after {int(timeout)}s")
        except Exception as exc:  # network / DNS errors
            raise ValueError(f"Cloudflare request failed: {exc}")

        if resp.status_code == 200:
            try:
                data = resp.json()
                result = data.get("result") or {}
                images = result.get("images") or []
                # Some models (e.g. flux-2 variants) return a single base64
                # image under `result.image` instead of an `images` array.
                if not images and result.get("image"):
                    images = [result["image"]]
                if not images:
                    raise ValueError("Cloudflare returned no image data")
                return _strip_data_uri(images[0])
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"Could not parse Cloudflare response: {exc}")

        # Non-200: extract provider error message.
        msg = ""
        try:
            err = resp.json()
            if isinstance(err, dict):
                errors = err.get("errors") or []
                if errors and isinstance(errors[0], dict):
                    msg = errors[0].get("message", "")
                msg = msg or err.get("message", "") or err.get("error", "")
        except Exception:
            pass
        msg = (msg or resp.text or f"HTTP {resp.status_code}")[:300]
        last_exc = ValueError(f"Cloudflare error ({resp.status_code}): {msg}")
        # Retry once as multipart if the model demands it.
        if use_multipart or "multipart" not in msg.lower():
            raise last_exc
    raise last_exc or ValueError("Cloudflare request failed")


def _generate_openai_compatible(config: Dict[str, Any], prompt: str, size: str, timeout: float) -> str:
    api_base = (config.get("api_base") or "").strip().rstrip("/")
    api_key = (config.get("api_key") or "").strip()
    model = (config.get("model") or "").strip()
    if not (api_base and model):
        raise ValueError("OpenAI-compatible provider is missing api_base or model")
    url = f"{api_base}/images/generations"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "prompt": prompt, "n": 1, "size": size or "1024x1024"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        raise ValueError(f"Request timed out after {int(timeout)}s")
    except Exception as exc:
        raise ValueError(f"Request failed: {exc}")
    if resp.status_code != 200:
        msg = ""
        try:
            err = resp.json()
            if isinstance(err, dict):
                e = err.get("error")
                msg = e.get("message", "") if isinstance(e, dict) else str(e)
        except Exception:
            pass
        msg = (msg or resp.text or f"HTTP {resp.status_code}")[:300]
        raise ValueError(f"Provider error ({resp.status_code}): {msg}")
    data = resp.json()
    items = data.get("data", [])
    if not items:
        raise ValueError("Provider returned no image data")
    b64 = items[0].get("b64_json")
    if b64:
        return _strip_data_uri(b64)
    # Fall back to downloading a URL result.
    img_url = items[0].get("url")
    if not img_url:
        raise ValueError("Provider returned no b64/json or url")
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(img_url)
            r.raise_for_status()
            import base64
            return base64.b64encode(r.content).decode("ascii")
    except Exception as exc:
        raise ValueError(f"Failed to download generated image: {exc}")


_ADAPTERS = {
    "cloudflare": _generate_cloudflare,
    "openai_compatible": _generate_openai_compatible,
}


def generate_via_provider(provider: Dict[str, Any], prompt: str, size: str = "1024x1024",
                          quality: str = "medium", timeout: float = 120.0) -> str:
    """Generate an image and return base64-encoded image data.

    ``provider`` is a decrypted provider dict (from load_providers/get_provider).
    Raises ValueError with a human-readable message on failure.
    """
    adapter = _ADAPTERS.get(provider.get("provider"))
    if not adapter:
        raise ValueError(f"Unsupported image provider '{provider.get('provider')}'")
    return adapter(provider.get("config", {}), prompt, size, timeout)


def test_provider(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Run a real (tiny) image generation to validate the full connection.

    Returns { "ok": bool, "message": str }.
    """
    if provider.get("provider") not in _ADAPTERS:
        return {"ok": False, "message": f"Unsupported provider '{provider.get('provider')}'"}
    try:
        b64 = generate_via_provider(
            provider,
            "a tiny blue dot on white background",
            size="512x512",
            quality="low",
            timeout=25.0,
        )
        if not b64:
            return {"ok": False, "message": "Provider returned empty image data"}
        return {
            "ok": True,
            "message": f"Connection OK — generated a test image ({len(b64)} base64 chars).",
        }
    except ValueError as exc:
        return {"ok": False, "message": str(exc)[:300]}
    except Exception as exc:  # defensive
        return {"ok": False, "message": f"Unexpected error: {exc}"[:300]}


# Default Cloudflare vision (image-to-text) model. Understands an image and
# answers questions about it. Distinct from the flux *image generation* model.
# LLaVA needs no license agreement and works out of the box on Workers AI;
# Llama-3.2-Vision requires a one-time "agree" on the account first.
CLOUDFLARE_VISION_MODEL = "@cf/llava-hf/llava-1.5-7b-hf"


def _cf_run_url(account_id: str, model: str) -> str:
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


def _cf_error_message(resp) -> str:
    msg = ""
    try:
        err = resp.json()
        if isinstance(err, dict):
            errors = err.get("errors") or []
            if errors and isinstance(errors[0], dict):
                msg = errors[0].get("message", "")
            msg = msg or err.get("message", "") or err.get("error", "")
    except Exception:
        pass
    return (msg or resp.text or f"HTTP {resp.status_code}")[:300]


def describe_image_via_cloudflare(
    config: Dict[str, Any],
    image_bytes: bytes,
    question: str,
    *,
    vision_model: str = "",
    timeout: float = 120.0,
) -> str:
    """Ask a Cloudflare Workers AI vision model about an image.

    Reuses the active image provider's Cloudflare credentials (account_id +
    api_token) but calls a *vision* model, not the image-generation one.

    Cloudflare vision models take the image as an array of byte values (0-255).
    LLaVA/uform use a ``prompt`` string; Llama-Vision uses a ``messages`` list.
    We try the LLaVA-style ``prompt`` shape first (that is the default model),
    then fall back to the ``messages`` shape for Llama-Vision-style models.

    ``config`` is a DECRYPTED cloudflare provider config. Returns the model's
    text answer, or raises ValueError with a human-readable message.
    """
    account_id = (config.get("account_id") or "").strip()
    api_token = (config.get("api_token") or "").strip()
    model = (vision_model or "").strip() or CLOUDFLARE_VISION_MODEL
    if not (account_id and api_token):
        raise ValueError("Cloudflare provider is missing account_id or api_token")

    url = _cf_run_url(account_id, model)
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    _img_arr = list(image_bytes)
    _q = question or "Describe this image in detail."

    # LLaVA-style (prompt) first, then Llama-Vision-style (messages).
    payloads = [
        {"image": _img_arr, "prompt": _q, "max_tokens": 512},
        {
            "image": _img_arr,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that describes images accurately."},
                {"role": "user", "content": _q},
            ],
        },
    ]

    last_msg = ""
    for i, payload in enumerate(payloads):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise ValueError(f"Cloudflare vision request timed out after {int(timeout)}s")
        except Exception as exc:
            raise ValueError(f"Cloudflare vision request failed: {exc}")

        if resp.status_code == 200:
            try:
                data = resp.json()
                result = data.get("result") or {}
                if isinstance(result, str):
                    text = result
                else:
                    text = result.get("response") or result.get("description") or ""
                if not text:
                    raise ValueError("Cloudflare vision returned no text")
                return text.strip()
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"Could not parse Cloudflare vision response: {exc}")

        last_msg = _cf_error_message(resp)
        # Only retry with the alternate payload shape on a 400 (bad schema);
        # 403/410/etc. are terminal (license/deprecation).
        if resp.status_code != 400 or i == len(payloads) - 1:
            raise ValueError(f"Cloudflare vision error ({resp.status_code}): {last_msg}")

    raise ValueError(f"Cloudflare vision error: {last_msg}")
