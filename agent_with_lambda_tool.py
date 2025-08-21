# agent_with_lambda_tool.py
import os
import json
import textwrap
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any, Optional, Tuple, List

import csv
import requests
import yaml
import httpx
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# StructuredTool import (old/new LC)
try:
    from langchain.tools import StructuredTool  # older LC
except Exception:
    from langchain_core.tools import StructuredTool  # newer LC

# Use Pydantic **v1** shim so LC sees a v1 BaseModel
from pydantic.v1 import BaseModel, Field, validator

# ─────────────────────────────────────────────────────────────
# 1) Load .env FIRST
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 2) Scrub proxy envs (avoid the “proxies” kwarg issue)
# ─────────────────────────────────────────────────────────────
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY",
]:
    os.environ.pop(k, None)

# ─────────────────────────────────────────────────────────────
# 3) Runtime configuration (set from UI)
# ─────────────────────────────────────────────────────────────
_TENANT_SUBDOMAIN: str = os.getenv("TENANT_SUBDOMAIN", "").strip()
_CSV_PATH: str = os.getenv("AGENT_CSV_PATH", "").strip()
_DEFAULT_HEADERS_JSON: str = os.getenv("NEWSTORE_DEFAULT_HEADERS_JSON", "").strip()

# internal caches (loaded lazily)
_ALLOWED_HOSTS: set[str] = set()
_DOC_MAP: Dict[str, Dict[str, Any]] = {}  # doc_name -> row dict
_ALIAS_TO_DOC: Dict[str, str] = {}        # alias(lower) -> doc_name
_ERRORS: List[str] = []

# timeouts
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "25"))
REQUESTS_TIMEOUT = int(os.getenv("REQUESTS_TIMEOUT", "25"))

# ─────────────────────────────────────────────────────────────
# 4) Setters called by UI BEFORE build_agent()
# ─────────────────────────────────────────────────────────────
def set_tenant_base(tenant: str):
    global _TENANT_SUBDOMAIN
    _TENANT_SUBDOMAIN = tenant.strip()

def set_csv_path(csv_path: str):
    global _CSV_PATH
    _CSV_PATH = csv_path.strip()

def set_default_headers_json(headers_json: str):
    global _DEFAULT_HEADERS_JSON
    _DEFAULT_HEADERS_JSON = (headers_json or "").strip()

# ─────────────────────────────────────────────────────────────
# 5) Lazy loaders
# ─────────────────────────────────────────────────────────────
def _yaml_safe_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _extract_allowed_hosts(doc: dict) -> set[str]:
    hosts = set()
    servers = (doc or {}).get("servers") or []
    for s in servers:
        url = s.get("url")
        if not url:
            continue
        try:
            netloc = urlparse(url).netloc
            if netloc:
                hosts.add(netloc.lower())
        except Exception:
            pass
    return hosts

def _load_allowed_hosts_from_yaml_folder(folder: Path) -> set[str]:
    hosts = set()
    if not folder.exists():
        return hosts
    for yfile in sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml"))):
        try:
            spec = _yaml_safe_load(yfile)
            hosts.update(_extract_allowed_hosts(spec))
        except Exception as e:
            _ERRORS.append(f"Skipping {yfile.name}: {e}")
    return hosts

def _load_csv_map(csv_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    doc_map: Dict[str, Dict[str, Any]] = {}
    alias_to_doc: Dict[str, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Doc Name", "Method", "URL"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in reader:
            doc_name = (row.get("Doc Name") or "").strip()
            method = (row.get("Method") or "").strip().upper()
            url = (row.get("URL") or "").strip()
            aliases = (row.get("Alias") or "").strip()
            tag = (row.get("Tag") or "").strip()
            ignore = (row.get("Ignore") or "").strip().lower() in {"yes", "true", "1"}

            if not doc_name or not method or not url or ignore:
                continue

            # tenant placeholder
            if "{BASE_URL}" in url:
                if not _TENANT_SUBDOMAIN:
                    raise ValueError("Tenant is not set; cannot expand {BASE_URL}.")
                base = f"https://{_TENANT_SUBDOMAIN}.p.newstore.net"
                url = url.replace("{BASE_URL}", base)

            doc_map[doc_name] = {
                "doc_name": doc_name,
                "method": method,
                "url": url,
                "tag": tag,
            }

            if aliases:
                for a in [x.strip() for x in aliases.split(",") if x.strip()]:
                    alias_to_doc[a.lower()] = doc_name

    return doc_map, alias_to_doc

# ─────────────────────────────────────────────────────────────
# 6) Request helpers
# ─────────────────────────────────────────────────────────────
def _default_headers() -> Dict[str, str]:
    hdrs: Dict[str, str] = {}
    if _DEFAULT_HEADERS_JSON:
        try:
            hdrs.update(json.loads(_DEFAULT_HEADERS_JSON))
        except Exception as e:
            _ERRORS.append(f"Default headers JSON parse error: {e}")
    return hdrs

def _guard_url(url: str):
    host = (urlparse(url).netloc or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Blocked call to unauthorized host '{host}'. Allowed: {sorted(_ALLOWED_HOSTS)}"
        )

def _requests_call(method: str, url: str, headers=None, params=None, json_body=None, timeout=REQUESTS_TIMEOUT):
    _guard_url(url)
    all_headers = _default_headers()
    if headers:
        all_headers.update(headers)
    if method.upper() == "GET":
        all_headers.setdefault("Cache-Control", "no-cache")
        all_headers.setdefault("Pragma", "no-cache")
    resp = requests.request(
        method=method.upper(),
        url=url,
        headers=all_headers,
        params=params,
        json=json_body,
        timeout=timeout,
    )
    ctype = (resp.headers.get("Content-Type") or "").lower()
    try:
        payload = resp.json() if "application/json" in ctype else resp.text
    except Exception:
        payload = resp.text
    return {"status": resp.status_code, "headers": dict(resp.headers), "body": payload, "url": resp.url}

# ─────────────────────────────────────────────────────────────
# 7) Structured tools (Pydantic v1)
# ─────────────────────────────────────────────────────────────
class GetInput(BaseModel):
    url: str = Field(..., description="Full URL to call. Must belong to an allowed host.")
    headers: Dict[str, Any] = Field(default_factory=dict, description="Optional HTTP headers.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Optional query params for GET.")

class PostInput(BaseModel):
    url: str = Field(..., description="Full URL to call. Must belong to an allowed host.")
    headers: Dict[str, Any] = Field(default_factory=dict, description="Optional HTTP headers.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Optional query params.")
    body: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="json",
        description="JSON body for POST/PATCH/PUT/DELETE (use key 'json' when invoking).",
    )

    # Avoid duplicate validator name clashes on hot-reload:
    @validator("url", allow_reuse=True)
    def _strip_url(cls, v):
        return v.strip()

def api_get_func(url: str, headers: Dict[str, Any] = None, params: Dict[str, Any] = None):
    return _requests_call("GET", url, headers=headers, params=params, json_body=None)

def api_post_func(
    url: str,
    headers: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    json_payload = body if body is not None else kwargs.get("json")
    method = (headers or {}).pop("__method__", "POST").upper()  # trick for PUT/PATCH/DELETE
    if method not in {"POST", "PUT", "PATCH", "DELETE"}:
        method = "POST"
    return _requests_call(method, url, headers=headers, params=params, json_body=json_payload)

# Generic tools – always available
api_get_tool = StructuredTool.from_function(
    name="api_get",
    description="Perform a GET request to whitelisted hosts.",
    func=api_get_func,
    args_schema=GetInput,
)
api_post_tool = StructuredTool.from_function(
    name="api_post",
    description="Perform a POST/PUT/PATCH/DELETE request to whitelisted hosts (method via headers['__method__']).",
    func=api_post_func,
    args_schema=PostInput,
)

# ─────────────────────────────────────────────────────────────
# 8) CSV‑backed “call by name” tool (maps aliases/Doc Name → method+URL)
# ─────────────────────────────────────────────────────────────
class CallByNameInput(BaseModel):
    name: str = Field(..., description="Doc Name or Alias (case-insensitive).")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters.")
    headers: Dict[str, Any] = Field(default_factory=dict, description="Extra headers.")
    body: Optional[Dict[str, Any]] = Field(default=None, alias="json", description="JSON body, if needed.")

    @validator("name", allow_reuse=True)
    def _strip_it(cls, v):
        return (v or "").strip()

def _resolve_doc_or_alias(name: str) -> Dict[str, Any]:
    key = name.lower().strip()
    if key in _ALIAS_TO_DOC:
        doc = _ALIAS_TO_DOC[key]
    else:
        # try exact Doc Name
        doc = None
        for dn in _DOC_MAP.keys():
            if dn.lower() == key:
                doc = dn
                break
    if not doc:
        raise ValueError(f"No endpoint found for '{name}'. Check 'Doc Name' or 'Alias' in your CSV.")
    return _DOC_MAP[doc]

def call_by_name(name: str, params: Dict[str, Any] = None, headers: Dict[str, Any] = None, body: Optional[Dict[str, Any]] = None):
    row = _resolve_doc_or_alias(name)
    method = row["method"].upper()
    url = row["url"]
    hdrs = {"__method__": method}
    if headers:
        hdrs.update(headers)
    if method == "GET":
        return _requests_call("GET", url, headers=hdrs, params=params, json_body=None)
    else:
        return _requests_call(method, url, headers=hdrs, params=params, json_body=body)

call_by_name_tool = StructuredTool.from_function(
    name="call_by_name",
    description="Call an endpoint by Doc Name/Alias from the CSV (maps to method+URL automatically).",
    func=call_by_name,
    args_schema=CallByNameInput,
)

# ─────────────────────────────────────────────────────────────
# 9) Build agent (lazy load CSV + YAMLs here; keep fast)
# ─────────────────────────────────────────────────────────────
def _make_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    httpx_client = httpx.Client(timeout=OPENAI_TIMEOUT, trust_env=False)
    openai_client = OpenAI(api_key=api_key, http_client=httpx_client)

    # IMPORTANT: pass the .chat.completions client so LC uses httpx with our timeout
    return ChatOpenAI(client=openai_client.chat.completions, model="gpt-4o", temperature=0)

def build_agent():
    """Called by UI after set_tenant_base/set_csv_path/set_default_headers_json."""
    _ERRORS.clear()
    # 1) load allowed hosts from openapi-yaml (if present)
    global _ALLOWED_HOSTS
    spec_dir = Path("openapi-yaml")
    _ALLOWED_HOSTS = _load_allowed_hosts_from_yaml_folder(spec_dir)

    # Always allow the tenant base if set (even if not present in yaml)
    if _TENANT_SUBDOMAIN:
        _ALLOWED_HOSTS.add(f"{_TENANT_SUBDOMAIN}.p.newstore.net")
        _ALLOWED_HOSTS.add(f"{_TENANT_SUBDOMAIN}.s.newstore.net")

    # 2) load CSV map
    if not _CSV_PATH:
        raise RuntimeError("CSV path not set.")
    csv_file = Path(_CSV_PATH)
    if not csv_file.exists():
        raise RuntimeError(f"CSV not found: {csv_file}")

    global _DOC_MAP, _ALIAS_TO_DOC
    _DOC_MAP, _ALIAS_TO_DOC = _load_csv_map(str(csv_file))

    # 3) build agent
    tools = [call_by_name_tool, api_get_tool, api_post_tool]
    llm = _make_llm()
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
    )
    return agent

# ─────────────────────────────────────────────────────────────
# 10) Diagnostics helpers
# ─────────────────────────────────────────────────────────────
def debug_snapshot() -> Dict[str, Any]:
    return {
        "tenant": _TENANT_SUBDOMAIN,
        "csv_path": _CSV_PATH,
        "default_headers_json": _DEFAULT_HEADERS_JSON,
        "allowed_hosts_count": len(_ALLOWED_HOSTS),
        "doc_count": len(_DOC_MAP),
        "alias_count": len(_ALIAS_TO_DOC),
        "errors": list(_ERRORS),
        "example_hosts": sorted(list(_ALLOWED_HOSTS))[:6],
        "example_docs": sorted(list(_DOC_MAP.keys()))[:6],
    }