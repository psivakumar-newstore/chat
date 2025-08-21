# csv_endpoint_tools.py
import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import requests

# Pydantic v1 so it works across LangChain versions
from pydantic.v1 import BaseModel, Field

try:
    from langchain.tools import StructuredTool  # older LC
except Exception:
    from langchain_core.tools import StructuredTool  # newer LC

# ─────────────────────────────────────────────────────────────
# ENV CONFIG
# ─────────────────────────────────────────────────────────────
CSV_PATH = os.getenv("ENDPOINT_CSV_PATH", "URL - for API calls.csv")
BASE_URL = os.getenv("NEWSTORE_BASE_URL", "").rstrip("/")  # e.g. https://retailsuccess-sandbox.p.newstore.net
ADAPTER_URL = os.getenv("ADAPTER_URL", "https://fgsoz6x9t2.execute-api.us-east-1.amazonaws.com/x/adapter")
ADAPTER_API_KEY = os.getenv("CONFIG_ENDPOINT_API_KEY", "")

# Optional: default headers JSON (e.g., {"Authorization":"Bearer ...","x-api-key":"..."} )
DEFAULT_HEADERS_JSON = os.getenv("NEWSTORE_DEFAULT_HEADERS_JSON", "")

# Optional: name of a column that, when the value is "yes", should be ignored (e.g., column E)
IGNORE_COLUMN = os.getenv("IGNORE_COLUMN", "")  # set to your Column E header if you want to skip those rows

# These let the loader understand your CSV even if headers vary a bit
NAME_HEADERS   = ["Doc Name", "Request Name", "Name", "API Name"]
METHOD_HEADERS = ["Method", "HTTP Method"]
URL_HEADERS    = ["URL", "Endpoint", "Path URL", "Full Path", "Path"]

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _discover_columns(header_row):
    if not header_row:
        raise ValueError("CSV appears to have no header row.")

    headers = [h.strip() for h in header_row]
    name_col = method_col = url_col = ignore_col = None

    for h in NAME_HEADERS:
        if h in headers:
            name_col = h
            break
    for h in METHOD_HEADERS:
        if h in headers:
            method_col = h
            break
    for h in URL_HEADERS:
        if h in headers:
            url_col = h
            break
    if IGNORE_COLUMN and IGNORE_COLUMN in headers:
        ignore_col = IGNORE_COLUMN

    if not name_col or not method_col or not url_col:
        raise ValueError(
            f"CSV must include a name({NAME_HEADERS}), method({METHOD_HEADERS}), "
            f"and url({URL_HEADERS}) column. Found: {headers}"
        )
    return name_col, method_col, url_col, ignore_col

class CatalogEntry(BaseModel):
    name: str
    method: str
    url_template: str

def _load_catalog(csv_path: str) -> Dict[Tuple[str, str], CatalogEntry]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    catalog: Dict[Tuple[str, str], CatalogEntry] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        name_col, method_col, url_col, ignore_col = _discover_columns(reader.fieldnames)

        for row in reader:
            if ignore_col and str(row.get(ignore_col, "")).strip().lower() == "yes":
                # Row explicitly marked to ignore
                continue

            name = str(row.get(name_col, "")).strip()
            method = str(row.get(method_col, "")).strip().upper()
            url_tmpl = str(row.get(url_col, "")).strip()

            if not name or not method or not url_tmpl:
                continue

            key = (_norm(name), method)
            catalog[key] = CatalogEntry(name=name, method=method, url_template=url_tmpl)

    if not catalog:
        raise ValueError("CSV loaded but no usable rows found.")
    return catalog

_CATALOG = _load_catalog(CSV_PATH)

def _apply_placeholders(url_template: str, path_params: Dict[str, Any]) -> str:
    """
    Replace placeholders in the URL template:
      - {BASE_URL} → NEWSTORE_BASE_URL
      - {param} → path_params['param']
    If the result is relative (no host), prefix with NEWSTORE_BASE_URL.
    """
    url = url_template

    # Expand {BASE_URL}
    if "{BASE_URL}" in url:
        if not BASE_URL:
            raise RuntimeError("NEWSTORE_BASE_URL not set but CSV uses {BASE_URL}.")
        url = url.replace("{BASE_URL}", BASE_URL)

    # Replace path params {id}, {tenant}, etc.
    for k, v in (path_params or {}).items():
        url = url.replace(f"{{{k}}}", str(v))

    # If still relative, prefix base
    from urllib.parse import urlparse, urljoin
    parsed = urlparse(url)
    if not parsed.netloc:  # relative or bare path
        if not BASE_URL:
            raise RuntimeError(f"Relative URL '{url}' cannot be normalized; NEWSTORE_BASE_URL is not set.")
        base = BASE_URL if BASE_URL.endswith("/") else BASE_URL + "/"
        url = urljoin(base, url.lstrip("/"))

    return url

def _join_query(url: str, query: Dict[str, Any]) -> str:
    if not query:
        return url
    from urllib.parse import urlencode
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode(query, doseq=True)}"

def _merge_default_headers(headers):
    merged: Dict[str, Any] = {}
    if DEFAULT_HEADERS_JSON:
        try:
            merged.update(json.loads(DEFAULT_HEADERS_JSON))
        except Exception:
            pass
    if headers:
        merged.update(headers)
    return merged

def _call_adapter(method: str, url: str, headers: Dict[str, Any], json_body: Optional[Dict[str, Any]]):
    """
    Calls your AWS adapter which handles auth (bearer / x-api-key / etc).
    Expects ADAPTER_API_KEY set. Returns adapter JSON or error info.
    """
    if not ADAPTER_API_KEY:
        raise RuntimeError("CONFIG_ENDPOINT_API_KEY is not set; required for adapter calls.")

    payload = {"url": url, "method": method}
    if headers:
        payload["headers"] = headers
    if json_body is not None:
        payload["payload"] = json_body

    resp = requests.post(
        ADAPTER_URL,
        json=payload,
        headers={"x-api-key": ADAPTER_API_KEY},
        timeout=90,
    )
    try:
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Adapter error: {e}", "statusCode": resp.status_code, "raw": resp.text}
    try:
        return resp.json()
    except Exception:
        return {"statusCode": resp.status_code, "raw": resp.text}

def _resolve_catalog(name: str, method: str) -> CatalogEntry:
    key = (_norm(name), method.upper())
    entry = _CATALOG.get(key)
    if not entry:
        # Helpful error with near matches
        close = sorted({n for (n, m) in _CATALOG.keys() if n.startswith(_norm(name)[:8])})[:10]
        raise ValueError(f"No CSV row for name='{name}' & method='{method}'. Nearby: {close}")
    return entry

def _csv_call(method: str, name: str, path_params: Dict[str, Any], query: Dict[str, Any],
              headers: Dict[str, Any], json_body: Optional[Dict[str, Any]]):
    entry = _resolve_catalog(name, method)
    url = _apply_placeholders(entry.url_template, path_params or {})
    url = _join_query(url, query or {})
    hdrs = _merge_default_headers(headers)
    return _call_adapter(method.upper(), url, hdrs, json_body)

# ─────────────────────────────────────────────────────────────
# Tool schema + tool functions
# ─────────────────────────────────────────────────────────────
class CSVToolInput(BaseModel):
    name: str = Field(..., description="CSV Doc Name (preferred) or Request Name.")
    path_params: Dict[str, Any] = Field(default_factory=dict, description="Values for URL placeholders (e.g., id, tenant).")
    query: Dict[str, Any] = Field(default_factory=dict, description="Querystring parameters.")
    headers: Dict[str, Any] = Field(default_factory=dict, description="Extra headers to merge.")
    json: Optional[Dict[str, Any]] = Field(default=None, description="Body for POST/PUT/PATCH/DELETE.")

def _tool_get(name: str, path_params: Dict[str, Any] = None,
              query: Dict[str, Any] = None, headers: Dict[str, Any] = None,
              json: Optional[Dict[str, Any]] = None):
    return _csv_call("GET", name, path_params or {}, query or {}, headers or {}, None)

def _tool_post(name: str, path_params: Dict[str, Any] = None,
               query: Dict[str, Any] = None, headers: Dict[str, Any] = None,
               json: Optional[Dict[str, Any]] = None):
    return _csv_call("POST", name, path_params or {}, query or {}, headers or {}, json)

def _tool_put(name: str, path_params: Dict[str, Any] = None,
              query: Dict[str, Any] = None, headers: Dict[str, Any] = None,
              json: Optional[Dict[str, Any]] = None):
    return _csv_call("PUT", name, path_params or {}, query or {}, headers or {}, json)

def _tool_patch(name: str, path_params: Dict[str, Any] = None,
                query: Dict[str, Any] = None, headers: Dict[str, Any] = None,
                json: Optional[Dict[str, Any]] = None):
    return _csv_call("PATCH", name, path_params or {}, query or {}, headers or {}, json)

def _tool_delete(name: str, path_params: Dict[str, Any] = None,
                 query: Dict[str, Any] = None, headers: Dict[str, Any] = None,
                 json: Optional[Dict[str, Any]] = None):
    return _csv_call("DELETE", name, path_params or {}, query or {}, headers or {}, json)

csv_get_tool    = StructuredTool.from_function(name="csv_get",    func=_tool_get,    args_schema=CSVToolInput, description="GET endpoint defined in CSV.")
csv_post_tool   = StructuredTool.from_function(name="csv_post",   func=_tool_post,   args_schema=CSVToolInput, description="POST endpoint defined in CSV.")
csv_put_tool    = StructuredTool.from_function(name="csv_put",    func=_tool_put,    args_schema=CSVToolInput, description="PUT endpoint defined in CSV.")
csv_patch_tool  = StructuredTool.from_function(name="csv_patch",  func=_tool_patch,  args_schema=CSVToolInput, description="PATCH endpoint defined in CSV.")
csv_delete_tool = StructuredTool.from_function(name="csv_delete", func=_tool_delete, args_schema=CSVToolInput, description="DELETE endpoint defined in CSV.")

TOOLS = [csv_get_tool, csv_post_tool, csv_put_tool, csv_patch_tool, csv_delete_tool]

if __name__ == "__main__":
    print(f"Loaded {len(_CATALOG)} catalog entries from: {CSV_PATH}")
    print("Tools:", [t.name for t in TOOLS])