from __future__ import annotations

import logging
from collections import defaultdict
from urllib.parse import urlparse

from models import DiscoveryResult

logger = logging.getLogger(__name__)


_SECOND_LEVEL_TLDS = {
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.tr",
    "com.pk",
    "com.ng",
    "co.ke",
    "co.za",
    "co.in",
    "com.sg",
    "com.mx",
    "com.eg",
    "net.au",
    "org.uk",
}


def normalize_domain(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc or parsed.path.split("/")[0]
    host = host.split("@")[-1].split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def root_domain(value: str) -> str:
    host = normalize_domain(value)
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return host
    suffix = ".".join(parts[-2:])
    if suffix in _SECOND_LEVEL_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return suffix


def domain_from_result(result: DiscoveryResult | dict[str, str]) -> str:
    domain = result.domain if isinstance(result, DiscoveryResult) else result.get("domain", "")
    url = result.url if isinstance(result, DiscoveryResult) else result.get("url", "")
    return root_domain(domain or url)


def group_urls_by_domain(
    discovery_results: list[DiscoveryResult] | list[dict[str, str]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for result in discovery_results:
        url = result.url if isinstance(result, DiscoveryResult) else result.get("url", "")
        domain = domain_from_result(result)
        if not url or not domain:
            logger.debug("Skipping discovery result without URL/domain: %s", result)
            continue
        key = (domain, url)
        if key in seen:
            continue
        seen.add(key)
        grouped[domain].append(url)
    logger.info("Grouped %d URLs into %d domains", len(seen), len(grouped))
    return dict(grouped)
