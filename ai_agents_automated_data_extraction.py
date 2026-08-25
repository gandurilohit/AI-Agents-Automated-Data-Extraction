"""
AI Agents & Automated Data Extraction
=====================================

What is grounded in the original work
-------------------------------------
1) Web scraping:
   - Playwright + BeautifulSoup
   - OLIPOP, Poppi, Coca-Cola Simply Pop
   - 10 output attributes
   - original run produced 42 list/product-page records

2) LLM/API data collection:
   - repeated shopping-prompt API calls
   - JSON persistence with prompt/run/timestamp/model/raw response metadata
   - original project summary records 98 successful JSON outputs

3) AI-agent / scientific workflow:
   - the original assignment asked for a Paper2Agent/AlphaGenome agent
   - the AlphaGenome execution itself could not be completed in the original
     environment because the required external runtime/API access was unavailable
   - this file therefore includes the reproducible launcher/run-plan helper,
     rather than pretending an AlphaGenome prediction was executed

IMPORTANT
---------
This is a consolidated portfolio version. It preserves the actual scraper logic
from the submission and packages the API/agent workflow into one readable file.

Install:
    pip install pandas beautifulsoup4 playwright nest_asyncio requests openai
    playwright install chromium

Environment variables (only needed for the corresponding feature):
    OPENAI_API_KEY=...
    ALPHAGENOME_API_KEY=...

Examples:
    python ai_agents_automated_data_extraction.py scrape
    python ai_agents_automated_data_extraction.py llm --runs 10
    python ai_agents_automated_data_extraction.py stats
    python ai_agents_automated_data_extraction.py variant --variant rs11174281
    python ai_agents_automated_data_extraction.py alphagenome-plan
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SCRAPE_OUTPUT = OUTPUT_DIR / "task1_output.csv"
LLM_OUTPUT_DIR = OUTPUT_DIR / "llm_json"
LLM_STATS_OUTPUT = OUTPUT_DIR / "llm_response_stats.csv"
ALPHAGENOME_OUTPUT_DIR = OUTPUT_DIR / "rs11174281_alphagenome"

SCRAPE_COLUMNS = [
    "URL",
    "list/product page",
    "product name",
    "price",
    "discounted price",
    "position",
    "number of photos",
    "flavor",
    "ingredients",
    "nutrition facts",
]

EXCLUDE_TERMS = [
    "gift card",
    "merch",
    "hat",
    "shirt",
    "socks",
    "bundle",
    "variety pack",
    "subscription",
]

# The assignment required 10 shopping-related prompts. Keep the prompt list
# editable: replace these with the exact sourced prompt list when reproducing
# the original collection.
SHOPPING_PROMPTS = [
    "What are the best running shoes for everyday training?",
    "What is the best laptop for a college student?",
    "What are the best noise-cancelling headphones?",
    "What is the best budget smartphone to buy?",
    "What is the best robot vacuum for pet hair?",
    "What are the best skincare products for dry skin?",
    "What is the best coffee maker for home use?",
    "What is the best carry-on luggage for frequent travel?",
    "What are the best wireless earbuds for working out?",
    "What is the best office chair for working from home?",
]

# Lightweight dictionaries used only for reproducible response statistics.
# They can be expanded without changing the collection pipeline.
KNOWN_BRANDS = {
    "adidas", "apple", "asus", "bose", "brooks", "cerave", "dell", "dyson",
    "google", "hp", "jbl", "lenovo", "lg", "nike", "roborock", "samsung",
    "sony", "the ordinary", "away", "samsonite", "aeropress", "keurig",
    "nespresso", "new balance", "hoka", "jabra", "anker", "beats", "irobot",
}

PRODUCT_FEATURE_TERMS = {
    "battery", "battery life", "price", "weight", "durability", "comfort",
    "fit", "support", "cushioning", "camera", "display", "screen", "processor",
    "storage", "memory", "ram", "noise cancellation", "anc", "sound quality",
    "water resistance", "wireless", "bluetooth", "warranty", "capacity",
    "suction", "navigation", "mapping", "ingredients", "hydration", "size",
    "ergonomics", "lumbar support", "adjustability", "portability",
}


# =============================================================================
# PART 1 — WEB SCRAPING
# Original Playwright + BeautifulSoup logic consolidated from the submission.
# =============================================================================

def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("Â®", "®").replace("Â", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def name_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").title()


def empty_product_fields() -> dict[str, str]:
    return {
        "number of photos": "",
        "flavor": "",
        "ingredients": "",
        "nutrition facts": "",
    }


def is_real_product(name: str, url: str) -> bool:
    combined = f"{name} {url}".lower()
    return not any(term in combined for term in EXCLUDE_TERMS)


async def get_soup(page: Any, url: str, wait_ms: int = 5000) -> BeautifulSoup:
    await page.goto(url, wait_until="domcontentloaded", timeout=90000)
    await page.wait_for_timeout(wait_ms)
    html = await page.content()
    return BeautifulSoup(html, "html.parser")


def get_prices(text: str) -> tuple[str, str]:
    """
    Extract prices visible on the brand page only.
    Does not follow marketplace links.

    Returns:
      (regular_price, discounted_price)
    """
    text = clean_text(text)
    prices = re.findall(r"\$\s?\d+(?:\.\d{2})?", text)
    prices = [p.replace(" ", "") for p in prices]

    blocked = {"$50", "$75", "$100"}
    cleaned: list[tuple[float, str]] = []

    for p in prices:
        if p in blocked:
            continue
        try:
            value = float(p.replace("$", ""))
            if 0 < value < 100:
                cleaned.append((value, p))
        except ValueError:
            continue

    if not cleaned:
        return "", ""

    unique = sorted(set(cleaned))
    if len(unique) == 1:
        return unique[0][1], ""

    low = unique[0][1]
    high = unique[-1][1]
    if unique[0][0] < unique[-1][0]:
        return high, low

    return unique[0][1], ""


def extract_between(text: str, starts: list[str], ends: list[str]) -> str:
    text = clean_text(text)

    for start in starts:
        start_match = re.search(re.escape(start), text, flags=re.I)
        if not start_match:
            continue

        extracted = text[start_match.end():]
        end_positions = []

        for end in ends:
            end_match = re.search(re.escape(end), extracted, flags=re.I)
            if end_match:
                end_positions.append(end_match.start())

        if end_positions:
            extracted = extracted[: min(end_positions)]

        return clean_text(extracted)

    return ""


def photo_count(soup: BeautifulSoup, brand: str, product_name: str) -> int | str:
    urls: set[str] = set()
    product_words = [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", product_name)
        if len(w) > 2
    ]

    bad_terms = [
        "logo", "icon", "footer", "header", "star", "review", "avatar",
        "payment", "sprite", "social", "klarna", "afterpay",
    ]

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("srcset") or ""
        alt = clean_text(img.get("alt", "")).lower()

        if not src:
            continue

        src_low = src.lower()
        if any(term in src_low or term in alt for term in bad_terms):
            continue

        if brand == "olipop":
            if any(word in alt or word in src_low for word in product_words):
                urls.add(src.split("?")[0])

        elif brand == "poppi":
            if "cdn.shopify" in src_low and any(
                word in alt or word in src_low for word in product_words
            ):
                urls.add(src.split("?")[0])

        elif brand == "coca":
            if (
                "simply" in src_low
                or "simply" in alt
                or "pop" in src_low
                or "pop" in alt
            ):
                urls.add(src.split("?")[0])

    return len(urls) if urls else ""


async def collect_product_links(
    page: Any,
    list_url: str,
    base_url: str,
    max_products: int = 10,
) -> list[dict[str, str]]:
    soup = await get_soup(page, list_url)
    products: list[dict[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/products/" not in href:
            continue

        url = urljoin(base_url, href).split("?")[0].split("#")[0]
        if url in seen:
            continue

        raw_name = clean_text(a.get_text(" "))
        raw_name = raw_name.replace("Sale", "").replace("Sold Out", "").strip()
        name = raw_name if raw_name else name_from_url(url)

        if not is_real_product(name, url):
            continue

        products.append({"url": url, "name": name})
        seen.add(url)

        if len(products) == max_products:
            break

    return products


async def scrape_olipop(page: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    list_url = "https://drinkolipop.com/"
    base_url = "https://drinkolipop.com"

    products = await collect_product_links(page, list_url, base_url, 10)

    for position, product in enumerate(products, start=1):
        product_soup = await get_soup(page, product["url"])
        text = clean_text(product_soup.get_text(" "))
        price, discounted_price = get_prices(text)

        rows.append({
            "URL": list_url,
            "list/product page": "list page",
            "product name": product["name"],
            "price": price,
            "discounted price": discounted_price,
            "position": position,
            **empty_product_fields(),
        })

        h1 = product_soup.find("h1")
        product_name = clean_text(h1.get_text(" ")) if h1 else product["name"]

        ingredients = extract_between(
            text,
            ["Ingredients:"],
            [
                "Nutrition Facts", "Nutrition", "Gently Swirl",
                "You're Going to Love", "Benefits", "Reviews",
            ],
        )

        nutrition = extract_between(
            text,
            ["Nutrition Facts", "Nutrition"],
            [
                "Ingredients:", "Ingredients", "Gently Swirl",
                "You're Going to Love", "Reviews", "Sign up",
            ],
        )

        rows.append({
            "URL": product["url"],
            "list/product page": "product page",
            "product name": product_name,
            "price": "",
            "discounted price": "",
            "position": "",
            "number of photos": photo_count(product_soup, "olipop", product_name),
            "flavor": product_name,
            "ingredients": ingredients,
            "nutrition facts": nutrition,
        })

    return rows


async def scrape_poppi(page: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    list_url = "https://drinkpoppi.com/collections/drinks"
    base_url = "https://drinkpoppi.com"

    products = await collect_product_links(page, list_url, base_url, 10)

    for position, product in enumerate(products, start=1):
        product_soup = await get_soup(page, product["url"])
        text = clean_text(product_soup.get_text(" "))
        price, discounted_price = get_prices(text)

        rows.append({
            "URL": list_url,
            "list/product page": "list page",
            "product name": product["name"],
            "price": price,
            "discounted price": discounted_price,
            "position": position,
            **empty_product_fields(),
        })

        h1 = product_soup.find("h1")
        product_name = clean_text(h1.get_text(" ")) if h1 else product["name"]

        ingredients = extract_between(
            text,
            ["Ingredients:"],
            [
                "Contains 5% Juice", "Taste The Obsession",
                "Nutrition Facts", "FAQ", "Reviews",
            ],
        )

        nutrition = ""
        start = re.search(r"Serving Size\s+1 can", text, flags=re.I)
        end = re.search(r"Ingredients:", text, flags=re.I)

        if start and end and end.start() > start.start():
            nutrition = clean_text(text[start.start():end.start()])
        else:
            nutrition = extract_between(
                text,
                ["Nutrition Facts", "Serving Size"],
                ["Ingredients:", "Ingredients", "FAQ", "Reviews"],
            )

        rows.append({
            "URL": product["url"],
            "list/product page": "product page",
            "product name": product_name,
            "price": "",
            "discounted price": "",
            "position": "",
            "number of photos": photo_count(product_soup, "poppi", product_name),
            "flavor": product_name,
            "ingredients": ingredients,
            "nutrition facts": nutrition,
        })

    return rows


async def scrape_coca(page: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    url = "https://www.coca-cola.com/us/en/brands/simply/products/pop"

    soup = await get_soup(page, url, wait_ms=7000)
    text = clean_text(soup.get_text(" "))
    price, discounted_price = get_prices(text)

    rows.append({
        "URL": url,
        "list/product page": "list page",
        "product name": "Simply® Pop",
        "price": price,
        "discounted price": discounted_price,
        "position": 1,
        **empty_product_fields(),
    })

    ingredients = extract_between(
        text,
        ["Ingredients"],
        [
            "Nutrition Facts", "Nutrition", "Calories",
            "More Products", "Explore More",
        ],
    )

    nutrition = extract_between(
        text,
        ["Nutrition Facts", "Nutrition"],
        ["Ingredients", "More Products", "Explore More", "Products"],
    )

    rows.append({
        "URL": url,
        "list/product page": "product page",
        "product name": "Simply® Pop",
        "price": "",
        "discounted price": "",
        "position": "",
        "number of photos": photo_count(soup, "coca", "Simply Pop"),
        "flavor": "Simply® Pop",
        "ingredients": ingredients,
        "nutrition facts": nutrition,
    })

    return rows


async def run_scraper() -> pd.DataFrame:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required. Run: pip install playwright && "
            "playwright install chromium"
        ) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        page = await browser.new_page(
            viewport={"width": 1440, "height": 1200},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
        )

        print("Scraping OLIPOP...")
        all_rows.extend(await scrape_olipop(page))

        print("Scraping Poppi...")
        all_rows.extend(await scrape_poppi(page))

        print("Scraping Coca-Cola Simply Pop...")
        all_rows.extend(await scrape_coca(page))

        await browser.close()

    df = pd.DataFrame(all_rows, columns=SCRAPE_COLUMNS)
    df.to_csv(SCRAPE_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} records -> {SCRAPE_OUTPUT}")
    return df


# =============================================================================
# PART 2 — REPEATED LLM API DATA COLLECTION
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, max_len: int = 45) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "prompt"


def response_text_from_openai(response: Any) -> str:
    # Newer OpenAI SDK exposes output_text directly.
    text = getattr(response, "output_text", None)
    if text:
        return text

    # Defensive fallback for serializable response structures.
    try:
        data = response.model_dump()
    except Exception:
        data = {}

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def call_llm(prompt: str, model: str = "gpt-5-mini") -> dict[str, Any]:
    """
    Execute one LLM request and return a JSON-serializable record.

    API keys are read from OPENAI_API_KEY and never stored in output files.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK required: pip install openai") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running LLM collection.")

    client = OpenAI()
    started = time.time()
    response = client.responses.create(model=model, input=prompt)
    elapsed = round(time.time() - started, 3)

    try:
        raw = response.model_dump()
    except Exception:
        raw = {"repr": repr(response)}

    return {
        "prompt": prompt,
        "model": model,
        "timestamp_utc": utc_now(),
        "latency_seconds": elapsed,
        "response_text": response_text_from_openai(response),
        "raw_api_output": raw,
    }


def collect_llm_runs(
    prompts: list[str],
    runs: int = 10,
    model: str = "gpt-5-mini",
    delay_seconds: float = 0.0,
) -> list[Path]:
    """
    Run every prompt repeatedly and save each response as an individual JSON file.

    The original assignment specified 10 prompts x 10 occasions. The project
    summary records 98 successful JSON outputs; this function is designed so
    failed calls remain visible instead of being silently replaced.
    """
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for prompt_index, prompt in enumerate(prompts, start=1):
        for run_index in range(1, runs + 1):
            stem = f"p{prompt_index:02d}_r{run_index:02d}_{slugify(prompt)}"
            output_path = LLM_OUTPUT_DIR / f"{stem}.json"

            try:
                record = call_llm(prompt, model=model)
                record.update({
                    "prompt_index": prompt_index,
                    "run_index": run_index,
                    "status": "success",
                })
            except Exception as exc:
                record = {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "run_index": run_index,
                    "model": model,
                    "timestamp_utc": utc_now(),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            output_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(output_path)
            print(f"[{len(written):03d}] {record['status']}: {output_path.name}")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return written


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def find_terms(text: str, terms: set[str]) -> list[str]:
    lowered = text.lower()
    return sorted(term for term in terms if term.lower() in lowered)


def build_llm_stats(directory: Path = LLM_OUTPUT_DIR) -> pd.DataFrame:
    """
    Compute per-file statistics requested in the assignment:
      - word count
      - brands mentioned
      - product-feature terms mentioned
    """
    rows: list[dict[str, Any]] = []

    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({
                "file": path.name,
                "status": "invalid_json",
                "error": str(exc),
            })
            continue

        text = str(payload.get("response_text", ""))
        brands = find_terms(text, KNOWN_BRANDS)
        features = find_terms(text, PRODUCT_FEATURE_TERMS)

        rows.append({
            "file": path.name,
            "status": payload.get("status", "unknown"),
            "prompt_index": payload.get("prompt_index"),
            "run_index": payload.get("run_index"),
            "timestamp_utc": payload.get("timestamp_utc"),
            "model": payload.get("model"),
            "word_count": count_words(text),
            "brand_count": len(brands),
            "brands": "; ".join(brands),
            "product_feature_count": len(features),
            "product_features": "; ".join(features),
        })

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LLM_STATS_OUTPUT, index=False)

    successful = int((df.get("status") == "success").sum()) if not df.empty else 0
    print(f"JSON files found: {len(df)}")
    print(f"Successful API outputs: {successful}")
    print(f"Saved statistics -> {LLM_STATS_OUTPUT}")
    return df


# =============================================================================
# PART 3 — MODULAR AGENT-STYLE SCIENTIFIC WORKFLOW
# =============================================================================

class RetrievalAgent:
    """Retrieve external variant metadata from the Ensembl REST API."""

    BASE = "https://rest.ensembl.org"

    def lookup_variant(self, variant_id: str) -> dict[str, Any]:
        url = f"{self.BASE}/variation/human/{variant_id}"
        response = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


class ProcessingAgent:
    """Normalize raw API output into a compact structured representation."""

    @staticmethod
    def normalize_variant(raw: dict[str, Any], variant_id: str) -> dict[str, Any]:
        mappings = raw.get("mappings", []) or []
        grch38 = [
            m for m in mappings
            if str(m.get("assembly_name", "")).upper() == "GRCH38"
        ]

        selected = grch38[0] if grch38 else (mappings[0] if mappings else {})

        return {
            "variant_id": variant_id,
            "name": raw.get("name", variant_id),
            "ancestral_allele": raw.get("ancestral_allele"),
            "minor_allele": raw.get("minor_allele"),
            "minor_allele_freq": raw.get("MAF"),
            "assembly": selected.get("assembly_name"),
            "chromosome": selected.get("seq_region_name"),
            "start": selected.get("start"),
            "end": selected.get("end"),
            "allele_string": selected.get("allele_string"),
            "strand": selected.get("strand"),
            "clinical_significance": raw.get("clinical_significance", []),
            "source": raw.get("source"),
            "raw_mapping_count": len(mappings),
        }


class ValidationAgent:
    """Run simple semantic and structural checks on normalized data."""

    REQUIRED = ["variant_id", "assembly", "chromosome", "start", "allele_string"]

    @classmethod
    def validate(cls, record: dict[str, Any]) -> dict[str, Any]:
        missing = [field for field in cls.REQUIRED if not record.get(field)]
        warnings = []

        if record.get("assembly") and str(record["assembly"]).upper() != "GRCH38":
            warnings.append("Selected mapping is not GRCh38.")
        if not record.get("allele_string"):
            warnings.append("Allele string unavailable.")

        return {
            "valid": not missing,
            "missing_required_fields": missing,
            "warnings": warnings,
        }


class AnalysisAgent:
    """
    Prepare the scientific question for downstream AlphaGenome execution.

    This agent does not fabricate AlphaGenome scores. It creates the structured
    analysis request that a real AlphaGenome/Paper2Agent runtime must execute.
    """

    @staticmethod
    def build_alphagenome_request(variant: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": variant,
            "requested_modalities": "all available modalities",
            "rna_seq_analysis": {
                "rank_tissues_by_effect_magnitude": True,
                "priority_tissues": [
                    "cortex", "hippocampus", "striatum", "brain"
                ],
            },
            "gene_analysis": {
                "identify_largest_expression_changes": True,
                "report_direction": True,
            },
            "regulatory_analysis": {
                "check_dnase": True,
                "check_enhancer_marks": True,
                "priority_tissue": "brain",
            },
            "plots": ["regulatory landscape around variant"],
            "status": "requires real AlphaGenome execution",
        }


class ReportingAgent:
    """Persist structured workflow output as JSON and Markdown."""

    @staticmethod
    def save(
        variant: dict[str, Any],
        validation: dict[str, Any],
        request: dict[str, Any],
        output_dir: Path = ALPHAGENOME_OUTPUT_DIR,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "variant_workflow.json"
        md_path = output_dir / "agent_request.md"

        payload = {
            "generated_at_utc": utc_now(),
            "variant": variant,
            "validation": validation,
            "alphagenome_request": request,
            "execution_note": (
                "AlphaGenome prediction is intentionally not fabricated. "
                "Run this request in a valid AlphaGenome/Paper2Agent environment."
            ),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md = f"""# AlphaGenome Agent Request

## Variant
- ID: {variant.get('variant_id')}
- Build: {variant.get('assembly')}
- Chromosome: {variant.get('chromosome')}
- Position: {variant.get('start')}
- Alleles: {variant.get('allele_string')}

## Validation
- Valid: {validation.get('valid')}
- Missing fields: {validation.get('missing_required_fields')}
- Warnings: {validation.get('warnings')}

## Requested analysis
1. Run variant-effect prediction across all available AlphaGenome modalities.
2. For RNA-seq, rank tissues by predicted effect magnitude, prioritizing brain tissues.
3. Identify genes with the largest predicted expression changes and direction.
4. Check DNase/accessibility and enhancer-related regulatory signals in brain tissue.
5. Save regulatory-landscape plots around the variant.
6. Summarize the most plausible gene/mechanism.

> This file is a request specification, not an executed AlphaGenome result.
"""
        md_path.write_text(md, encoding="utf-8")
        return json_path, md_path


def run_variant_workflow(variant_id: str = "rs11174281") -> dict[str, Any]:
    """
    Five-component workflow:
      1. RetrievalAgent
      2. ProcessingAgent
      3. ValidationAgent
      4. AnalysisAgent
      5. ReportingAgent
    """
    retrieval = RetrievalAgent()
    processing = ProcessingAgent()
    validation_agent = ValidationAgent()
    analysis = AnalysisAgent()
    reporting = ReportingAgent()

    raw = retrieval.lookup_variant(variant_id)
    variant = processing.normalize_variant(raw, variant_id)
    validation = validation_agent.validate(variant)
    request = analysis.build_alphagenome_request(variant)
    json_path, md_path = reporting.save(variant, validation, request)

    print(json.dumps(variant, indent=2))
    print(f"Saved -> {json_path}")
    print(f"Saved -> {md_path}")

    return {
        "variant": variant,
        "validation": validation,
        "request": request,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


# =============================================================================
# PART 4 — PAPER2AGENT / ALPHAGENOME REPRODUCIBLE RUN PLAN
# =============================================================================

ALPHAGENOME_PROMPT = """I'm doing Mendelian randomization with rs11174281 as an
instrument for lifetime cannabis use (LCU), identified in a within-family GWAS.
Please use AlphaGenome to:
1. Look up rs11174281's chromosome, position, and alleles (build GRCh38 if
   possible — confirm which build you're using)
2. Run variant effect prediction for both alleles across all available modalities
3. For RNA-seq specifically, rank tissues by predicted effect magnitude — I'm
   especially interested in brain tissues (cortex, hippocampus, striatum, etc.)
4. Identify which gene(s) within the prediction window show the largest
   expression changes, and report direction (up or down) for the LCU-risk allele
5. Check whether the variant overlaps a predicted regulatory element
   (DNase peak, enhancer mark) in brain tissues
6. Save plots of the regulatory landscape around the variant
7. Summarize AlphaGenome's best guess about the gene and mechanism through
   which this variant exerts its effect.
"""


def alphagenome_run_plan() -> str:
    plan = r"""
Paper2Agent / AlphaGenome execution plan
----------------------------------------

Option A — local agent generation

git clone https://github.com/jmiao24/Paper2Agent.git
cd Paper2Agent
pip install fastmcp
npm install -g @anthropic-ai/claude-code

export ALPHAGENOME_API_KEY="YOUR_ALPHAGENOME_API_KEY"

bash Paper2Agent.sh \
  --project_dir AlphaGenome_Agent \
  --github_url https://github.com/google-deepmind/alphagenome \
  --api "$ALPHAGENOME_API_KEY"

cd AlphaGenome_Agent
claude
claude mcp list

Then submit ALPHAGENOME_PROMPT from this Python file.

Option B — hosted MCP

git clone https://github.com/jmiao24/Paper2Agent.git
cd Paper2Agent
npm install -g @anthropic-ai/claude-code

bash launch_remote_mcp.sh \
  --working_dir analysis_dir \
  --mcp_name alphagenome \
  --mcp_url https://Paper2Agent-alphagenome-mcp.hf.space

cd analysis_dir
claude

IMPORTANT:
The original execution environment did not have the required network/API/MCP
runtime, so no AlphaGenome biological result is claimed here.
"""
    print(plan)
    print("\nPrompt to submit:\n")
    print(ALPHAGENOME_PROMPT)
    return plan


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Agents & Automated Data Extraction — consolidated project"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scrape", help="Run Playwright/BeautifulSoup product scraper")

    llm = sub.add_parser("llm", help="Collect repeated LLM JSON responses")
    llm.add_argument("--runs", type=int, default=10)
    llm.add_argument("--model", default="gpt-5-mini")
    llm.add_argument("--delay", type=float, default=0.0)

    sub.add_parser("stats", help="Build per-JSON LLM response statistics")

    variant = sub.add_parser(
        "variant",
        help="Run 5-component Ensembl/AlphaGenome-prep workflow",
    )
    variant.add_argument("--variant", default="rs11174281")

    sub.add_parser(
        "alphagenome-plan",
        help="Print the reproducible Paper2Agent/AlphaGenome execution plan",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "scrape":
        asyncio.run(run_scraper())

    elif args.command == "llm":
        collect_llm_runs(
            SHOPPING_PROMPTS,
            runs=args.runs,
            model=args.model,
            delay_seconds=args.delay,
        )
        build_llm_stats()

    elif args.command == "stats":
        build_llm_stats()

    elif args.command == "variant":
        run_variant_workflow(args.variant)

    elif args.command == "alphagenome-plan":
        alphagenome_run_plan()


if __name__ == "__main__":
    main()
