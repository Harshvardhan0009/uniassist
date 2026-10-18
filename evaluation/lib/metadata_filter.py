"""Metadata Filter and Domain Routing module (Phase 16).

Explores domain metadata filtering for narrowing vector retrieval spaces:
1. Maps corpus documents to domain topics:
   - library, dress_code, residential, academic_benefits, placement_career, ojt, semester_abroad, nss, campus_map
2. Implements an automated domain router (intent classification).
3. Evaluates both:
   a) Oracle Metadata Filtering (theoretical upper bound where topic is perfectly known).
   b) Automated Metadata Filtering (realistic pipeline with intent routing).
4. Measures retrieval precision, recall, and the 'misclassification catastrophic drop' risk.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import paths

paths.ensure_backend_on_path()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("eval.metadata_filter")

# Official corpus file to primary domain folder/topic mapping
FILE_TO_DOMAINS = {
    "DressCode/Dress Code and Uniform Policy for Students.pdf": ["dress_code"],
    "DressCode/FAQ's.pdf": ["dress_code"],
    "EDU REV/Academic Benefits.pdf": ["academic_benefits"],
    "LibraryPolicy/Library Policy.pdf": ["library"],
    "NSSPolicy/LPUNSSPOLICY.pdf": ["nss"],
    "PlacementPloicy/2023 Career Services Policy.pdf": ["placement_career", "ojt"],
    "PlacementPloicy/Academic Benefit Plan - 3rd Party Industry Tests - 2023 Batch.docx": ["placement_career", "academic_benefits"],
    "PlacementPloicy/Academic Benefit Plan - Clearing Competitive Exams.pdf": ["placement_career", "academic_benefits"],
    "PlacementPloicy/Academic Benefit Plan - Placement and Internship - 2023 and 2024 Batch.pdf": ["placement_career", "academic_benefits"],
    "PlacementPloicy/OJT Internship policy 2019-2020 onwards (10 Aug 2019).pdf": ["ojt", "placement_career"],
    "ResidentialFacilities/Charges For Residential Facilities.pdf": ["residential"],
    "ResidentialFacilities/Residential Facilities Refund Guidelines.pdf": ["residential"],
    "SemesterExchange/Semester_Year_Abroad_Policy.pdf": ["semester_abroad"],
    "UNImap/UNIMAP_Campus_Guide.txt": ["campus_map"],
}

DOMAIN_DESCRIPTIONS = {
    "library": "Library timings, book issue limits, fines, catalog, study rooms, Central Library",
    "dress_code": "Dress code rules, formal uniform guidelines, lab coats, winter dress code",
    "residential": "Hostel accommodation, hostel fees, room types, mess rebate, curfew timings, refund guidelines",
    "academic_benefits": "Grade upgradation, academic honors, scholarships, Dean's list, credits for patents or MOOCs",
    "placement_career": "Placement drives, eligibility criteria, dream packages, company registration, debarment",
    "ojt": "On-the-job training, internship policies, duration, evaluation criteria, industry training",
    "semester_abroad": "Semester exchange, study abroad, foreign partner universities, eligibility, duration",
    "nss": "National Service Scheme (NSS), community service, social activities, volunteering hours",
    "campus_map": "Campus layout, building numbers, block locations, navigation, school locations",
    "general_other": "Out-of-scope questions, general university inquiries, or queries without a specific domain",
}

ROUTER_PROMPT = (
    "You are an intent classification router for a university query assistant. "
    "Given a student query, identify the single most relevant policy domain from the following list:\n\n"
    + "\n".join(f"- {d}: {desc}" for d, desc in DOMAIN_DESCRIPTIONS.items())
    + "\n\nOutput ONLY the domain name (e.g., 'library', 'placement_career', 'residential', etc.). Do not include explanation."
)


def classify_domain(
    query: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> Tuple[str, float]:
    """Classify the user query into a domain topic using an LLM router."""
    t0 = time.perf_counter()
    key = api_key or getattr(settings, "FALLBACK_LLM_API_KEY", "") or settings.LLM_API_KEY
    url = base_url or getattr(settings, "FALLBACK_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    mdl = model or getattr(settings, "FALLBACK_LLM_MODEL", "openai/gpt-oss-120b")

    if not key:
        return "general_other", 0.0

    try:
        llm = ChatOpenAI(
            api_key=key,
            base_url=url,
            model=mdl,
            temperature=0.0,
            max_tokens=256,
        )
        resp = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"Query: {query}\nDomain:"),
        ])
        pred = resp.content.strip().lower()
        pred = re.sub(r"<think>[\s\S]*?</think>", "", pred).strip()
        pred = re.sub(r"<think>[\s\S]*$", "", pred).strip()
        # Clean up pred
        for valid in DOMAIN_DESCRIPTIONS.keys():
            if valid in pred:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return valid, latency_ms
    except Exception as e:
        logger.warning(f"Domain router call failed: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return "general_other", latency_ms


def matches_domain(doc_source: str, domain: str) -> bool:
    """Check if a document source file matches the given domain topic."""
    if not domain or domain == "general_other":
        return True
    allowed_domains = FILE_TO_DOMAINS.get(doc_source, [])
    return domain in allowed_domains
