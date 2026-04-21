import re
from typing import NamedTuple

KEYWORD_GROUPS = {
    # Exclusions — highest weight, most directly actionable
    "exclusions": (4.0, [
        "not covered", "not payable", "not admissible", "not entitled",
        "excluded", "exclusion", "exclusions", "shall not be liable",
        "shall not cover", "will not cover", "does not cover",
        "no benefit", "no claim", "no coverage", "outside the scope",
        "beyond the scope", "expressly excluded", "specifically excluded",
    ]),
    # Waiting periods — high weight, directly causes claim rejection
    "waiting_periods": (3.5, [
        "waiting period", "waiting periods", "initial waiting",
        "specific illness", "pre-existing disease waiting",
        "ped waiting", "30-day waiting", "30 day waiting",
        "first 30 days", "first year", "first two years",
        "first 2 years", "first 4 years", "moratorium",
        "cooling period", "qualifying period",
    ]),
    # Co-payment and deductibles — moderate weight
    "copay": (3.0, [
        "co-payment", "co payment", "copayment", "copay", "co-pay",
        "deductible", "you shall bear", "insured shall bear",
        "policyholder shall pay", "out of pocket", "your share",
        "proportionate deduction",
    ]),
    # Sub-limits and caps — moderate weight
    "sublimits": (3.0, [
        "sub-limit", "sub limit", "sublimit", "capped at", "cap of",
        "maximum payable", "maximum benefit", "maximum liability",
        "not exceed", "shall not exceed", "up to a maximum",
        "room rent", "room charge", "icu charges", "icu limit",
        "day care", "ambulance charge", "organ donor",
        "ayurvedic", "homeopathic", "dental limit",
        "maternity limit", "newborn limit",
    ]),
    # General risk signals — lower weight but still worth including
    "risk_signals": (1.5, [
        "shall not", "will not", "is not", "are not", "cannot",
        "liable", "liability", "obligation", "obligation",
        "clause", "condition", "provision", "exception",
        "herein", "notwithstanding", "irrespective",
        "subject to", "provided that", "provided however",
        "in no event", "under no circumstances",
    ]),
}

IMPORTANT_SECTION_HEADERS = [
    r"exclusion", r"not covered", r"what (is|are) not",
    r"waiting period", r"waiting clause",
    r"co.?pay", r"co.?payment", r"deductible",
    r"sub.?limit", r"coverage limit", r"benefit limit",
    r"general condition", r"special condition", r"important condition",
    r"terms and condition", r"definitions", r"interpretation",
    r"claim procedure", r"claim process",
]

# How many neighbouring paragraphs to include around a hit paragraph
CONTEXT_WINDOW = 1   # paragraphs before and after each hit

# How many paragraphs to include after a section header match
HEADER_FOLLOWTHROUGH = 8


class FilterStats(NamedTuple):
    total_paragraphs: int
    selected_paragraphs: int
    total_chars: int
    filtered_chars: int
    reduction_pct: float


def _split_paragraphs(text: str) -> list[str]:
    """
    Split text into meaningful paragraphs.
    Treats double-newlines, bullet points, and numbered list items as boundaries.
    Strips very short fragments (likely page headers/footers).
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on double newlines OR numbered list patterns like "1." "2." "a)" "(i)"
    parts = re.split(r"\n{2,}|(?=\n\s*(?:\d+\.|[a-z]\)|•|–|—|\*)\s)", text)

    paras = []
    for p in parts:
        p = p.strip()
        # Skip very short fragments (page numbers, headers like "Page 3 of 40")
        if len(p) < 30:
            continue
        # Skip fragments that are purely numeric or purely whitespace
        if re.fullmatch(r"[\d\s\-\.]+", p):
            continue
        paras.append(p)

    return paras


def _score_paragraph(para: str) -> float:
    """
    Return a relevance score for a paragraph.
    Higher = more likely to contain policy risk information.
    """
    lower = para.lower()
    score = 0.0

    for group_name, (weight, keywords) in KEYWORD_GROUPS.items():
        for kw in keywords:
            if kw in lower:
                score += weight
                # Bonus for exact phrase matches (not just substrings)
                if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                    score += weight * 0.3

    return score


def _is_section_header(para: str) -> bool:
    """
    Returns True if this paragraph looks like a section heading that introduces
    an important section (exclusions, waiting periods, etc.).
    """
    lower = para.lower().strip()

    # Very long paragraphs are not headers
    if len(para) > 300:
        return False

    # Check for important header keywords
    for pattern in IMPORTANT_SECTION_HEADERS:
        if re.search(pattern, lower):
            return True

    # Also flag ALL-CAPS short lines (common in policy docs for section titles)
    if para.isupper() and len(para) < 100:
        return True

    # Numbered section headings like "Section 4: Exclusions"
    if re.match(r"^(section|clause|article|schedule|part)\s+[\dIVXivx]+", lower):
        return True

    return False


def extract_relevant_text(text: str, min_score: float = 2.0) -> tuple[str, FilterStats]:
    """
    Main entry point. Returns:
        filtered_text   — concatenated relevant paragraphs, ready for LLM
        stats           — FilterStats namedtuple for reporting

    Parameters:
        text        — full extracted PDF text
        min_score   — minimum relevance score for a paragraph to be included
                      directly (neighbours and header followthroughs are always included)
    """
    paragraphs = _split_paragraphs(text)
    n = len(paragraphs)

    if n == 0:
        return text, FilterStats(0, 0, len(text), len(text), 0.0)

    # Score every paragraph
    scores = [_score_paragraph(p) for p in paragraphs]

    # Determine which paragraphs to include
    include = [False] * n

    for i, (para, score) in enumerate(zip(paragraphs, scores)):

        # Direct hit — paragraph has enough signal
        if score >= min_score:
            include[i] = True
            # Include context neighbours
            for j in range(max(0, i - CONTEXT_WINDOW), min(n, i + CONTEXT_WINDOW + 1)):
                include[j] = True

        # Section header — include the header + next N paragraphs
        if _is_section_header(para):
            include[i] = True
            for j in range(i + 1, min(n, i + HEADER_FOLLOWTHROUGH + 1)):
                include[j] = True

    # Safety net: if we captured less than 15% of paragraphs, something is
    # wrong (unusual formatting, scanned text, etc.) — fall back to full text
    selected_count = sum(include)
    if selected_count < max(3, int(n * 0.10)):
        # Fall back: lower threshold and retry
        for i, score in enumerate(scores):
            if score >= 0.5:
                include[i] = True
                for j in range(max(0, i - 1), min(n, i + 2)):
                    include[j] = True

    selected_count = sum(include)

    filtered_parts = []
    prev_included = False

    for i, (para, inc) in enumerate(zip(paragraphs, include)):
        if inc:
            if filtered_parts and not prev_included:
                filtered_parts.append("---")  
            filtered_parts.append(para)
            prev_included = True
        else:
            prev_included = False

    filtered_text = "\n\n".join(filtered_parts)

    stats = FilterStats(
        total_paragraphs=n,
        selected_paragraphs=selected_count,
        total_chars=len(text),
        filtered_chars=len(filtered_text),
        reduction_pct=round((1 - len(filtered_text) / max(len(text), 1)) * 100, 1),
    )

    return filtered_text, stats

def chunk_text(text, chunk_size=3000, overlap=200):
    """Legacy chunker — kept for compatibility. Use extract_relevant_text() instead."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            bp = text.rfind('\n', start + chunk_size - 400, end)
            if bp == -1:
                bp = text.rfind('. ', start + chunk_size - 400, end)
            if bp != -1:
                end = bp + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks