import re
import time
import streamlit as st
from pdf_utils import extract_text
from text_utils import chunk_text
from llm import insurance_decoder
from risk_engine import calculate_risk_score, AVG_TREATMENT_COST

# Page config
st.set_page_config(
    page_title="LegalX - Legal Summarizer",
    page_icon="",
    layout="wide"
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main .block-container {
    background-color: #0a0a0f !important;
}

[data-testid="stSidebar"] { background: #0e0e16 !important; }

html, body, [class*="css"], .stMarkdown, p, li, span, div, label {
    font-family: 'Inter', sans-serif !important;
    color: #e8e8f0 !important;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: #f5f0e8 !important;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

.stSlider label, .stNumberInput label, .stSelectbox label,
.stMultiSelect label, .stFileUploader label {
    color: #a0a0b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

.stTextInput input, .stNumberInput input,
.stSelectbox [data-baseweb="select"] div,
[data-baseweb="input"] input {
    background: #16161f !important;
    border: 1px solid #2a2a3d !important;
    color: #e8e8f0 !important;
    border-radius: 6px !important;
}

.stSelectbox [data-baseweb="select"] div { color: #e8e8f0 !important; }

.stSlider [data-testid="stSlider"] div[role="slider"] {
    background: #7c6af7 !important;
}

.stFormSubmitButton button, .stButton button {
    background: linear-gradient(135deg, #7c6af7, #5b8def) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stFormSubmitButton button:hover, .stButton button:hover {
    opacity: 0.88 !important;
}

[data-testid="stExpander"] {
    background: #13131e !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #c0b8ff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
[data-testid="stExpander"] summary:hover { color: #e0d8ff !important; }
[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    color: #c8c8d8 !important;
    background: #13131e !important;
}
[data-testid="stExpander"] p, [data-testid="stExpander"] li,
[data-testid="stExpander"] strong {
    color: #c8c8d8 !important;
}
[data-testid="stExpander"] strong { color: #e8e8f0 !important; }
[data-testid="stExpander"] hr { border-color: #2a2a3d !important; }

.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c6af7, #5b8def) !important;
}
.stProgress > div > div { background: #1e1e2e !important; }

.stAlert { border-radius: 8px !important; }

.stCaption { color: #606078 !important; font-size: 0.8rem !important; }

#App header
.app-header {
    position: relative;
    overflow: hidden;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    border-radius: 16px;
    background: #0d0d18;
    border: 1px solid #1e1e30;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 380px; height: 380px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,106,247,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(91,141,239,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.app-header-inner {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 2rem;
    position: relative;
    z-index: 1;
}
.app-header-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 3.2rem;
    font-weight: 900;
    color: #f5f0e8 !important;
    line-height: 1.05;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.01em;
}
.app-header-title em {
    font-style: italic;
    color: #c0b8ff !important;
}
.app-header-sub {
    font-size: 1rem;
    color: #8888a8 !important;
    margin: 0;
    font-weight: 400;
    max-width: 500px;
    line-height: 1.6;
}
.header-stats-grid {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    text-align: right;
}
.stat-pill-dark {
    background: rgba(124,106,247,0.12);
    border: 1px solid rgba(124,106,247,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #a8a0e8 !important;
    text-transform: uppercase;
    white-space: nowrap;
}
.stat-pill-dark strong {
    color: #c8c0ff !important;
    font-weight: 700;
}

Upload zone
.upload-zone {
    border: 1.5px dashed #2a2a40;
    border-radius: 12px;
    padding: 3.5rem 2rem;
    text-align: center;
    background: #0e0e18;
    margin-top: 1rem;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #7c6af7; }
.upload-zone-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    display: block;
    opacity: 0.6;
}
.upload-zone-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    color: #e8e8f0 !important;
    margin-bottom: 0.4rem;
}
.upload-zone-sub { font-size: 0.88rem; color: #5a5a78 !important; }

/* ── Section heading ── */
.section-heading {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0ecff !important;
    margin: 2.8rem 0 0.3rem 0;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid #1e1e30;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.86rem;
    color: #686888 !important;
    margin: 0.2rem 0 1.2rem 0;
    line-height: 1.5;
}

/* ── Stat row (summary boxes) ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-box {
    background: #0e0e18;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.6rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-box:hover { border-color: #3a3a5a; transform: translateY(-2px); }
.stat-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 12px 12px 0 0;
}
.stat-box-red::before   { background: #e05568; }
.stat-box-orange::before { background: #e8903a; }
.stat-box-yellow::before { background: #d4a820; }
.stat-box-blue::before  { background: #4a90d9; }

.stat-box-number {
    font-family: 'Inter', sans-serif !important;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.stat-box-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a5a78 !important;
}

/* ── Risk meter ── */
.meter-wrap {
    position: relative;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #27ae60, #f39c12, #e05568);
    margin: 0.8rem 0 0.3rem;
}
.meter-needle {
    position: absolute;
    top: -7px;
    width: 2px;
    height: 22px;
    background: #ffffff;
    border-radius: 1px;
    transform: translateX(-50%);
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
}
.meter-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #4a4a68 !important;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Banner ── */
.banner {
    padding: 1rem 1.4rem;
    border-radius: 8px;
    font-size: 0.93rem;
    font-weight: 500;
    margin: 1rem 0;
    border: 1px solid;
    line-height: 1.5;
}
.banner strong { font-weight: 700; }
.banner-red    { background: rgba(224,85,104,0.12); border-color: rgba(224,85,104,0.4); color: #f0a0a8 !important; }
.banner-red strong { color: #f5b8be !important; }
.banner-orange { background: rgba(232,144,58,0.12); border-color: rgba(232,144,58,0.4); color: #f0b878 !important; }
.banner-orange strong { color: #f5c898 !important; }
.banner-green  { background: rgba(39,174,96,0.12);  border-color: rgba(39,174,96,0.4);  color: #80c898 !important; }
.banner-green strong { color: #a0d8b0 !important; }

/* ── Cards ── */
.card {
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
    border-left: 3px solid;
    position: relative;
    transition: transform 0.15s;
}
.card:hover { transform: translateX(3px); }
.card-red    { background: rgba(224,85,104,0.08);  border-color: #e05568; }
.card-orange { background: rgba(232,144,58,0.08);  border-color: #e8903a; }
.card-yellow { background: rgba(212,168,32,0.08);  border-color: #d4a820; }
.card-blue   { background: rgba(74,144,217,0.08);  border-color: #4a90d9; }
.card-green  { background: rgba(39,174,96,0.08);   border-color: #27ae60; }

.card-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.label-red    { color: #e05568 !important; }
.label-orange { color: #e8903a !important; }
.label-yellow { color: #d4a820 !important; }
.label-blue   { color: #4a90d9 !important; }
.label-green  { color: #27ae60 !important; }

.card-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.02rem;
    font-weight: 700;
    color: #f0ecff !important;
    margin-bottom: 0.3rem;
    line-height: 1.35;
}
.card-text {
    font-size: 0.88rem;
    color: #8888a8 !important;
    line-height: 1.65;
}

/* ── Simulator section ── */
.sim-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #100e1e 0%, #13101f 50%, #0e1220 100%);
    border: 1px solid #2a2240;
    border-radius: 16px;
    padding: 2.5rem 2.5rem;
    margin: 3rem 0 1.8rem 0;
}
.sim-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,106,247,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.sim-header-tag {
    display: inline-block;
    background: rgba(124,106,247,0.15);
    border: 1px solid rgba(124,106,247,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c0b8ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}
.sim-header h2 {
    font-family: 'Inter', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f5f0e8 !important;
    margin: 0 0 0.6rem 0 !important;
    line-height: 1.2;
    position: relative;
    z-index: 1;
}
.sim-header p {
    color: #686888 !important;
    font-size: 0.93rem !important;
    margin: 0 !important;
    line-height: 1.6;
    max-width: 580px;
    position: relative;
    z-index: 1;
}

/* ── Result boxes (Phase 2) ── */
.result-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.result-box {
    background: #0e0e18;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-box::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
}
.result-number {
    font-family: 'Inter', sans-serif  !important;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a5a78 !important;
}

/* ── Footer ── */
.footer-text {
    font-size: 0.76rem;
    color: #3a3a58 !important;
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid #1a1a28;
    line-height: 1.7;
    margin-top: 3rem;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { display: none; }

""", unsafe_allow_html=True)


def safe_dict(item):
    return item if isinstance(item, dict) else {"value": str(item)}

def fmt_inr(amount: float) -> str:
    if amount >= 1_00_00_000: return f"Rs. {amount/1_00_00_000:.1f} Cr"
    if amount >= 1_00_000:    return f"Rs. {amount/1_00_000:.1f} L"
    if amount >= 1_000:       return f"Rs. {amount/1_000:.0f}K"
    return f"Rs. {int(amount)}"

def risk_color(score: int) -> str:
    if score >= 70: return "#c0392b"
    if score >= 45: return "#d4860a"
    return "#27ae60"

def meter(score: int):
    st.markdown(f"""
    <div class="meter-wrap">
        <div class="meter-needle" style="left:{score}%"></div>
    </div>
    <div class="meter-labels"><span>Safe</span><span>Moderate</span><span>High Risk</span></div>
    """, unsafe_allow_html=True)

def card(label: str, color: str, title: str, body: str):
    st.markdown(f"""
    <div class="card card-{color}">
        <div class="card-label label-{color}">{label}</div>
        <div class="card-title">{title}</div>
        {"<div class='card-text'>" + body + "</div>" if body else ""}
    </div>
    """, unsafe_allow_html=True)

def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def deduplicate(items):
    seen, out = set(), []
    for item in items:
        k = str(item)
        if k not in seen:
            seen.add(k); out.append(item)
    return out


def parse_copay_pct(copayments: list) -> float:
    """Extract numeric co-payment % from LLM output. Returns 0.0 if not found."""
    for cp in copayments:
        cp = safe_dict(cp)
        raw = cp.get("percentage") or cp.get("value", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(raw))
        if match:
            return float(match.group(1)) / 100
    return 0.0

def parse_room_rent(hidden_limits: list) -> float:
    """Look for room rent cap in hidden limits. Returns 0 if not found."""
    for hl in hidden_limits:
        hl = safe_dict(hl)
        text = str(hl.get("limit", "") or hl.get("value", "")).lower()
        if "room" in text and "rent" in text:
            match = re.search(r"(?:rs\.?|inr|rupees?)?\s*(\d[\d,]*)", text)
            if match:
                return float(match.group(1).replace(",", ""))
    return 0.0

def parse_deductible(hidden_limits: list) -> float:
    """Look for deductible in hidden limits. Returns 0 if not found."""
    for hl in hidden_limits:
        hl = safe_dict(hl)
        text = str(hl.get("limit", "") or hl.get("value", "")).lower()
        if "deductible" in text or "excess" in text:
            match = re.search(r"(?:rs\.?|inr|rupees?)?\s*(\d[\d,]*)", text)
            if match:
                return float(match.group(1).replace(",", ""))
    return 0.0


st.markdown("""
<div class="app-header">
    <div class="app-header-inner">
        <div>
            <div class="app-header-title">LegalX -
            Legal<br><em>Summarizer</em></div>
            <p class="app-header-sub">
                Upload your health insurance policy. We translate 60 pages of legal language
                into plain English and show you exactly what to watch out for.
            </p>
        </div>
        <div class="header-stats-grid">
            <div class="stat-pill-dark"><strong>68%</strong> of Indians never read their policy</div>
            <div class="stat-pill-dark"><strong>30–40%</strong> of claims get rejected</div>
            <div class="stat-pill-dark"><strong>Only 23%</strong> know what is covered</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Insurance Policy (PDF)",
    type=["pdf"],
    label_visibility="collapsed"
)

if not uploaded_file:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone-title">Drop your policy PDF here</div>
        <div class="upload-zone-sub">Text-based PDFs only. We do not store your document.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Extract text
with st.spinner("Reading policy document..."):
    try:
        text = extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the PDF: {e}")
        st.stop()

if not text or len(text.strip()) < 200:
    st.error("No readable text found. This may be a scanned PDF. Please use a text-based PDF.")
    st.stop()

chunks = chunk_text(text, chunk_size=3000, overlap=200)

# LLM analysis
all_alerts, waiting_periods, exclusions = [], [], []
copayments, hidden_limits, llm_risk_scores = [], [], []

st.markdown("""
<div style="font-size:0.78rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;
            color:#6868a8;margin-bottom:0.5rem;font-family:'JetBrains Mono',monospace">
    Step 1 of 2 — Reading Policy Document
</div>
""", unsafe_allow_html=True)
progress_bar = st.progress(0)
status_slot  = st.empty()

PHASE1_STEPS = [
    "Extracting policy structure...",
    "Scanning for exclusions...",
    "Identifying waiting periods...",
    "Checking co-payment clauses...",
    "Detecting hidden sub-limits...",
    "Flagging critical risk alerts...",
    "Scoring overall policy risk...",
    "Finalising analysis...",
]

for i, chunk in enumerate(chunks):
    step_label = PHASE1_STEPS[min(i, len(PHASE1_STEPS) - 1)]
    status_slot.markdown(
        f'<div style="font-size:0.82rem;color:#5858888;font-family:\'JetBrains Mono\',monospace">'
        f'{step_label}</div>',
        unsafe_allow_html=True
    )
    result = insurance_decoder(chunk)
    if result:
        s = result.get("risk_score", 0)
        if isinstance(s, (int, float)) and 0 <= s <= 100:
            llm_risk_scores.append(s)
        all_alerts.extend(result.get("danger_alerts", []))
        waiting_periods.extend(result.get("waiting_periods", []))
        exclusions.extend(result.get("exclusions", []))
        copayments.extend(result.get("co_payment", []))
        hidden_limits.extend(result.get("hidden_limits", []))
    progress_bar.progress((i + 1) / len(chunks))

status_slot.markdown(
    '<div style="font-size:0.82rem;color:#27ae60;font-family:\'JetBrains Mono\',monospace">'
    'Analysis complete.</div>',
    unsafe_allow_html=True
)
import time; time.sleep(0.6)
progress_bar.empty()
status_slot.empty()

waiting_periods = deduplicate(waiting_periods)
exclusions      = deduplicate(exclusions)
copayments      = deduplicate(copayments)
hidden_limits   = deduplicate(hidden_limits)
all_alerts      = deduplicate(all_alerts)

if not llm_risk_scores and not exclusions and not waiting_periods and not all_alerts:
    st.error("Could not extract data from the policy. Please check that the PDF has readable text.")
    st.stop()

llm_avg_risk = int(sum(llm_risk_scores) / len(llm_risk_scores)) if llm_risk_scores else 50

# Auto-extract policy parameters from LLM output
extracted_copay       = parse_copay_pct(copayments)
extracted_room_rent   = parse_room_rent(hidden_limits)
extracted_deductible  = parse_deductible(hidden_limits)

section("Policy Summary")

st.markdown(f"""
<div class="stat-row">
    <div class="stat-box stat-box-red">
        <div class="stat-box-number" style="color:#e05568">{len(exclusions)}</div>
        <div class="stat-box-label">Things Not Covered</div>
    </div>
    <div class="stat-box stat-box-orange">
        <div class="stat-box-number" style="color:#e8903a">{len(waiting_periods)}</div>
        <div class="stat-box-label">Waiting Period Traps</div>
    </div>
    <div class="stat-box stat-box-yellow">
        <div class="stat-box-number" style="color:#d4a820">{len(copayments)}</div>
        <div class="stat-box-label">Times You Pay Extra</div>
    </div>
    <div class="stat-box stat-box-blue">
        <div class="stat-box-number" style="color:#4a90d9">{len(hidden_limits)}</div>
        <div class="stat-box-label">Hidden Coverage Limits</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="font-family:'Playfair Display',serif;font-size:1rem;color:#8888a8;margin-bottom:0.4rem;font-weight:400">
    Overall Policy Risk Score
    &nbsp;&nbsp;<span style="font-size:2.2rem;font-weight:700;color:{risk_color(llm_avg_risk)}">{llm_avg_risk} <span style="font-size:1rem;color:#4a4a68">/&nbsp;100</span></span>
</div>
""", unsafe_allow_html=True)
meter(llm_avg_risk)

if llm_avg_risk >= 70:
    st.markdown('<div class="banner banner-red"><strong>High Risk Policy.</strong> This policy has many traps. Read every section below before you sign or renew.</div>', unsafe_allow_html=True)
elif llm_avg_risk >= 45:
    st.markdown('<div class="banner banner-orange"><strong>Moderate Risk Policy.</strong> Important limitations exist. Know them before a medical emergency hits.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="banner banner-green"><strong>Relatively Safe Policy.</strong> Fewer hidden traps than average. Still review the sections below.</div>', unsafe_allow_html=True)


all_alerts = sorted(
    all_alerts,
    key=lambda x: {"Critical": 3, "High": 2, "Medium": 1}.get(
        safe_dict(x).get("severity", "Medium"), 0),
    reverse=True
)

if all_alerts:
    section("Critical Alerts", "The most alarming flags found in your policy document.")
    for alert in all_alerts:
        a   = safe_dict(alert)
        msg = a.get("message") or a.get("value", "")
        sev = a.get("severity", "Medium")
        if not msg: continue
        if sev == "Critical":
            card("Critical Alert", "red", msg, "")
        elif sev == "High":
            card("Important", "orange", msg, "")
        else:
            card("Note", "blue", msg, "")

if waiting_periods:
    section(
        "Waiting Period Traps",
        "During these periods your claim will be rejected; even though you are paying the premium every month."
    )
    for wp in waiting_periods:
        wp        = safe_dict(wp)
        condition = wp.get("condition") or wp.get("value", "Some conditions")
        duration  = wp.get("duration", "")
        impact    = wp.get("impact", "")
        title     = f"No coverage for {condition}" + (f" — {duration}" if duration else "")
        body      = impact or f"Any claim for {condition} during this waiting period will be rejected. You pay the full hospital bill yourself."
        card("Waiting Period", "orange", title, body)

if exclusions:
    section(
        "Treatments Not Covered",
        "If you are treated for any of these, you pay 100% from your own pocket. Insurance will not help."
    )
    cols = st.columns(2)
    for i, ex in enumerate(exclusions):
        ex     = safe_dict(ex)
        item   = ex.get("item") or ex.get("value", "Treatment")
        impact = ex.get("impact", "You will have to pay the full cost yourself.")
        with cols[i % 2]:
            card("Not Covered", "red", item, impact)

if copayments:
    section(
        "Times You Pay Extra",
        "Even after insurance pays, you still pay a share of the bill. This is called a co-payment."
    )
    for cp in copayments:
        cp      = safe_dict(cp)
        pct     = cp.get("percentage") or cp.get("value", "A percentage")
        cond    = cp.get("condition", "")
        impact  = cp.get("impact", "")
        title   = f"You pay {pct} of the bill yourself"
        body    = (f"When: {cond}. " if cond else "") + (
            impact or f"Example: On a Rs. 1 lakh bill, you pay {pct} directly to the hospital."
        )
        card("Co-payment", "yellow", title, body)

if hidden_limits:
    section(
        "Hidden Coverage Limits",
        "Your policy has a headline amount; but pays much less for specific treatments."
    )
    for hl in hidden_limits:
        hl         = safe_dict(hl)
        limit      = hl.get("limit") or hl.get("value", "A limit exists")
        applies_to = hl.get("applies_to", "")
        impact     = hl.get("impact", "")
        body       = (f"Applies to: {applies_to}. " if applies_to else "") + (
            impact or "Even if your total cover is higher, this specific limit means you pay the difference."
        )
        card("Hidden Limit", "blue", limit, body)

st.markdown("""
<div class="sim-header">
    <div class="sim-header-tag">Step 2 — Personalise</div>
    <h2>Calculate Your Risk Score</h2>
    <p>
        The above reflects what is written in your policy document.
        Now tell us about your age, health conditions, and income.
        We calculate exactly how risky this policy is for your specific situation.
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("risk_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        age = st.slider(
            "Your Age",
            min_value=18, max_value=80, value=25,
            help="Affects the statistical likelihood of each disease in our risk model."
        )
        annual_income = st.number_input(
            "Your Annual Income",
            min_value=1_00_000, max_value=5_00_00_000,
            value=8_00_000, step=50_000, format="%d",
            help="Used to determine if a major illness could push you into financial crisis."
        )

    with col_b:
        sum_insured = st.selectbox(
            "Your Policy's Total Cover",
            options=[2_00_000, 3_00_000, 5_00_000, 10_00_000, 15_00_000, 25_00_000, 50_00_000],
            default=[],
            format_func=fmt_inr,
            help="The maximum amount your insurer will pay in a claim year."
        )
        declared_diseases = st.multiselect(
            "Define Pre-existing Conditions",
            options=list(AVG_TREATMENT_COST.keys()),
            default=[],
            help="We will cross-check these against the waiting periods found in your policy."
        )

    submitted = st.form_submit_button("Calculate My Personal Risk", use_container_width=True)


if submitted:

    # Phase 2 progress bar
    st.markdown("""
    <div style="font-size:0.78rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;
                color:#6868a8;margin-bottom:0.5rem;font-family:'JetBrains Mono',monospace">
        Step 2 of 2 — Running Personal Risk Simulation
    </div>
    """, unsafe_allow_html=True)
    p2_bar    = st.progress(0)
    p2_status = st.empty()

    PHASE2_STEPS = [
        ("Building your policy profile...",    0.15),
        ("Mapping waiting periods to conditions...", 0.35),
        ("Calculating expected out-of-pocket...", 0.55),
        ("Computing claim rejection probability...", 0.72),
        ("Scoring financial exposure...",       0.88),
        ("Generating your report...",           1.0),
    ]

    def p2_step(label, pct):
        p2_status.markdown(
            f'<div style="font-size:0.82rem;color:#585888;font-family:\'JetBrains Mono\',monospace">'
            f'{label}</div>',
            unsafe_allow_html=True
        )
        p2_bar.progress(pct)
        time.sleep(0.3)

    p2_step(*PHASE2_STEPS[0])

    waiting_names = [
        safe_dict(wp).get("condition") or safe_dict(wp).get("value", "")
        for wp in waiting_periods
        if safe_dict(wp).get("condition") or safe_dict(wp).get("value")
    ]
    p2_step(*PHASE2_STEPS[1])

    policy = {
        "copay":           extracted_copay,
        "deductible":      extracted_deductible,
        "room_rent_daily": extracted_room_rent if extracted_room_rent > 0 else None,
        "waiting_periods": {d: 2 for d in waiting_names},
        "sub_limits": {},
    }
    p2_step(*PHASE2_STEPS[2])

    risk = calculate_risk_score(
        policy=policy,
        age=age,
        declared_diseases=declared_diseases,
        annual_income=annual_income,
        sum_insured=sum_insured,
        llm_exclusions=exclusions,
        llm_waiting_periods=waiting_periods,
        llm_hidden_limits=hidden_limits,
        llm_copayments=copayments,
        llm_risk_score=llm_avg_risk,
    )
    p2_step(*PHASE2_STEPS[3])
    p2_step(*PHASE2_STEPS[4])
    p2_step(*PHASE2_STEPS[5])

    p2_status.markdown(
        '<div style="font-size:0.82rem;color:#27ae60;font-family:\'JetBrains Mono\',monospace">'
        'Simulation complete.</div>',
        unsafe_allow_html=True
    )
    time.sleep(0.5)
    p2_bar.empty()
    p2_status.empty()

    final_score = risk["final_score"]
    oop         = risk["expected_oop_5yr"]
    rejection   = risk["rejection_probability_pct"]

    section("Your Personal Risk Report")

    # Three headline numbers 
    st.markdown(f"""
    <div class="result-row">
        <div class="result-box">
            <div class="result-number" style="color:{risk_color(final_score)}">{final_score}<span style="font-size:1.1rem;color:#888">/100</span></div>
            <div class="result-label">Your Risk Score</div>
        </div>
        <div class="result-box">
            <div class="result-number" style="color:#c0392b">{fmt_inr(oop)}</div>
            <div class="result-label">Estimated Out-of-Pocket (5 Years)</div>
        </div>
        <div class="result-box">
            <div class="result-number" style="color:#d4860a">{rejection}%</div>
            <div class="result-label">Chance a Claim Gets Rejected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    meter(final_score)
    st.markdown("<br>", unsafe_allow_html=True)

    # Verdict
    tier   = risk["risk_tier"]
    detail = risk["risk_tier_detail"]
    if final_score >= 70:
        card("Your Verdict", "red", tier, detail)
    elif final_score >= 45:
        card("Your Verdict", "orange", tier, detail)
    else:
        card("Your Verdict", "green", tier, detail)

    # Catastrophic warning
    if risk["catastrophic_expenditure_warning"]:
        card(
            "Financial Crisis Warning",
            "red",
            "A serious illness could wipe out a large part of your savings",
            f"Based on your income of {fmt_inr(annual_income)}/year, your expected out-of-pocket costs "
            f"({fmt_inr(oop)} over 5 years) would exceed 40% of your take-home savings. "
            f"WHO defines this as catastrophic health expenditure. "
            f"Consider increasing your sum insured or switching to a policy with fewer exclusions."
        )

    # Per-condition breakdown
    if declared_diseases:
        section(
            "What This Policy Means for Your Conditions",
        )
        breakdown = risk.get("disease_breakdown", {})
        for disease in declared_diseases:
            data      = breakdown.get(disease, {})
            cost      = data.get("treatment_cost", 0)
            oop_d     = data.get("expected_oop_5yr", 0)
            in_wait   = data.get("in_waiting_period", False)
            shortfall = data.get("sub_limit_shortfall", 0)

            if in_wait:
                card(
                    "Claim Will Be Rejected",
                    "red",
                    f"Your policy will not cover {disease} right now",
                    f"This condition falls under a waiting period. If you are hospitalised for {disease} "
                    f"before it ends, the insurer will reject the claim. You will pay the full "
                    f"{fmt_inr(cost)} yourself."
                )
            else:
                body = f"Treatment typically costs {fmt_inr(cost)}. "
                if shortfall > 0:
                    body += f"Sub-limits mean you will still pay roughly {fmt_inr(shortfall)} yourself. "
                if oop_d > 0:
                    body += f"Over 5 years, your estimated out-of-pocket for {disease} is {fmt_inr(oop_d)}."
                card(
                    "Covered with out-of-pocket costs",
                    "yellow" if oop_d > 10_000 else "green",
                    f"{disease} is covered by your policy",
                    body
                )

    st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-text">
    For informational purposes only. Not legal or financial advice.<br>
    Consult a licensed insurance advisor before making decisions.<br>
    Risk estimates use population-level data from ICMR, NHA, and IRDAI reports.
</div>
""", unsafe_allow_html=True)