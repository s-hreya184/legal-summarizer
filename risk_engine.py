# risk_engine.py
# Actuarially-grounded risk scoring for Indian health insurance policies
# Uses Expected Value, Moral Hazard weighting, and Financial Exposure Ratio

import math

# Source: NHA Health Accounts, IRDAI Annual Reports

AVG_TREATMENT_COST = {
    "Diabetes":            85_000,
    "Hypertension":        65_000,
    "Heart Disease":      350_000,
    "Thyroid Disorders":   70_000,
    "Dental Treatment":    45_000,
    "Cancer":             600_000,
    "Kidney Disease":     400_000,
    "Orthopedic Surgery": 200_000,
    "Maternity":          120_000,
    "Cataract":            55_000,
}

# Age-adjusted base disease probabilities
# Based on ICMR epidemiological data
BASE_DISEASE_PREVALENCE = {
    "Diabetes":            [0.04, 0.10, 0.18, 0.28, 0.38],
    "Hypertension":        [0.05, 0.12, 0.22, 0.35, 0.48],
    "Heart Disease":       [0.02, 0.06, 0.14, 0.25, 0.40],
    "Thyroid Disorders":   [0.03, 0.07, 0.12, 0.16, 0.20],
    "Dental Treatment":    [0.15, 0.20, 0.25, 0.30, 0.35],
    "Cancer":              [0.01, 0.02, 0.05, 0.09, 0.14],
    "Kidney Disease":      [0.01, 0.03, 0.07, 0.12, 0.18],
    "Orthopedic Surgery":  [0.03, 0.06, 0.10, 0.18, 0.28],
    "Maternity":           [0.10, 0.20, 0.05, 0.00, 0.00],
    "Cataract":            [0.00, 0.01, 0.05, 0.15, 0.35],
}

AGE_BANDS = [25, 35, 45, 55, 120]  


def _age_band_index(age: int) -> int:
    for i, upper in enumerate(AGE_BANDS):
        if age <= upper:
            return i
    return len(AGE_BANDS) - 1


def disease_probability(age: int, disease: str) -> float:
    """
    Age-stratified probability of hospitalization for a disease in a given year.
    For pre-existing conditions (user declared), probability is set to 1.0 (certain cost).
    """
    band = _age_band_index(age)
    probs = BASE_DISEASE_PREVALENCE.get(disease, [0.05, 0.08, 0.12, 0.18, 0.25])
    return probs[band]

# Expected Out-of-Pocket Calculator
# Uses actuarial Expected Value formula:
#   E[OOP] = P(claim) × [max(0, TreatmentCost − SumInsured)
#             + CoPay% × min(TreatmentCost, SumInsured)
#             + SubLimit shortfall]
def expected_out_of_pocket(
    policy: dict,
    age: int,
    declared_diseases: list[str],
    sum_insured: float = 500_000
) -> tuple[float, float, dict]:
    """
    Returns:
        total_oop       — expected out-of-pocket spend over next 5 years (INR)
        rejection_risk  — probability-weighted chance of at least one claim rejection
        breakdown       — per-disease detail for transparency
    """
    copay_rate     = policy.get("copay", 0.0)        
    waiting        = policy.get("waiting_periods", {}) 
    sub_limits     = policy.get("sub_limits", {})      
    room_rent_cap  = policy.get("room_rent_daily", None)
    deductible     = policy.get("deductible", 0.0)

    total_oop = 0.0
    total_rejection_prob = 0.0 
    no_rejection_prob = 1.0
    breakdown = {}

    all_diseases = list(set(
        list(AVG_TREATMENT_COST.keys()) +
        [d for d in declared_diseases if d in AVG_TREATMENT_COST]
    ))

    for disease in all_diseases:
        is_declared = disease in declared_diseases
        p = 1.0 if is_declared else disease_probability(age, disease)
        cost = AVG_TREATMENT_COST.get(disease, 100_000)

        # Room rent sub-limit: capped rooms inflate total bill
        # Rule: if room rent cap < standard, ~40% of total bill may be proportionately reduced
        room_rent_penalty = 0.0
        if room_rent_cap is not None:
            assumed_standard_rent = 5_000 
            if room_rent_cap < assumed_standard_rent:
                proportion = room_rent_cap / assumed_standard_rent
                room_rent_penalty = cost * (1 - proportion) * 0.4  

        # Sub-limit shortfall
        sub_limit = sub_limits.get(disease, sum_insured)
        sub_limit_shortfall = max(0, cost - sub_limit)

        # Deductible (flat per claim)
        deductible_exposure = min(deductible, cost)

        # Co-pay on admissible amount (after sub-limit)
        admissible = max(0, min(cost, sub_limit) - deductible)
        copay_exposure = admissible * copay_rate

        # Sum-insured exhaustion risk (if cost > SI)
        si_shortfall = max(0, cost - sum_insured)

        gross_oop = (
            sub_limit_shortfall +
            deductible_exposure +
            copay_exposure +
            si_shortfall +
            room_rent_penalty
        )

        # Waiting period → full claim rejection during waiting window
        in_waiting = disease in waiting
        if in_waiting:
            oop_this = cost  # policyholder bears entire cost
            rejection_contribution = p
        else:
            oop_this = p * gross_oop
            rejection_contribution = 0.0

        # Project over 5 years with compound probability
        # P(at least one event in 5 years) = 1 - (1-p)^5
        five_year_p = 1 - (1 - min(p, 0.99)) ** 5
        oop_5yr = five_year_p * gross_oop if not in_waiting else cost

        total_oop += oop_5yr
        no_rejection_prob *= (1 - rejection_contribution)

        breakdown[disease] = {
            "annual_probability": round(p, 3),
            "treatment_cost": cost,
            "expected_oop_5yr": round(oop_5yr),
            "in_waiting_period": in_waiting,
            "sub_limit_shortfall": round(sub_limit_shortfall),
            "copay_exposure": round(copay_exposure),
        }

    total_rejection_prob = 1 - no_rejection_prob
    return round(total_oop), round(total_rejection_prob, 4), breakdown

# Financial Exposure Ratio (FER)
# FER = Expected OOP / Annual Disposable Income
# Standard threshold: FER > 0.3 is "catastrophic"
# (WHO definition of catastrophic health expenditure)
def financial_exposure_ratio(expected_oop_5yr: float, annual_income: float) -> float:
    if annual_income <= 0:
        return 1.0
    disposable_income_5yr = annual_income * 5 * 0.7  # assume 70% disposable
    return min(expected_oop_5yr / disposable_income_5yr, 1.0)


# Policy Exclusion Density Score
# Penalizes policies with many exclusions / short waiting periods

def exclusion_density_score(
    num_exclusions: int,
    num_waiting_periods: int,
    num_hidden_limits: int,
    num_copayments: int
) -> float:
    """Returns a 0–1 score where 1 = maximum exclusion burden."""
    raw = (
        num_exclusions     * 3.0 +
        num_waiting_periods * 2.5 +
        num_hidden_limits  * 2.0 +
        num_copayments     * 1.5
    )
    # Sigmoid normalization: maps raw score to (0, 1)
    return 1 / (1 + math.exp(-0.1 * (raw - 15)))

# Final Composite Risk Score
# Combines:
#   1. Financial Exposure Ratio (40%) — personal financial impact
#   2. Claim Rejection Probability (35%) — actuarial rejection risk
#   3. Exclusion Density (25%) — policy complexity / bad faith signals

def calculate_risk_score(
    policy: dict,
    age: int,
    declared_diseases: list[str],
    annual_income: float,
    sum_insured: float,
    llm_exclusions: list,
    llm_waiting_periods: list,
    llm_hidden_limits: list,
    llm_copayments: list,
    llm_risk_score: int = 0
) -> dict:
    """
    Returns a comprehensive risk assessment dict.

    Args:
        policy: dict with keys copay, waiting_periods, sub_limits, deductible, room_rent_daily
        age: policyholder age
        declared_diseases: list of pre-existing conditions
        annual_income: annual income in INR
        sum_insured: policy sum insured in INR
        llm_*: lists from LLM extraction for density scoring
        llm_risk_score: LLM's own risk estimate (used as a soft signal)
    """

    # 1. Expected OOP and rejection risk
    oop_5yr, rejection_prob, breakdown = expected_out_of_pocket(
        policy, age, declared_diseases, sum_insured
    )

    # 2. Financial Exposure Ratio → 0 to 100
    fer = financial_exposure_ratio(oop_5yr, annual_income)
    fer_score = fer * 100

    # 3. Rejection probability → 0 to 100
    rejection_score = rejection_prob * 100

    # 4. Exclusion density → 0 to 100
    density = exclusion_density_score(
        num_exclusions=len(llm_exclusions),
        num_waiting_periods=len(llm_waiting_periods),
        num_hidden_limits=len(llm_hidden_limits),
        num_copayments=len(llm_copayments),
    )
    density_score = density * 100

    # 5. Weighted composite (LLM score used as a 10% soft signal)
    composite = (
        0.38 * fer_score +
        0.32 * rejection_score +
        0.22 * density_score +
        0.08 * llm_risk_score
    )
    final_score = int(min(100, max(0, composite)))

    # Risk tier labels 
    if final_score >= 75:
        tier = "High Risk"
        tier_detail = (
            "This policy poses serious financial risk. "
            "Claim rejection likelihood is high and out-of-pocket costs could be catastrophic."
        )
    elif final_score >= 50:
        tier = "Moderate Risk"
        tier_detail = (
            "Significant exclusions or waiting periods exist. "
            "You may face sizeable out-of-pocket expenses."
        )
    elif final_score >= 25:
        tier = "Low–Moderate Risk"
        tier_detail = (
            "Policy has some limitations but is generally manageable. "
            "Review waiting periods before filing claims."
        )
    else:
        tier = "Low Risk"
        tier_detail = "Policy appears transparent and claimant-friendly."

    # Catastrophic expenditure warning (WHO threshold: OOP > 40% annual income)
    catastrophic = fer >= 0.40

    return {
        "final_score": final_score,
        "risk_tier": tier,
        "risk_tier_detail": tier_detail,
        "expected_oop_5yr": oop_5yr,
        "rejection_probability_pct": round(rejection_prob * 100, 1),
        "financial_exposure_ratio": round(fer * 100, 1),
        "exclusion_density_score": round(density_score, 1),
        "catastrophic_expenditure_warning": catastrophic,
        "disease_breakdown": breakdown,
        "score_components": {
            "financial_exposure": round(fer_score, 1),
            "rejection_risk": round(rejection_score, 1),
            "exclusion_density": round(density_score, 1),
            "llm_signal": llm_risk_score,
        }
    }