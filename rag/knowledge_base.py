"""Medical Knowledge Base — Curated clinical knowledge corpus.

Contains evidence-based medical knowledge organized by category:
clinical guidelines, drug references, diagnostic criteria, and
treatment protocols. Each entry carries a source citation.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    """A single chunk of medical knowledge with provenance."""
    text: str
    source: str
    category: str  # guideline, drug, diagnostic, protocol
    tags: list[str]


# Curated medical knowledge corpus — expandable via document ingestion
MEDICAL_CORPUS: list[KnowledgeEntry] = [
    # ── Cardiology ────────────────────────────────────────────
    KnowledgeEntry(
        text="Acute coronary syndrome (ACS) includes STEMI, NSTEMI, and unstable angina. "
             "STEMI is defined by ST elevation >= 1mm in 2+ contiguous leads or new LBBB. "
             "Management: aspirin 325mg, heparin, dual antiplatelet therapy, and PCI within "
             "90 minutes of first medical contact for STEMI.",
        source="AHA/ACC STEMI Guidelines 2023",
        category="guideline",
        tags=["cardiology", "acs", "stemi", "chest pain"],
    ),
    KnowledgeEntry(
        text="Troponin I and T are cardiac-specific biomarkers. High-sensitivity troponin "
             "(hs-cTn) should be measured at presentation and 3-6 hours. A rise/fall pattern "
             "with at least one value above the 99th percentile indicates myocardial injury.",
        source="ESC Fourth Universal Definition of MI 2018",
        category="diagnostic",
        tags=["cardiology", "troponin", "biomarker", "mi"],
    ),
    KnowledgeEntry(
        text="Heart failure with reduced ejection fraction (HFrEF, EF <= 40%) is treated with "
             "guideline-directed medical therapy: ACEi/ARB/ARNI + beta-blocker + MRA + SGLT2i. "
             "Diuretics for congestion relief. CRT/ICD for EF <= 35%.",
        source="AHA/ACC/HFSA Heart Failure Guidelines 2022",
        category="guideline",
        tags=["cardiology", "heart failure", "hfref"],
    ),

    # ── Pulmonology ───────────────────────────────────────────
    KnowledgeEntry(
        text="Community-acquired pneumonia (CAP) empiric treatment: outpatient — amoxicillin "
             "or doxycycline; inpatient non-ICU — beta-lactam + macrolide or respiratory "
             "fluoroquinolone; ICU — beta-lactam + macrolide, add anti-MRSA if risk factors.",
        source="IDSA/ATS CAP Guidelines 2019",
        category="guideline",
        tags=["pulmonology", "pneumonia", "antibiotics"],
    ),
    KnowledgeEntry(
        text="COPD exacerbation severity: mild (managed with short-acting bronchodilators), "
             "moderate (requires antibiotics and/or systemic corticosteroids), severe (requires "
             "hospitalization or ED visit). Oral prednisone 40mg x5 days is standard.",
        source="GOLD Report 2024",
        category="guideline",
        tags=["pulmonology", "copd", "exacerbation"],
    ),
    KnowledgeEntry(
        text="Pulmonary embolism risk stratification: use Wells score or Geneva score. "
             "Low probability + negative D-dimer excludes PE. Intermediate/high probability "
             "requires CT pulmonary angiography. Massive PE (hemodynamic instability) needs "
             "systemic thrombolysis or catheter-directed therapy.",
        source="ESC PE Guidelines 2019",
        category="guideline",
        tags=["pulmonology", "pe", "embolism", "d-dimer"],
    ),

    # ── Emergency Medicine ────────────────────────────────────
    KnowledgeEntry(
        text="Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated "
             "host response to infection (SOFA score >= 2). qSOFA: RR >= 22, altered mentation, "
             "SBP <= 100. Septic shock: vasopressors needed for MAP >= 65 + lactate > 2.",
        source="Surviving Sepsis Campaign 2021",
        category="guideline",
        tags=["emergency", "sepsis", "sofa", "shock"],
    ),
    KnowledgeEntry(
        text="SEP-1 bundle: measure lactate, obtain blood cultures before antibiotics, "
             "administer broad-spectrum antibiotics within 1 hour, administer 30mL/kg "
             "crystalloid for hypotension or lactate >= 4, vasopressors if hypotension "
             "persists after fluid resuscitation.",
        source="CMS SEP-1 Core Measure",
        category="protocol",
        tags=["emergency", "sepsis", "bundle"],
    ),
    KnowledgeEntry(
        text="The Emergency Severity Index (ESI) is a 5-level triage algorithm: "
             "Level 1 — immediate life-saving intervention. Level 2 — high-risk situation, "
             "confused/lethargic, severe pain. Level 3 — multiple resources needed. "
             "Level 4 — one resource. Level 5 — no resources.",
        source="ESI Handbook v4, AHRQ",
        category="protocol",
        tags=["emergency", "triage", "esi"],
    ),

    # ── Neurology ─────────────────────────────────────────────
    KnowledgeEntry(
        text="Acute ischemic stroke: IV alteplase (tPA) within 4.5 hours of symptom onset. "
             "Dose: 0.9 mg/kg (max 90mg), 10% bolus, remainder over 60 min. Exclusions: "
             "hemorrhage on CT, BP > 185/110, INR > 1.7, platelets < 100K.",
        source="AHA/ASA Stroke Guidelines 2019",
        category="guideline",
        tags=["neurology", "stroke", "tpa", "thrombolysis"],
    ),
    KnowledgeEntry(
        text="NIHSS (National Institutes of Health Stroke Scale) scores: 0 = no stroke, "
             "1-4 = minor, 5-15 = moderate, 16-20 = moderate-severe, 21-42 = severe. "
             "Scores > 25 strongly predict need for mechanical thrombectomy evaluation.",
        source="AHA/ASA 2019",
        category="diagnostic",
        tags=["neurology", "stroke", "nihss"],
    ),

    # ── Pharmacology ──────────────────────────────────────────
    KnowledgeEntry(
        text="Nitroglycerin is CONTRAINDICATED in patients who have taken PDE5 inhibitors "
             "(sildenafil within 24h, tadalafil within 48h) due to risk of severe "
             "refractory hypotension. Also avoid in right ventricular infarction.",
        source="AHA STEMI Guidelines",
        category="drug",
        tags=["pharmacology", "nitroglycerin", "contraindication"],
    ),
    KnowledgeEntry(
        text="Warfarin-drug interactions: many antibiotics (especially fluoroquinolones, "
             "metronidazole, TMP-SMX) potentiate warfarin effect. NSAIDs increase bleeding "
             "risk. Amiodarone significantly increases INR. Monitor INR closely with any "
             "new medication start.",
        source="Clinical Pharmacology Reference",
        category="drug",
        tags=["pharmacology", "warfarin", "interactions"],
    ),
    KnowledgeEntry(
        text="Opioid dosing equivalence: Morphine 10mg IV = Hydromorphone 1.5mg IV = "
             "Fentanyl 100mcg IV. Always start low in opioid-naive patients. Naloxone "
             "0.4-2mg IV for reversal; may need repeated doses (half-life shorter than most opioids).",
        source="WHO Pain Ladder / Equianalgesic Tables",
        category="drug",
        tags=["pharmacology", "opioid", "pain", "naloxone"],
    ),

    # ── Infectious Disease ────────────────────────────────────
    KnowledgeEntry(
        text="Empiric antibiotic therapy for UTI: uncomplicated cystitis — nitrofurantoin "
             "100mg BID x5d or TMP-SMX 160/800 BID x3d. Pyelonephritis — ceftriaxone 1g IV "
             "or fluoroquinolone. Adjust based on local antibiogram and culture results.",
        source="IDSA UTI Guidelines 2010",
        category="guideline",
        tags=["infectious disease", "uti", "antibiotics"],
    ),

    # ── Surgery ───────────────────────────────────────────────
    KnowledgeEntry(
        text="Appendicitis: Alvarado score >= 7 strongly suggests appendicitis. CT abdomen "
             "with contrast is gold standard (sensitivity 94%, specificity 95%). Treatment: "
             "laparoscopic appendectomy. Antibiotics alone may be considered for uncomplicated "
             "cases per CODA trial.",
        source="CODA Trial, NEJM 2020; Surgical Guidelines",
        category="guideline",
        tags=["surgery", "appendicitis", "alvarado"],
    ),
]


class MedicalKnowledgeBase:
    """Manages the medical knowledge corpus for RAG retrieval."""

    def __init__(self, entries: list[KnowledgeEntry] | None = None):
        self.entries = entries or MEDICAL_CORPUS

    def get_texts(self) -> list[str]:
        return [e.text for e in self.entries]

    def get_sources(self) -> list[str]:
        return [e.source for e in self.entries]

    def get_metadata(self) -> list[dict]:
        return [
            {"source": e.source, "category": e.category, "tags": e.tags}
            for e in self.entries
        ]

    def filter_by_tags(self, tags: list[str]) -> list[KnowledgeEntry]:
        return [
            e for e in self.entries
            if any(t in e.tags for t in tags)
        ]

    def add_entry(self, entry: KnowledgeEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)
