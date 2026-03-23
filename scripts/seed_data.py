#!/usr/bin/env python3
"""Seed the HERA database with diverse patient scenarios.

Covers all risk levels (Low/Medium/High/Critical), ESI levels (1-5),
varied diagnoses, CDC events, and streaming events — so the DE dashboard
looks populated and colorful.

Usage:
    python -m scripts.seed_data          # default: hit localhost:8000
    python -m scripts.seed_data --url http://host:port
"""

import argparse
import json
import sys
import time
import requests

# ── 20 diverse patient scenarios ────────────────────────────────
PATIENTS = [
    # ESI-1: Life-threatening
    {
        "patient_id": "PT-SEED-001",
        "chief_complaint": "cardiac arrest, found unresponsive",
        "clinical_note": "72-year-old male found unresponsive at home. No pulse detected by EMS. CPR initiated in field. History of CHF and prior MI. Arrived intubated. ROSC achieved after 3 rounds of epinephrine.",
        "heart_rate": 38,
        "respiratory_rate": 6,
        "body_temperature": 35.2,
        "oxygen_saturation": 78,
        "systolic_bp": 65,
        "diastolic_bp": 35,
        "age": 72,
        "gender": "male",
        "medical_history": ["CHF", "prior MI", "hypertension"],
        "current_medications": ["metoprolol", "lisinopril", "aspirin"],
        "allergies": ["penicillin"],
    },
    {
        "patient_id": "PT-SEED-002",
        "chief_complaint": "respiratory arrest, apneic",
        "clinical_note": "68-year-old female with end-stage COPD, found apneic by family. Agonal respirations on arrival. SpO2 unreadable initially. Bag-valve mask ventilation started. Known DNR status being verified.",
        "heart_rate": 145,
        "respiratory_rate": 5,
        "body_temperature": 35.0,
        "oxygen_saturation": 72,
        "systolic_bp": 75,
        "diastolic_bp": 38,
        "age": 68,
        "gender": "female",
        "medical_history": ["COPD stage IV", "pulmonary hypertension"],
        "current_medications": ["albuterol", "prednisone", "oxygen 4L"],
        "allergies": [],
    },
    # ESI-2: Emergent high-risk
    {
        "patient_id": "PT-SEED-003",
        "chief_complaint": "chest pain radiating to left arm, diaphoretic",
        "clinical_note": "58-year-old male presenting with acute substernal chest pain radiating to left arm for 45 minutes. Diaphoretic, nauseous. ECG shows ST-elevation in leads II, III, aVF. Troponin pending. Aspirin and nitroglycerin administered.",
        "heart_rate": 110,
        "respiratory_rate": 24,
        "body_temperature": 37.2,
        "oxygen_saturation": 93,
        "systolic_bp": 165,
        "diastolic_bp": 95,
        "age": 58,
        "gender": "male",
        "medical_history": ["diabetes type 2", "hyperlipidemia", "smoker"],
        "current_medications": ["metformin", "atorvastatin"],
        "allergies": ["sulfa"],
    },
    {
        "patient_id": "PT-SEED-004",
        "chief_complaint": "acute stroke symptoms, left-sided weakness",
        "clinical_note": "76-year-old female with sudden onset left-sided weakness and slurred speech. Last known well 90 minutes ago. NIHSS score 14. CT head negative for hemorrhage. tPA candidacy being evaluated. BP 192/108.",
        "heart_rate": 88,
        "respiratory_rate": 18,
        "body_temperature": 37.0,
        "oxygen_saturation": 96,
        "systolic_bp": 192,
        "diastolic_bp": 108,
        "age": 76,
        "gender": "female",
        "medical_history": ["atrial fibrillation", "hypertension"],
        "current_medications": ["warfarin", "amlodipine"],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-005",
        "chief_complaint": "sepsis, fever and altered mental status",
        "clinical_note": "82-year-old male nursing home resident with fever 39.8C, confusion, and hypotension. WBC 22,000. Lactate 4.2. Urine cloudy. Suspected urosepsis. IV fluids and broad-spectrum antibiotics started. qSOFA score 3.",
        "heart_rate": 128,
        "respiratory_rate": 28,
        "body_temperature": 39.8,
        "oxygen_saturation": 90,
        "systolic_bp": 82,
        "diastolic_bp": 48,
        "age": 82,
        "gender": "male",
        "medical_history": ["BPH", "dementia", "recurrent UTIs"],
        "current_medications": ["tamsulosin", "donepezil"],
        "allergies": ["ciprofloxacin"],
    },
    {
        "patient_id": "PT-SEED-006",
        "chief_complaint": "anaphylaxis after bee sting",
        "clinical_note": "34-year-old female stung by bee 20 minutes ago. Rapidly developing facial swelling, urticaria, wheezing, and throat tightness. Known bee allergy but epipen not available. IM epinephrine 0.3mg given by EMS.",
        "heart_rate": 132,
        "respiratory_rate": 30,
        "body_temperature": 37.1,
        "oxygen_saturation": 88,
        "systolic_bp": 78,
        "diastolic_bp": 42,
        "age": 34,
        "gender": "female",
        "medical_history": ["bee allergy", "asthma"],
        "current_medications": ["fluticasone inhaler"],
        "allergies": ["bee venom", "shellfish"],
    },
    # ESI-3: Urgent — needs labs/imaging
    {
        "patient_id": "PT-SEED-007",
        "chief_complaint": "abdominal pain, right lower quadrant",
        "clinical_note": "28-year-old male with 12 hours of progressive RLQ pain, nausea, and low-grade fever. Rebound tenderness positive. McBurney's point tender. WBC 14,500. CT abdomen ordered to confirm appendicitis. NPO status.",
        "heart_rate": 92,
        "respiratory_rate": 18,
        "body_temperature": 38.2,
        "oxygen_saturation": 98,
        "systolic_bp": 128,
        "diastolic_bp": 78,
        "age": 28,
        "gender": "male",
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-008",
        "chief_complaint": "fever and cough for 5 days, worsening",
        "clinical_note": "55-year-old female diabetic with productive cough, fever to 38.6C, and increasing dyspnea over 5 days. Rhonchi in right lower lobe. SpO2 94% on room air. Chest X-ray shows right lower lobe infiltrate. Community-acquired pneumonia suspected.",
        "heart_rate": 98,
        "respiratory_rate": 22,
        "body_temperature": 38.6,
        "oxygen_saturation": 94,
        "systolic_bp": 118,
        "diastolic_bp": 72,
        "age": 55,
        "gender": "female",
        "medical_history": ["diabetes type 2", "obesity"],
        "current_medications": ["metformin", "glipizide"],
        "allergies": ["azithromycin"],
    },
    {
        "patient_id": "PT-SEED-009",
        "chief_complaint": "back pain with weakness in legs",
        "clinical_note": "64-year-old male with acute low back pain radiating to both legs, progressive weakness, and urinary retention since yesterday. History of prostate cancer. MRI spine urgently needed to rule out cauda equina syndrome.",
        "heart_rate": 85,
        "respiratory_rate": 16,
        "body_temperature": 37.0,
        "oxygen_saturation": 97,
        "systolic_bp": 145,
        "diastolic_bp": 88,
        "age": 64,
        "gender": "male",
        "medical_history": ["prostate cancer", "lumbar stenosis"],
        "current_medications": ["oxycodone", "gabapentin"],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-010",
        "chief_complaint": "dizziness and near-syncope episode",
        "clinical_note": "70-year-old female with two episodes of near-syncope today while standing. Feels lightheaded and weak. Orthostatic vitals positive. Hemoglobin 8.2 on last check 2 months ago. Guaiac positive stool in ED. GI bleed workup initiated.",
        "heart_rate": 102,
        "respiratory_rate": 18,
        "body_temperature": 36.8,
        "oxygen_saturation": 96,
        "systolic_bp": 105,
        "diastolic_bp": 62,
        "age": 70,
        "gender": "female",
        "medical_history": ["anemia", "diverticulosis", "NSAID use"],
        "current_medications": ["ibuprofen", "omeprazole"],
        "allergies": [],
    },
    # ESI-3: Moderate urgent
    {
        "patient_id": "PT-SEED-011",
        "chief_complaint": "painful urination and flank pain",
        "clinical_note": "42-year-old male with dysuria, frequency, and left flank pain for 2 days. Low-grade fever 37.9C. UA shows pyuria and bacteriuria. CT KUB ordered to rule out renal calculus. IV fluids started.",
        "heart_rate": 82,
        "respiratory_rate": 16,
        "body_temperature": 37.9,
        "oxygen_saturation": 99,
        "systolic_bp": 132,
        "diastolic_bp": 82,
        "age": 42,
        "gender": "male",
        "medical_history": ["kidney stones x2"],
        "current_medications": [],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-012",
        "chief_complaint": "weakness and fatigue for 2 weeks",
        "clinical_note": "48-year-old female with progressive fatigue, pallor, and exertional dyspnea. No bleeding identified. Labs show pancytopenia. Hematology consult requested. Peripheral smear shows abnormal cells.",
        "heart_rate": 95,
        "respiratory_rate": 19,
        "body_temperature": 37.1,
        "oxygen_saturation": 96,
        "systolic_bp": 110,
        "diastolic_bp": 68,
        "age": 48,
        "gender": "female",
        "medical_history": [],
        "current_medications": ["multivitamin"],
        "allergies": [],
    },
    # ESI-4: Less urgent, single resource
    {
        "patient_id": "PT-SEED-013",
        "chief_complaint": "laceration on right hand from kitchen knife",
        "clinical_note": "32-year-old male with 3cm laceration to right palm from a kitchen knife 1 hour ago. Bleeding controlled with pressure. Full sensation and motor function intact in all fingers. No tendon involvement suspected. Needs sutures.",
        "heart_rate": 78,
        "respiratory_rate": 14,
        "body_temperature": 36.9,
        "oxygen_saturation": 99,
        "systolic_bp": 124,
        "diastolic_bp": 76,
        "age": 32,
        "gender": "male",
        "medical_history": [],
        "current_medications": [],
        "allergies": ["latex"],
    },
    {
        "patient_id": "PT-SEED-014",
        "chief_complaint": "migraine headache, typical pattern",
        "clinical_note": "29-year-old female with recurrent migraine, typical aura followed by unilateral throbbing headache with photophobia and nausea. Onset 4 hours ago. Home medications not effective. Requesting IV treatment. Neuro exam normal.",
        "heart_rate": 72,
        "respiratory_rate": 14,
        "body_temperature": 36.7,
        "oxygen_saturation": 99,
        "systolic_bp": 118,
        "diastolic_bp": 72,
        "age": 29,
        "gender": "female",
        "medical_history": ["migraines with aura"],
        "current_medications": ["sumatriptan PRN"],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-015",
        "chief_complaint": "sprain of left ankle after fall",
        "clinical_note": "22-year-old male twisted left ankle playing basketball 3 hours ago. Moderate swelling over lateral malleolus. Weight-bearing with difficulty. Ottawa ankle rules negative for fracture criteria but X-ray ordered per protocol.",
        "heart_rate": 70,
        "respiratory_rate": 14,
        "body_temperature": 36.8,
        "oxygen_saturation": 100,
        "systolic_bp": 120,
        "diastolic_bp": 74,
        "age": 22,
        "gender": "male",
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-016",
        "chief_complaint": "rash on arms and trunk for 3 days",
        "clinical_note": "38-year-old female with pruritic maculopapular rash on arms and trunk for 3 days. Started new laundry detergent last week. No fever, no mucosal involvement. Likely contact dermatitis. Antihistamine and topical steroid to be prescribed.",
        "heart_rate": 74,
        "respiratory_rate": 14,
        "body_temperature": 36.8,
        "oxygen_saturation": 99,
        "systolic_bp": 116,
        "diastolic_bp": 72,
        "age": 38,
        "gender": "female",
        "medical_history": ["eczema"],
        "current_medications": [],
        "allergies": [],
    },
    # ESI-5: Non-urgent / routine
    {
        "patient_id": "PT-SEED-017",
        "chief_complaint": "prescription refill for blood pressure medication",
        "clinical_note": "56-year-old male requesting refill of amlodipine. Ran out 2 days ago. BP slightly elevated at 142/88, baseline is 130/80. No symptoms. Regular PCP follow-up scheduled next week. 30-day bridge prescription to be provided.",
        "heart_rate": 76,
        "respiratory_rate": 14,
        "body_temperature": 36.7,
        "oxygen_saturation": 99,
        "systolic_bp": 142,
        "diastolic_bp": 88,
        "age": 56,
        "gender": "male",
        "medical_history": ["hypertension"],
        "current_medications": ["amlodipine 5mg"],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-018",
        "chief_complaint": "suture removal from prior laceration repair",
        "clinical_note": "45-year-old female returning for suture removal from forearm laceration repaired 10 days ago. Wound well-healed, no signs of infection. Sutures to be removed. Wound care instructions provided.",
        "heart_rate": 68,
        "respiratory_rate": 12,
        "body_temperature": 36.6,
        "oxygen_saturation": 99,
        "systolic_bp": 118,
        "diastolic_bp": 74,
        "age": 45,
        "gender": "female",
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
    },
    # More mid-range cases to fill out Medium/High risk
    {
        "patient_id": "PT-SEED-019",
        "chief_complaint": "anxiety and racing heart since morning",
        "clinical_note": "25-year-old female with acute anxiety attack. Reports racing heart, tingling in hands, and feeling of impending doom. Started after stressful work meeting. No chest pain. ECG shows sinus tachycardia, otherwise normal.",
        "heart_rate": 112,
        "respiratory_rate": 22,
        "body_temperature": 37.0,
        "oxygen_saturation": 99,
        "systolic_bp": 138,
        "diastolic_bp": 86,
        "age": 25,
        "gender": "female",
        "medical_history": ["generalized anxiety disorder"],
        "current_medications": ["sertraline"],
        "allergies": [],
    },
    {
        "patient_id": "PT-SEED-020",
        "chief_complaint": "well-child checkup, routine",
        "clinical_note": "3-year-old male brought in for well-child check. Up to date on vaccinations. Growing along 50th percentile. No acute concerns. Parents report normal development milestones. Active and playful in exam room.",
        "heart_rate": 105,
        "respiratory_rate": 24,
        "body_temperature": 37.0,
        "oxygen_saturation": 99,
        "systolic_bp": 90,
        "diastolic_bp": 55,
        "age": 3,
        "gender": "male",
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
    },
]


def seed(base_url: str = "http://localhost:8000"):
    """Send all patient scenarios through the command center."""
    print(f"\n{'=' * 60}")
    print(f"  HERA Seed Data — {len(PATIENTS)} diverse patients")
    print(f"  Target: {base_url}")
    print(f"{'=' * 60}\n")

    # Check health first
    try:
        r = requests.get(f"{base_url}/api/health", timeout=5)
        r.raise_for_status()
        print(f"✓ API healthy: {r.json().get('version')}\n")
    except Exception as e:
        print(f"✗ API not reachable: {e}")
        print("  Start the server first: uvicorn serving.api:app --reload")
        sys.exit(1)

    results = {"success": 0, "failed": 0, "risk_levels": {}, "esi_levels": {}}

    for i, patient in enumerate(PATIENTS, 1):
        pid = patient["patient_id"]
        complaint = patient["chief_complaint"][:50]
        print(f"[{i:2d}/{len(PATIENTS)}] {pid}: {complaint}...", end=" ", flush=True)

        try:
            start = time.time()
            r = requests.post(
                f"{base_url}/api/command-center",
                json=patient,
                timeout=60,
                headers={"X-API-Key": "hera-dev-key"},
            )
            elapsed = time.time() - start

            if r.status_code == 200:
                data = r.json()
                # Extract risk and ESI from stages
                risk_level = "?"
                esi = "?"
                for stage in data.get("stages", []):
                    if stage.get("system") == "risk_predictor":
                        sr = stage.get("result", {})
                        risk_level = sr.get("risk_level", sr.get("prediction", "?"))
                    elif stage.get("system") == "agents":
                        sr = stage.get("result", {})
                        esi = sr.get("triage", {}).get("esi_level", "?")

                results["success"] += 1
                results["risk_levels"][risk_level] = (
                    results["risk_levels"].get(risk_level, 0) + 1
                )
                results["esi_levels"][str(esi)] = (
                    results["esi_levels"].get(str(esi), 0) + 1
                )
                print(f"✓ {elapsed:.1f}s  Risk={risk_level}  ESI={esi}")
            else:
                results["failed"] += 1
                err = r.text[:80]
                print(f"✗ HTTP {r.status_code}: {err}")
        except Exception as e:
            results["failed"] += 1
            print(f"✗ Error: {e}")

        # Small delay to spread timestamps
        time.sleep(0.3)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Results: {results['success']} success, {results['failed']} failed")
    print(f"  Risk Distribution: {json.dumps(results['risk_levels'], indent=2)}")
    print(f"  ESI Distribution:  {json.dumps(results['esi_levels'], indent=2)}")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed HERA with diverse patient data")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    seed(args.url)
