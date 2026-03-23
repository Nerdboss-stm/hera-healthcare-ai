"""Diagnostic Agent — Generates ranked differential diagnoses.

Combines clinical note analysis, vital sign patterns, triage context,
and a comprehensive medical knowledge base to produce probable diagnoses
with ICD-10 codes, supporting evidence, and recommended confirmatory tests.

Knowledge base covers 20+ symptom categories with 80+ conditions including
rare and life-threatening presentations.
"""

from __future__ import annotations

import logging

from agents.protocols import (
    PatientContext,
    TriageResult,
    DiagnosticResult,
    Diagnosis,
)

logger = logging.getLogger(__name__)


def _dx(condition, icd10, evidence, ruling_out):
    """Shorthand constructor for Diagnosis dataclass."""
    return Diagnosis(
        condition=condition,
        icd10_code=icd10,
        probability=0.0,
        supporting_evidence=evidence,
        ruling_out=ruling_out,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLINICAL KNOWLEDGE BASE — 20 categories, 80+ diagnoses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLINICAL_KNOWLEDGE = {
    "chest pain": [
        _dx(
            "Acute Coronary Syndrome (STEMI)",
            "I21.3",
            [
                "chest pain",
                "diaphoresis",
                "radiation to arm",
                "ST elevation",
                "troponin elevated",
            ],
            ["normal troponin", "normal ECG"],
        ),
        _dx(
            "Aortic Dissection",
            "I71.01",
            [
                "tearing pain",
                "radiating to back",
                "unequal pulses",
                "hypertension",
                "widened mediastinum",
            ],
            ["normal CT angiography", "equal pulses"],
        ),
        _dx(
            "Cardiac Tamponade",
            "I31.4",
            [
                "muffled heart sounds",
                "JVD",
                "hypotension",
                "pulsus paradoxus",
                "Beck's triad",
            ],
            ["normal echo", "no pericardial effusion"],
        ),
        _dx(
            "Pulmonary Embolism",
            "I26.99",
            [
                "pleuritic chest pain",
                "dyspnea",
                "tachycardia",
                "hypoxia",
                "DVT history",
                "hemoptysis",
            ],
            ["negative D-dimer", "normal CT-PA"],
        ),
        _dx(
            "Tension Pneumothorax",
            "J93.0",
            [
                "sudden onset",
                "absent breath sounds",
                "tracheal deviation",
                "hypotension",
                "distended neck veins",
            ],
            ["normal chest X-ray", "bilateral breath sounds"],
        ),
        _dx(
            "Myocarditis",
            "I40.9",
            [
                "chest pain",
                "recent viral illness",
                "heart failure",
                "arrhythmia",
                "elevated troponin",
                "young patient",
            ],
            ["normal echo", "normal troponin"],
        ),
        _dx(
            "Pericarditis",
            "I30.9",
            [
                "sharp chest pain",
                "worse lying flat",
                "friction rub",
                "diffuse ST elevation",
                "relieved by leaning forward",
            ],
            ["normal ECG", "no friction rub"],
        ),
    ],
    "dyspnea": [
        _dx(
            "Acute Exacerbation of COPD",
            "J44.1",
            [
                "wheezing",
                "productive cough",
                "smoking history",
                "barrel chest",
                "pursed lip breathing",
            ],
            ["no prior COPD diagnosis", "no smoking history"],
        ),
        _dx(
            "Congestive Heart Failure (Acute)",
            "I50.9",
            [
                "orthopnea",
                "edema",
                "JVD",
                "crackles",
                "weight gain",
                "paroxysmal nocturnal dyspnea",
            ],
            ["normal BNP", "normal echo"],
        ),
        _dx(
            "Pneumonia",
            "J18.9",
            ["fever", "productive cough", "crackles", "consolidation", "rigors"],
            ["clear chest X-ray", "no fever"],
        ),
        _dx(
            "Acute Respiratory Distress Syndrome",
            "J80",
            [
                "bilateral infiltrates",
                "hypoxemia",
                "rapid onset",
                "mechanical ventilation",
                "PaO2/FiO2 < 300",
            ],
            ["cardiogenic pulmonary edema", "normal PaO2"],
        ),
        _dx(
            "Anaphylaxis",
            "T78.2",
            [
                "urticaria",
                "angioedema",
                "wheezing",
                "hypotension",
                "allergen exposure",
                "throat swelling",
                "epinephrine",
            ],
            ["no allergen exposure", "gradual onset"],
        ),
        _dx(
            "Pulmonary Hypertension Crisis",
            "I27.0",
            [
                "progressive dyspnea",
                "syncope",
                "right heart failure",
                "loud P2",
                "elevated PA pressure",
            ],
            ["normal echo", "normal PA pressure"],
        ),
    ],
    "headache": [
        _dx(
            "Subarachnoid Hemorrhage",
            "I60.9",
            [
                "thunderclap onset",
                "worst headache of life",
                "neck stiffness",
                "photophobia",
                "vomiting",
            ],
            ["gradual onset", "normal CT head"],
        ),
        _dx(
            "Meningitis",
            "G03.9",
            [
                "fever",
                "neck stiffness",
                "photophobia",
                "altered mental status",
                "Kernig sign",
                "Brudzinski sign",
            ],
            ["afebrile", "normal LP"],
        ),
        _dx(
            "Epidural Hematoma",
            "S06.4X0A",
            [
                "lucid interval",
                "head trauma",
                "temporal fracture",
                "pupil dilation",
                "rapid deterioration",
            ],
            ["no trauma history", "normal CT head"],
        ),
        _dx(
            "Brain Tumor / Mass Lesion",
            "C71.9",
            [
                "progressive headache",
                "worse in morning",
                "nausea",
                "focal deficits",
                "papilledema",
                "seizure",
            ],
            ["normal MRI", "acute onset"],
        ),
        _dx(
            "Migraine with Aura",
            "G43.109",
            [
                "unilateral",
                "pulsating",
                "photophobia",
                "nausea",
                "visual aura",
                "preceding scotoma",
            ],
            ["thunderclap onset", "focal neuro deficits", "fever"],
        ),
        _dx(
            "Cerebral Venous Sinus Thrombosis",
            "I67.6",
            [
                "headache",
                "oral contraceptives",
                "papilledema",
                "seizure",
                "focal deficits",
                "pregnancy",
            ],
            ["normal MRV", "no risk factors"],
        ),
    ],
    "abdominal pain": [
        _dx(
            "Acute Appendicitis",
            "K35.80",
            [
                "RLQ pain",
                "rebound tenderness",
                "fever",
                "anorexia",
                "migration from periumbilical",
            ],
            ["normal CT abdomen", "pain resolves"],
        ),
        _dx(
            "Ruptured Ectopic Pregnancy",
            "O00.10",
            [
                "lower abdominal pain",
                "vaginal bleeding",
                "missed period",
                "positive pregnancy test",
                "shoulder pain",
                "hemodynamic instability",
            ],
            ["negative pregnancy test", "IUP on ultrasound"],
        ),
        _dx(
            "Perforated Viscus",
            "K63.1",
            [
                "sudden severe pain",
                "rigid abdomen",
                "guarding",
                "free air on X-ray",
                "peritonitis",
                "rebound",
            ],
            ["soft abdomen", "no free air"],
        ),
        _dx(
            "Mesenteric Ischemia",
            "K55.069",
            [
                "pain out of proportion",
                "bloody stool",
                "atrial fibrillation",
                "lactic acidosis",
                "elderly patient",
                "minimal tenderness",
            ],
            ["young patient", "normal lactate"],
        ),
        _dx(
            "Acute Pancreatitis",
            "K85.9",
            [
                "epigastric pain",
                "radiating to back",
                "nausea",
                "vomiting",
                "elevated lipase",
                "alcohol history",
                "gallstones",
            ],
            ["normal lipase", "no tenderness"],
        ),
        _dx(
            "Small Bowel Obstruction",
            "K56.60",
            [
                "distension",
                "vomiting",
                "obstipation",
                "prior surgery",
                "high-pitched bowel sounds",
            ],
            ["normal abdominal X-ray", "passing flatus"],
        ),
        _dx(
            "Abdominal Aortic Aneurysm Rupture",
            "I71.3",
            [
                "sudden abdominal pain",
                "back pain",
                "pulsatile mass",
                "hypotension",
                "elderly male",
                "syncope",
            ],
            ["normal CT", "hemodynamically stable"],
        ),
    ],
    "fever": [
        _dx(
            "Sepsis / Septic Shock",
            "A41.9",
            [
                "fever",
                "tachycardia",
                "hypotension",
                "elevated WBC",
                "elevated lactate",
                "organ dysfunction",
            ],
            ["hemodynamically stable", "normal lactate"],
        ),
        _dx(
            "Bacterial Meningitis",
            "G00.9",
            [
                "fever",
                "neck stiffness",
                "altered mental status",
                "petechial rash",
                "photophobia",
            ],
            ["afebrile", "normal LP", "no neck stiffness"],
        ),
        _dx(
            "Endocarditis",
            "I33.0",
            [
                "fever",
                "new murmur",
                "Janeway lesions",
                "Osler nodes",
                "splinter hemorrhages",
                "IV drug use",
                "prosthetic valve",
            ],
            ["normal echo", "negative blood cultures"],
        ),
        _dx(
            "Necrotizing Fasciitis",
            "M72.6",
            [
                "rapidly spreading erythema",
                "pain out of proportion",
                "crepitus",
                "bullae",
                "fever",
                "toxic appearance",
            ],
            ["well-appearing", "slowly progressive"],
        ),
        _dx(
            "Malignant Hyperthermia",
            "T88.3",
            [
                "extreme hyperthermia",
                "muscle rigidity",
                "tachycardia",
                "recent anesthesia",
                "elevated CK",
                "metabolic acidosis",
            ],
            ["no recent anesthesia", "gradual fever"],
        ),
        _dx(
            "Febrile Neutropenia",
            "D70.9",
            [
                "fever",
                "recent chemotherapy",
                "neutropenia",
                "immunosuppressed",
                "mucositis",
            ],
            ["normal WBC", "no recent chemotherapy"],
        ),
    ],
    "altered mental status": [
        _dx(
            "Stroke (CVA) — Ischemic",
            "I63.9",
            [
                "focal deficits",
                "sudden onset",
                "facial droop",
                "arm drift",
                "speech difficulty",
                "atrial fibrillation",
            ],
            ["normal CT/MRI", "no focal findings"],
        ),
        _dx(
            "Intracranial Hemorrhage",
            "I61.9",
            [
                "sudden headache",
                "vomiting",
                "hypertension",
                "focal deficits",
                "decreased consciousness",
                "anticoagulant use",
            ],
            ["normal CT head", "no headache"],
        ),
        _dx(
            "Diabetic Ketoacidosis",
            "E10.10",
            [
                "fruity breath",
                "Kussmaul breathing",
                "polyuria",
                "polydipsia",
                "abdominal pain",
                "dehydration",
                "diabetes history",
            ],
            ["normal glucose", "normal pH"],
        ),
        _dx(
            "Status Epilepticus",
            "G41.0",
            [
                "prolonged seizure",
                "postictal state",
                "incontinence",
                "tongue bite",
                "witnessed convulsion",
            ],
            ["alert and oriented", "no seizure history"],
        ),
        _dx(
            "Hyponatremia (Severe)",
            "E87.1",
            [
                "confusion",
                "seizure",
                "nausea",
                "diuretic use",
                "low sodium",
                "cerebral edema",
            ],
            ["normal sodium", "alert"],
        ),
        _dx(
            "Opioid Overdose",
            "T40.2X1A",
            [
                "pinpoint pupils",
                "respiratory depression",
                "altered consciousness",
                "needle marks",
                "responds to naloxone",
            ],
            ["normal pupils", "negative tox screen"],
        ),
        _dx(
            "Thyroid Storm",
            "E05.5",
            [
                "hyperthermia",
                "tachycardia",
                "agitation",
                "tremor",
                "weight loss",
                "goiter",
                "exophthalmos",
            ],
            ["normal TSH", "hypothermic"],
        ),
    ],
    "back pain": [
        _dx(
            "Cauda Equina Syndrome",
            "G83.4",
            [
                "saddle anesthesia",
                "urinary retention",
                "bilateral leg weakness",
                "bowel incontinence",
                "progressive neurological deficit",
            ],
            ["normal MRI", "intact sphincter tone"],
        ),
        _dx(
            "Lumbar Disc Herniation",
            "M51.16",
            [
                "radiculopathy",
                "leg pain",
                "numbness",
                "disc bulge",
                "sciatica",
                "positive SLR",
            ],
            ["normal MRI", "no neurological deficit"],
        ),
        _dx(
            "Spinal Epidural Abscess",
            "G06.1",
            [
                "fever",
                "back pain",
                "neurological deficit",
                "IV drug use",
                "point tenderness",
                "recent procedure",
            ],
            ["afebrile", "normal MRI", "no risk factors"],
        ),
        _dx(
            "Vertebral Compression Fracture",
            "M80.08XA",
            [
                "osteoporosis",
                "point tenderness",
                "kyphosis",
                "height loss",
                "post-menopausal",
                "steroid use",
            ],
            ["young patient", "no trauma"],
        ),
        _dx(
            "Kidney Stone (Nephrolithiasis)",
            "N20.0",
            [
                "flank pain",
                "colicky",
                "hematuria",
                "radiating to groin",
                "restless",
                "CVA tenderness",
            ],
            ["normal CT", "no hematuria"],
        ),
        _dx(
            "Abdominal Aortic Aneurysm (Leaking)",
            "I71.3",
            [
                "sudden back pain",
                "abdominal pain",
                "pulsatile mass",
                "hypotension",
                "elderly",
            ],
            ["young patient", "normal CT", "stable vitals"],
        ),
    ],
    "skin wound": [
        _dx(
            "Necrotizing Fasciitis",
            "M72.6",
            [
                "rapidly spreading",
                "pain out of proportion",
                "crepitus",
                "bullae",
                "dusky skin",
                "fever",
            ],
            ["well-appearing", "slowly progressive"],
        ),
        _dx(
            "Cellulitis with Sepsis",
            "L03.90",
            ["erythema", "warmth", "spreading", "fever", "lymphangitis", "bacteremia"],
            ["afebrile", "localized only"],
        ),
        _dx(
            "Compartment Syndrome",
            "T79.A0",
            [
                "pain out of proportion",
                "pain with passive stretch",
                "tense compartment",
                "paresthesia",
                "trauma history",
                "pulselessness late",
            ],
            ["soft compartment", "no trauma"],
        ),
        _dx(
            "Stevens-Johnson Syndrome / TEN",
            "L51.1",
            [
                "mucosal involvement",
                "target lesions",
                "recent medication change",
                "fever",
                "skin sloughing",
                "Nikolsky sign positive",
            ],
            ["no medication changes", "no mucosal lesions"],
        ),
        _dx(
            "Deep Vein Thrombosis",
            "I82.40",
            [
                "unilateral leg swelling",
                "calf pain",
                "warmth",
                "Homan sign",
                "recent immobilization",
            ],
            ["bilateral", "normal ultrasound"],
        ),
    ],
    "joint pain": [
        _dx(
            "Septic Arthritis",
            "M00.9",
            [
                "fever",
                "single hot joint",
                "erythema",
                "inability to bear weight",
                "elevated WBC",
                "recent bacteremia",
            ],
            ["afebrile", "multiple joints", "chronic"],
        ),
        _dx(
            "Gout / Pseudogout",
            "M10.9",
            [
                "acute onset",
                "erythema",
                "swelling",
                "first MTP",
                "elevated uric acid",
                "crystal on aspirate",
            ],
            ["normal uric acid", "chronic symptoms"],
        ),
        _dx(
            "Rheumatoid Arthritis Flare",
            "M06.9",
            [
                "symmetric polyarthritis",
                "morning stiffness",
                "MCP/PIP joints",
                "rheumatoid factor positive",
                "swan neck deformity",
            ],
            ["single joint", "acute monoarthritis"],
        ),
        _dx(
            "Osteomyelitis",
            "M86.9",
            [
                "bone pain",
                "fever",
                "erythema over bone",
                "elevated ESR/CRP",
                "diabetes",
                "open wound",
            ],
            ["normal MRI", "afebrile", "no wound"],
        ),
        _dx(
            "Lupus Arthritis (SLE Flare)",
            "M32.9",
            [
                "polyarthritis",
                "malar rash",
                "photosensitivity",
                "oral ulcers",
                "serositis",
                "positive ANA",
            ],
            ["negative ANA", "male patient"],
        ),
    ],
    "pregnancy complication": [
        _dx(
            "Eclampsia",
            "O15.0",
            [
                "seizure",
                "hypertension",
                "proteinuria",
                "headache",
                "visual changes",
                "edema",
                "pregnant",
                "third trimester",
            ],
            ["normotensive", "not pregnant"],
        ),
        _dx(
            "Placental Abruption",
            "O45.9",
            [
                "vaginal bleeding",
                "abdominal pain",
                "uterine tenderness",
                "fetal distress",
                "board-like uterus",
                "hypertension",
            ],
            ["painless bleeding", "normal fetal heart tones"],
        ),
        _dx(
            "HELLP Syndrome",
            "O14.2",
            [
                "hemolysis",
                "elevated liver enzymes",
                "low platelets",
                "RUQ pain",
                "nausea",
                "pregnant",
                "hypertension",
            ],
            ["normal platelets", "normal LFTs"],
        ),
        _dx(
            "Ruptured Ectopic Pregnancy",
            "O00.10",
            [
                "lower abdominal pain",
                "vaginal bleeding",
                "missed period",
                "positive pregnancy test",
                "shoulder pain",
                "syncope",
            ],
            ["negative pregnancy test", "IUP confirmed"],
        ),
    ],
    "trauma": [
        _dx(
            "Traumatic Brain Injury",
            "S06.9X0A",
            [
                "head trauma",
                "loss of consciousness",
                "vomiting",
                "amnesia",
                "GCS < 15",
                "pupil asymmetry",
                "battle sign",
            ],
            ["GCS 15", "no loss of consciousness"],
        ),
        _dx(
            "Splenic Rupture",
            "S36.09XA",
            [
                "left upper quadrant pain",
                "Kehr sign",
                "hypotension",
                "blunt abdominal trauma",
                "left shoulder pain",
                "tachycardia",
            ],
            ["hemodynamically stable", "normal FAST"],
        ),
        _dx(
            "Flail Chest",
            "S22.5XXA",
            [
                "paradoxical chest movement",
                "multiple rib fractures",
                "respiratory distress",
                "blunt chest trauma",
                "crepitus",
                "hypoxia",
            ],
            ["single rib fracture", "normal breathing"],
        ),
        _dx(
            "Hemothorax",
            "S27.1XXA",
            [
                "decreased breath sounds",
                "dullness to percussion",
                "chest trauma",
                "hypotension",
                "tachycardia",
                "blood on chest tube",
            ],
            ["clear lung fields", "normal CXR"],
        ),
        _dx(
            "Pelvic Fracture with Hemorrhage",
            "S32.9XXA",
            [
                "pelvic pain",
                "hemodynamic instability",
                "MVC",
                "fall from height",
                "unable to bear weight",
                "blood at urethral meatus",
            ],
            ["stable pelvis", "ambulatory"],
        ),
    ],
    "cardiac arrest": [
        _dx(
            "Ventricular Fibrillation",
            "I49.01",
            [
                "witnessed arrest",
                "shockable rhythm",
                "no pulse",
                "cardiac history",
                "chest pain preceding",
            ],
            ["asystole", "PEA"],
        ),
        _dx(
            "Massive Pulmonary Embolism",
            "I26.02",
            [
                "sudden cardiac arrest",
                "preceding dyspnea",
                "DVT history",
                "right heart strain",
                "PEA arrest",
                "distended neck veins",
            ],
            ["shockable rhythm", "no DVT risk"],
        ),
        _dx(
            "Tension Pneumothorax",
            "J93.0",
            [
                "absent breath sounds",
                "tracheal deviation",
                "trauma",
                "distended neck veins",
                "hypoxia",
            ],
            ["bilateral breath sounds", "normal CXR"],
        ),
        _dx(
            "Cardiac Tamponade",
            "I31.4",
            [
                "muffled heart sounds",
                "distended neck veins",
                "PEA arrest",
                "pericardial effusion",
                "recent cardiac surgery",
            ],
            ["normal echo", "loud heart sounds"],
        ),
    ],
    "eye symptoms": [
        _dx(
            "Acute Angle-Closure Glaucoma",
            "H40.20X0",
            [
                "severe eye pain",
                "halos around lights",
                "nausea",
                "mid-dilated pupil",
                "rock-hard eye",
                "decreased vision",
            ],
            ["normal IOP", "normal pupil reactivity"],
        ),
        _dx(
            "Retinal Detachment",
            "H33.00",
            [
                "flashing lights",
                "floaters",
                "curtain over vision",
                "painless vision loss",
                "myopia",
            ],
            ["normal fundoscopy", "no visual changes"],
        ),
        _dx(
            "Central Retinal Artery Occlusion",
            "H34.10",
            [
                "sudden painless vision loss",
                "cherry red spot",
                "afferent pupillary defect",
                "pale retina",
            ],
            ["gradual onset", "normal fundus"],
        ),
    ],
    "psychiatric emergency": [
        _dx(
            "Serotonin Syndrome",
            "T43.205A",
            [
                "agitation",
                "hyperthermia",
                "clonus",
                "tremor",
                "diarrhea",
                "SSRI use",
                "recent medication change",
            ],
            ["no serotonergic medications", "normal temperature"],
        ),
        _dx(
            "Neuroleptic Malignant Syndrome",
            "G21.0",
            [
                "muscle rigidity",
                "hyperthermia",
                "altered consciousness",
                "autonomic instability",
                "elevated CK",
                "antipsychotic use",
            ],
            ["no antipsychotic use", "normal CK"],
        ),
        _dx(
            "Acute Psychosis with Medical Cause",
            "F06.0",
            [
                "hallucinations",
                "confusion",
                "new onset in elderly",
                "metabolic derangement",
                "infection",
                "no psychiatric history",
            ],
            ["known psychiatric history", "young patient"],
        ),
    ],
    "pediatric emergency": [
        _dx(
            "Intussusception",
            "K56.1",
            [
                "currant jelly stool",
                "intermittent crying",
                "sausage-shaped mass",
                "vomiting",
                "drawing up legs",
                "infant",
            ],
            ["adult patient", "normal ultrasound"],
        ),
        _dx(
            "Kawasaki Disease",
            "M30.3",
            [
                "prolonged fever > 5 days",
                "conjunctivitis",
                "strawberry tongue",
                "rash",
                "extremity changes",
                "lymphadenopathy",
                "child",
            ],
            ["adult patient", "brief fever"],
        ),
        _dx(
            "Epiglottitis",
            "J05.1",
            [
                "stridor",
                "drooling",
                "tripod position",
                "high fever",
                "muffled voice",
                "dysphagia",
                "unvaccinated",
            ],
            ["barking cough", "no fever"],
        ),
    ],
    "endocrine crisis": [
        _dx(
            "Diabetic Ketoacidosis",
            "E10.10",
            [
                "fruity breath",
                "Kussmaul breathing",
                "polyuria",
                "polydipsia",
                "dehydration",
                "abdominal pain",
                "hyperglycemia",
            ],
            ["normal glucose", "normal pH"],
        ),
        _dx(
            "Adrenal Crisis (Addisonian)",
            "E27.2",
            [
                "hypotension refractory to fluids",
                "hypoglycemia",
                "hyponatremia",
                "hyperkalemia",
                "hyperpigmentation",
                "recent steroid cessation",
            ],
            ["normal cortisol", "normotensive"],
        ),
        _dx(
            "Thyroid Storm",
            "E05.5",
            [
                "extreme tachycardia",
                "hyperthermia",
                "agitation",
                "delirium",
                "goiter",
                "recent surgery",
                "iodine exposure",
            ],
            ["normal TSH", "bradycardia"],
        ),
        _dx(
            "Pheochromocytoma Crisis",
            "D35.00",
            [
                "paroxysmal hypertension",
                "headache",
                "diaphoresis",
                "palpitations",
                "pallor",
                "adrenal mass",
                "anxiety",
            ],
            ["sustained normotension", "normal metanephrines"],
        ),
        _dx(
            "Myxedema Coma",
            "E03.5",
            [
                "hypothermia",
                "bradycardia",
                "altered mental status",
                "hyponatremia",
                "hypoglycemia",
                "non-pitting edema",
                "hypothyroidism history",
            ],
            ["normal TSH", "hyperthermia"],
        ),
    ],
    "neurological emergency": [
        _dx(
            "Guillain-Barré Syndrome",
            "G61.0",
            [
                "ascending weakness",
                "areflexia",
                "recent infection",
                "bilateral",
                "respiratory compromise",
                "back pain",
            ],
            ["upper motor neuron signs", "asymmetric"],
        ),
        _dx(
            "Spinal Cord Compression",
            "G95.20",
            [
                "bilateral weakness",
                "sensory level",
                "bowel/bladder dysfunction",
                "back pain",
                "known malignancy",
                "band-like pain",
            ],
            ["normal MRI", "unilateral symptoms"],
        ),
        _dx(
            "Myasthenia Gravis Crisis",
            "G70.01",
            [
                "progressive weakness",
                "diplopia",
                "ptosis",
                "dysphagia",
                "respiratory failure",
                "fatigable weakness",
                "worse with activity",
            ],
            ["fixed weakness", "normal EMG"],
        ),
        _dx(
            "Acute Spinal Cord Infarct",
            "G95.11",
            [
                "sudden bilateral weakness",
                "loss of pain/temp sensation",
                "preserved proprioception",
                "back pain",
                "aortic surgery history",
            ],
            ["gradual onset", "normal MRI"],
        ),
    ],
    "vascular emergency": [
        _dx(
            "Ruptured Abdominal Aortic Aneurysm",
            "I71.3",
            [
                "sudden abdominal/back pain",
                "pulsatile mass",
                "hypotension",
                "syncope",
                "elderly male",
                "known aneurysm",
            ],
            ["hemodynamically stable", "no pulsatile mass"],
        ),
        _dx(
            "Acute Limb Ischemia",
            "I74.3",
            [
                "six Ps — pain",
                "pallor",
                "pulselessness",
                "paresthesia",
                "paralysis",
                "poikilothermia",
                "atrial fibrillation",
            ],
            ["warm extremity", "palpable pulses"],
        ),
        _dx(
            "Aortic Dissection (Type A)",
            "I71.01",
            [
                "tearing chest pain",
                "back pain",
                "blood pressure differential",
                "aortic regurgitation",
                "widened mediastinum",
                "Marfan habitus",
            ],
            ["normal CT aorta", "equal BP bilateral"],
        ),
    ],
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SYMPTOM ALIASES — maps common phrases to canonical categories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYMPTOM_ALIASES: dict[str, str] = {
    # Respiratory
    "shortness of breath": "dyspnea",
    "sob": "dyspnea",
    "difficulty breathing": "dyspnea",
    "breathless": "dyspnea",
    "can't breathe": "dyspnea",
    "cough": "dyspnea",
    "wheezing": "dyspnea",
    "asthma": "dyspnea",
    "hypoxia": "dyspnea",
    "oxygen": "dyspnea",
    # Chest
    "chest tightness": "chest pain",
    "heart attack": "chest pain",
    "substernal": "chest pain",
    "angina": "chest pain",
    "palpitations": "chest pain",
    "tearing pain": "chest pain",
    "radiating to back": "chest pain",
    # Abdominal
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "nausea": "abdominal pain",
    "vomiting": "abdominal pain",
    "diarrhea": "abdominal pain",
    "constipation": "abdominal pain",
    "epigastric": "abdominal pain",
    "RLQ": "abdominal pain",
    "RUQ": "abdominal pain",
    "peritonitis": "abdominal pain",
    # Headache
    "migraine": "headache",
    "head pain": "headache",
    "worst headache": "headache",
    "thunderclap": "headache",
    # Neurological
    "dizziness": "neurological emergency",
    "vertigo": "neurological emergency",
    "weakness": "neurological emergency",
    "paralysis": "neurological emergency",
    "numbness": "neurological emergency",
    "tingling": "neurological emergency",
    "diplopia": "neurological emergency",
    "slurred speech": "altered mental status",
    "facial droop": "altered mental status",
    "confused": "altered mental status",
    "confusion": "altered mental status",
    "unresponsive": "altered mental status",
    "drowsy": "altered mental status",
    "syncope": "altered mental status",
    "fainted": "altered mental status",
    "passed out": "altered mental status",
    "seizure": "altered mental status",
    "unconscious": "altered mental status",
    "fall": "altered mental status",
    # Fever / Infection
    "high temperature": "fever",
    "chills": "fever",
    "infection": "fever",
    "septic": "fever",
    "rigors": "fever",
    # Back
    "back pain": "back pain",
    "flank pain": "back pain",
    "sciatica": "back pain",
    "disc": "back pain",
    "lumbar": "back pain",
    "spine": "back pain",
    "radiculopathy": "back pain",
    # Skin / Wound
    "rash": "skin wound",
    "laceration": "skin wound",
    "wound": "skin wound",
    "cut": "skin wound",
    "abscess": "skin wound",
    "cellulitis": "skin wound",
    "swollen leg": "skin wound",
    "leg swelling": "skin wound",
    # Joint
    "knee pain": "joint pain",
    "hip pain": "joint pain",
    "shoulder pain": "joint pain",
    "ankle pain": "joint pain",
    "wrist pain": "joint pain",
    "elbow pain": "joint pain",
    "stiff": "joint pain",
    "arthritis": "joint pain",
    "gout": "joint pain",
    "swollen joint": "joint pain",
    # Pregnancy
    "pregnant": "pregnancy complication",
    "pregnancy": "pregnancy complication",
    "vaginal bleeding": "pregnancy complication",
    "contractions": "pregnancy complication",
    "eclampsia": "pregnancy complication",
    "preeclampsia": "pregnancy complication",
    # Trauma
    "car accident": "trauma",
    "MVC": "trauma",
    "motor vehicle": "trauma",
    "gunshot": "trauma",
    "stab wound": "trauma",
    "assault": "trauma",
    "fall from height": "trauma",
    "crush injury": "trauma",
    # Eye
    "eye pain": "eye symptoms",
    "vision loss": "eye symptoms",
    "blurry vision": "eye symptoms",
    "floaters": "eye symptoms",
    "flashing lights": "eye symptoms",
    "double vision": "eye symptoms",
    # Cardiac arrest
    "cardiac arrest": "cardiac arrest",
    "no pulse": "cardiac arrest",
    "CPR": "cardiac arrest",
    "code blue": "cardiac arrest",
    # Endocrine
    "diabetic": "endocrine crisis",
    "DKA": "endocrine crisis",
    "hyperglycemia": "endocrine crisis",
    "hypoglycemia": "endocrine crisis",
    "thyroid": "endocrine crisis",
    "adrenal": "endocrine crisis",
    # Psych
    "agitation": "psychiatric emergency",
    "hallucinations": "psychiatric emergency",
    "suicidal": "psychiatric emergency",
    "overdose": "altered mental status",
    # Pediatric
    "infant": "pediatric emergency",
    "neonate": "pediatric emergency",
    "child crying": "pediatric emergency",
    "stridor": "pediatric emergency",
    "drooling": "pediatric emergency",
    # Urinary
    "urinary": "fever",
    "dysuria": "fever",
    "hematuria": "back pain",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RECOMMENDED CONFIRMATORY TESTS by ICD-10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED_TESTS = {
    # Chest
    "I21.3": ["Troponin q3h", "12-lead ECG", "CXR", "Cath lab activation"],
    "I71.01": ["CT Aorta with contrast (STAT)", "Type & Screen", "BP both arms", "TEE"],
    "I31.4": ["Bedside Echo (STAT)", "Pericardiocentesis", "CBC", "BMP"],
    "I26.99": ["CT-PA", "D-dimer", "Lower extremity duplex", "ABG"],
    "J93.0": ["CXR (STAT)", "Needle decompression if tension", "Chest tube"],
    "I40.9": ["Troponin", "Echo", "Cardiac MRI", "Viral panel"],
    "I30.9": ["ECG", "Echo", "ESR/CRP", "Troponin"],
    # Respiratory
    "J44.1": ["ABG", "CXR", "CBC", "BMP", "Sputum culture"],
    "I50.9": ["BNP/NT-proBNP", "CXR", "Echo", "BMP"],
    "J18.9": ["CXR", "CBC", "Blood cultures", "Procalcitonin"],
    "J80": ["ABG", "CXR", "CT Chest", "Ventilator settings"],
    "T78.2": ["Tryptase level", "Observation 4-6h", "Allergy referral"],
    # Headache / Neuro
    "I60.9": [
        "CT Head non-contrast (STAT)",
        "LP if CT negative",
        "CTA Head/Neck",
        "Neurosurgery consult",
    ],
    "G03.9": ["LP (STAT)", "Blood cultures", "CBC", "CT Head first"],
    "S06.4X0A": ["CT Head (STAT)", "Neurosurgery consult", "Neuro checks q15min"],
    "C71.9": ["MRI Brain with contrast", "Neurosurgery consult", "Dexamethasone"],
    "G43.109": ["Clinical diagnosis", "Triptans PRN", "Outpatient neurology"],
    # Abdominal
    "K35.80": ["CT Abdomen/Pelvis with contrast", "CBC", "CRP", "Surgery consult"],
    "O00.10": [
        "Transvaginal ultrasound (STAT)",
        "Quantitative hCG",
        "Type & Screen",
        "OB/GYN STAT",
    ],
    "K63.1": [
        "Upright CXR/AXR",
        "CT Abdomen",
        "Surgery consult (STAT)",
        "Lactate",
        "Blood cultures",
    ],
    "K55.069": [
        "CT Angiography mesenteric",
        "Lactate (STAT)",
        "ABG",
        "Surgery/IR consult",
    ],
    "K85.9": ["Lipase (STAT)", "CT Abdomen", "CBC", "BMP", "LFTs"],
    "K56.60": ["CT Abdomen", "AXR", "Surgery consult", "NGT"],
    "I71.3": [
        "CT Abdomen (STAT)",
        "Type & Screen",
        "Massive transfusion",
        "Vascular surgery STAT",
    ],
    # Fever / Infection
    "A41.9": [
        "Blood cultures x2",
        "Lactate (STAT)",
        "CBC",
        "BMP",
        "Procalcitonin",
        "UA",
        "CXR",
    ],
    "G00.9": ["LP (STAT)", "Blood cultures", "CBC", "Empiric antibiotics before LP"],
    "I33.0": ["Blood cultures x3", "TTE/TEE", "CBC", "ESR/CRP"],
    "M72.6": [
        "Surgery consult (STAT)",
        "CBC",
        "Lactate",
        "CMP",
        "Blood cultures",
        "OR for debridement",
    ],
    "D70.9": [
        "Blood cultures",
        "CBC",
        "CMP",
        "UA",
        "CXR",
        "Empiric broad-spectrum antibiotics",
    ],
    # AMS
    "I63.9": [
        "CT Head (STAT)",
        "CTA Head/Neck",
        "tPA if < 4.5h",
        "Neurology consult",
        "CBC",
        "BMP",
        "Glucose",
    ],
    "I61.9": [
        "CT Head (STAT)",
        "Neurosurgery consult",
        "Reverse anticoagulation",
        "BP management",
    ],
    "E10.10": [
        "BMP (STAT)",
        "ABG/VBG",
        "CBC",
        "UA",
        "Insulin drip",
        "Fluid resuscitation",
    ],
    "G41.0": [
        "Glucose",
        "BMP",
        "Benzos (lorazepam/midazolam)",
        "EEG if refractory",
        "CT Head",
    ],
    "E87.1": [
        "BMP (STAT)",
        "Urine osmolality",
        "Urine sodium",
        "Hypertonic saline if severe",
    ],
    "T40.2X1A": ["Naloxone (STAT)", "Tox screen", "ABG", "Observation 4h minimum"],
    "E05.5": [
        "TSH/Free T4 (STAT)",
        "BMP",
        "Beta-blocker",
        "PTU/methimazole",
        "Hydrocortisone",
    ],
    # Back
    "G83.4": [
        "MRI Lumbar Spine (STAT)",
        "Neurosurgery consult",
        "Post-void residual",
        "Neuro exam",
    ],
    "M51.16": ["MRI Lumbar Spine", "X-ray L-spine", "Neurological exam", "NSAIDs"],
    "G06.1": [
        "MRI Spine with contrast (STAT)",
        "Blood cultures",
        "CBC",
        "ESR/CRP",
        "Neurosurgery consult",
    ],
    "M80.08XA": [
        "X-ray thoracolumbar",
        "CT if unclear",
        "DEXA scan outpatient",
        "Pain management",
    ],
    "N20.0": [
        "CT Abdomen/Pelvis non-contrast",
        "Urinalysis",
        "BMP",
        "Urology if > 10mm",
    ],
    # Skin
    "L03.90": [
        "CBC",
        "Blood cultures if febrile",
        "Mark borders",
        "IV antibiotics if severe",
    ],
    "T79.A0": [
        "Compartment pressures (STAT)",
        "Surgery consult",
        "Fasciotomy if > 30mmHg",
    ],
    "L51.1": [
        "Dermatology consult (STAT)",
        "Stop offending drug",
        "Burn unit if > 30% BSA",
        "Supportive care",
    ],
    "I82.40": ["Lower extremity duplex ultrasound", "D-dimer", "Anticoagulation"],
    # Joint
    "M00.9": [
        "Joint aspiration (STAT)",
        "Cell count & crystal",
        "Blood cultures",
        "CBC",
        "ESR/CRP",
    ],
    "M10.9": ["Serum uric acid", "Joint aspiration", "CBC", "BMP", "Colchicine/NSAIDs"],
    "M06.9": ["RF", "Anti-CCP", "ESR/CRP", "X-ray hands/feet", "Rheumatology consult"],
    "M86.9": ["MRI affected area", "Blood cultures", "CBC", "ESR/CRP", "Bone biopsy"],
    "M32.9": [
        "ANA",
        "dsDNA",
        "Complement C3/C4",
        "CBC",
        "Urinalysis",
        "Rheumatology consult",
    ],
    # Pregnancy
    "O15.0": [
        "Magnesium sulfate (STAT)",
        "CBC",
        "CMP",
        "LFTs",
        "Uric acid",
        "OB delivery",
    ],
    "O45.9": ["Type & Screen", "CBC", "Fibrinogen", "Fetal monitoring", "OB STAT"],
    "O14.2": ["CBC with smear", "LFTs", "LDH", "Haptoglobin", "Delivery planning"],
    # Trauma
    "S06.9X0A": [
        "CT Head (STAT)",
        "C-spine CT",
        "Neuro checks",
        "Neurosurgery consult",
    ],
    "S36.09XA": [
        "FAST exam",
        "CT Abdomen with contrast",
        "Type & Screen",
        "Surgery consult",
    ],
    "S22.5XXA": ["CXR", "CT Chest", "ABG", "Pain management", "Consider intubation"],
    "S27.1XXA": ["CXR", "Chest tube (STAT)", "CBC", "Type & Screen", "Surgery consult"],
    "S32.9XXA": [
        "Pelvic X-ray",
        "CT Pelvis",
        "Type & Screen",
        "Pelvic binder",
        "Massive transfusion protocol",
    ],
    # Cardiac arrest
    "I49.01": [
        "Defibrillation (STAT)",
        "Epinephrine",
        "Amiodarone",
        "Post-ROSC 12-lead ECG",
    ],
    "I26.02": [
        "tPA if massive PE suspected",
        "Echo",
        "ECMO if available",
        "CT-PA post-ROSC",
    ],
    # Eye
    "H40.20X0": [
        "IOP measurement (STAT)",
        "Timolol drops",
        "Pilocarpine",
        "Ophthalmology consult",
    ],
    "H33.00": [
        "Dilated fundoscopy",
        "Ophthalmology consult (STAT)",
        "B-scan ultrasound",
    ],
    "H34.10": [
        "ESR (STAT) to rule out GCA",
        "Ophthalmology STAT",
        "Carotid duplex",
        "Echo",
    ],
    # Endocrine
    "E27.2": [
        "Cortisol level (STAT)",
        "ACTH",
        "BMP",
        "IV hydrocortisone 100mg",
        "Fluid resuscitation",
    ],
    "D35.00": [
        "24h urine metanephrines",
        "Plasma metanephrines",
        "CT/MRI Abdomen",
        "Alpha-blocker first",
    ],
    "E03.5": [
        "TSH/Free T4",
        "Cortisol",
        "BMP",
        "IV levothyroxine",
        "Stress-dose steroids",
    ],
    # Psych
    "T43.205A": [
        "Cyproheptadine",
        "Cooling measures",
        "Benzos for agitation",
        "CK",
        "BMP",
    ],
    "G21.0": [
        "Stop offending agent",
        "Dantrolene/bromocriptine",
        "CK (STAT)",
        "Cooling",
        "ICU admission",
    ],
    # Neuro
    "G61.0": [
        "LP (albuminocytologic dissociation)",
        "NCS/EMG",
        "FVC q4h",
        "IVIG or plasmapheresis",
    ],
    "G95.20": [
        "MRI Spine (STAT)",
        "Dexamethasone",
        "Oncology/neurosurgery consult",
        "Radiation",
    ],
    "G70.01": [
        "FVC/NIF (STAT)",
        "AChR antibodies",
        "Pyridostigmine",
        "IVIG/plasmapheresis",
        "ICU if FVC < 20mL/kg",
    ],
    # Pediatric
    "K56.1": [
        "Ultrasound abdomen (STAT)",
        "Air enema reduction",
        "Surgery consult if failed",
    ],
    "M30.3": [
        "Echo",
        "CBC",
        "ESR/CRP",
        "LFTs",
        "IVIG + Aspirin",
        "Cardiology follow-up",
    ],
    "J05.1": [
        "Lateral neck X-ray",
        "Do NOT examine throat",
        "Anesthesia/ENT STAT",
        "Racemic epinephrine",
    ],
}


class DiagnosticAgent:
    """Generates ranked differential diagnoses from patient context.

    Matches symptoms from the clinical note against a medical knowledge
    base covering 80+ conditions across 20 categories. Scores
    probabilities using vital-sign signals, symptom overlap, and
    clinical acuity to produce a ranked differential.
    """

    def __init__(self, summarizer=None):
        self._summarizer = summarizer

    def diagnose(self, ctx: PatientContext, triage: TriageResult) -> DiagnosticResult:
        logger.info("Diagnostic assessment for patient %s", ctx.patient_id)

        note_lower = ctx.clinical_note.lower()
        complaint_lower = ctx.chief_complaint.lower()
        combined_text = f"{complaint_lower} {note_lower}"

        # Resolve symptom aliases to canonical category names
        resolved_categories: set[str] = set()
        for alias, canonical in SYMPTOM_ALIASES.items():
            if alias in combined_text:
                resolved_categories.add(canonical)

        # Determine which categories match the chief complaint directly
        complaint_categories: set[str] = set()
        for category in CLINICAL_KNOWLEDGE:
            if category in complaint_lower:
                complaint_categories.add(category)
        for alias, canonical in SYMPTOM_ALIASES.items():
            if alias in complaint_lower:
                complaint_categories.add(canonical)

        # Collect candidate diagnoses from matching categories
        candidates: list[Diagnosis] = []
        matched_categories = set()
        seen_conditions: set[str] = set()  # deduplicate
        for category, diagnoses in CLINICAL_KNOWLEDGE.items():
            if (
                category in complaint_lower
                or category in note_lower
                or category in resolved_categories
            ):
                matched_categories.add(category)
                # Chief complaint categories get a priority boost
                is_primary = category in complaint_categories
                for dx in diagnoses:
                    if dx.condition in seen_conditions:
                        continue
                    seen_conditions.add(dx.condition)
                    candidates.append(
                        Diagnosis(
                            condition=dx.condition,
                            icd10_code=dx.icd10_code,
                            # Primary category gets 0.15 head start
                            probability=0.15 if is_primary else 0.0,
                            supporting_evidence=list(dx.supporting_evidence),
                            ruling_out=list(dx.ruling_out),
                        )
                    )

        if not candidates:
            # Fallback: keyword match
            for category in CLINICAL_KNOWLEDGE:
                words = category.split()
                if any(w in complaint_lower for w in words):
                    for dx in CLINICAL_KNOWLEDGE[category]:
                        if dx.condition not in seen_conditions:
                            seen_conditions.add(dx.condition)
                            candidates.append(
                                Diagnosis(
                                    condition=dx.condition,
                                    icd10_code=dx.icd10_code,
                                    probability=0.15,
                                    supporting_evidence=list(dx.supporting_evidence),
                                    ruling_out=list(dx.ruling_out),
                                )
                            )
                    break

        # Score each candidate using evidence + vitals + acuity
        reasoning_chain = []
        for dx in candidates:
            evidence_found = []
            ruling_out_found = []
            for ev in dx.supporting_evidence:
                if ev.lower() in note_lower:
                    evidence_found.append(ev)
            for ro in dx.ruling_out:
                if ro.lower() in note_lower:
                    ruling_out_found.append(ro)

            evidence_score = len(evidence_found) / max(len(dx.supporting_evidence), 1)
            ruling_penalty = len(ruling_out_found) * 0.3
            acuity_boost = 0.1 if triage.esi_level.value <= 2 else 0.0
            age_factor = 0.05 if ctx.age > 60 else 0.0

            # Vital sign severity boost
            vitals_boost = 0.0
            v = ctx.vitals
            if v.get("heart_rate", 80) > 120:
                vitals_boost += 0.05
            if v.get("oxygen_saturation", 98) < 92:
                vitals_boost += 0.05
            if v.get("systolic_bp", 120) < 90:
                vitals_boost += 0.05
            if v.get("body_temperature", 37) > 39:
                vitals_boost += 0.03

            complaint_boost = dx.probability  # head start from primary
            dx.probability = round(
                min(
                    max(
                        complaint_boost
                        + evidence_score
                        - ruling_penalty
                        + acuity_boost
                        + age_factor
                        + vitals_boost,
                        0.05,
                    ),
                    0.95,
                ),
                3,
            )
            dx.supporting_evidence = evidence_found or dx.supporting_evidence[:2]

            reasoning_chain.append(
                f"{dx.condition} ({dx.icd10_code}): "
                f"evidence={len(evidence_found)}/{len(dx.supporting_evidence)}"
                f", rule-outs={len(ruling_out_found)}"
                f", P={dx.probability}"
            )

        # Sort by probability descending
        candidates.sort(key=lambda d: d.probability, reverse=True)

        # Collect recommended tests from top 3
        tests = []
        seen_tests: set[str] = set()
        for dx in candidates[:3]:
            for test in RECOMMENDED_TESTS.get(dx.icd10_code, []):
                if test not in seen_tests:
                    tests.append(test)
                    seen_tests.add(test)

        primary = candidates[0] if candidates else None

        return DiagnosticResult(
            differentials=candidates[:5],
            primary_diagnosis=(primary.condition if primary else "Undifferentiated"),
            confidence=primary.probability if primary else 0.1,
            reasoning_chain=reasoning_chain,
            recommended_tests=tests[:10],
        )
