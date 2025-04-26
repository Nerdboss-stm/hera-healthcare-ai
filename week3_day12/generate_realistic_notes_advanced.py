import pandas as pd
import random
import os

# Create output folder
os.makedirs('week3_day12', exist_ok=True)

# Vocabulary pools
genders = ["male", "female"]
chief_complaints = [
    "chest pain", "shortness of breath", "severe headache",
    "abdominal pain", "persistent cough", "fever", "extreme fatigue"
]
histories = [
    "Patient experienced {symptom} starting {days} days ago after {trigger}.",
    "Reports worsening {symptom} when {activity}.",
    "Symptoms have been {progression} over the past {days} days."
]
review_systems = [
    "Denies fever, chills, or nausea.",
    "Positive for {symptom}. No recent infections.",
    "Denies dizziness but reports mild {symptom}."
]
physical_exams = [
    "Vital signs stable, BP {bp}, HR {hr}.",
    "Lungs clear to auscultation. Cardiac exam shows {finding}.",
    "Mild tenderness noted over {area}."
]
lab_findings = [
    "Troponin elevated at {troponin} ng/mL.",
    "CBC shows WBC {wbc} x10^3/uL.",
    "ABG reveals pH {ph_level}."
]
imaging_findings = [
    "Chest X-ray unremarkable.",
    "CT scan reveals no acute abnormalities.",
    "MRI shows small area of ischemia."
]
assessments = [
    "Likely diagnosis: {diagnosis}.",
    "Differential includes {diagnosis} and {alt_diagnosis}."
]
plans = [
    "Plan includes {tests}, monitor vitals, and start {treatment}.",
    "Admit to telemetry unit. Start {treatment} immediately."
]

diagnoses = [
    "acute coronary syndrome", "pneumonia", "stroke",
    "diabetic ketoacidosis", "pulmonary embolism"
]
tests = [
    "serial troponins", "chest X-ray", "brain MRI",
    "arterial blood gas analysis", "CT angiogram"
]
treatments = [
    "anticoagulation therapy", "IV antibiotics",
    "insulin drip", "dual antiplatelet therapy"
]

# Random typo generator
def introduce_typo(text):
    if random.random() < 0.1:  # 10% chance
        typo_positions = random.sample(range(len(text)), k=2)
        typo_text = list(text)
        for pos in typo_positions:
            typo_text[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return "".join(typo_text)
    return text

# Single clinical note generator
def generate_note():
    gender = random.choice(genders)
    age = random.randint(20, 90)
    symptom = random.choice(chief_complaints)
    diagnosis = random.choice(diagnoses)
    alt_diagnosis = random.choice([d for d in diagnoses if d != diagnosis])
    days = random.choice(["1", "2", "3", "5"])
    trigger = random.choice(["physical exertion", "emotional stress", "heavy meals"])
    activity = random.choice(["walking", "climbing stairs", "talking"])
    progression = random.choice(["stable", "worsening", "improving"])
    bp = f"{random.randint(110, 160)}/{random.randint(70, 100)}"
    hr = random.randint(60, 120)
    finding = random.choice(["regular rhythm", "tachycardia", "bradycardia"])
    area = random.choice(["abdomen", "chest wall", "lower limbs"])
    troponin = round(random.uniform(0.02, 1.5), 2)
    wbc = random.randint(4, 18)
    ph_level = round(random.uniform(7.1, 7.5), 2)
    test_plan = random.choice(tests)
    treatment_plan = random.choice(treatments)

    sections = []

    # Chief Complaint
    sections.append(f"Chief Complaint: {symptom.capitalize()}.")

    # History of Present Illness
    sections.extend([
        f"The patient is a {age}-year-old {gender} presenting with {symptom}.",
        histories[0].format(symptom=symptom, days=days, trigger=trigger),
        histories[1].format(symptom=symptom, activity=activity),
        histories[2].format(symptom=symptom, days=days, progression=progression),
    ])

    # Review of Systems
    sections.extend([
        review_systems[0],
        review_systems[1].format(symptom=symptom),
        review_systems[2].format(symptom=symptom),
    ])

    # Vitals and Physical Exam
    sections.extend([
        physical_exams[0].format(bp=bp, hr=hr),
        physical_exams[1].format(finding=finding),
        physical_exams[2].format(area=area)
    ])

    # Labs
    sections.extend([
        lab_findings[0].format(troponin=troponin),
        lab_findings[1].format(wbc=wbc),
        lab_findings[2].format(ph_level=ph_level)
    ])

    # Imaging
    sections.extend([
        imaging_findings[0],
        imaging_findings[1],
        imaging_findings[2]
    ])

    # Assessment and Plan
    sections.extend([
        assessments[0].format(diagnosis=diagnosis),
        assessments[1].format(diagnosis=diagnosis, alt_diagnosis=alt_diagnosis),
        plans[0].format(tests=test_plan, treatment=treatment_plan),
        plans[1].format(treatment=treatment_plan)
    ])

    # Introduce missing sections randomly
    if random.random() < 0.1:
        drop = random.choice([2, 3, 4])  # drop Review of Systems, Physical Exam, Labs randomly
        del sections[drop*3 : drop*3 + 3]  # drop 3 sentences together

    # Introduce typos randomly
    sections = [introduce_typo(sent) for sent in sections]

    # Join sections
    full_note = " ".join(sections)

    # Target Summary
    summary = f"{diagnosis.capitalize()} suspected. {treatment_plan.capitalize()} started. Monitoring with {test_plan.lower()}."

    return full_note, summary

# Batch generator
def generate_dataset(num_samples, output_file):
    notes = []
    summaries = []

    for _ in range(num_samples):
        note, summary = generate_note()
        notes.append(note)
        summaries.append(summary)

    df = pd.DataFrame({"original_note": notes, "target_summary": summaries})
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {num_samples} samples to {output_file}")

# Generate both datasets
generate_dataset(1000, 'week3_day12/notes_1000.csv')
generate_dataset(5000, 'week3_day12/notes_5000.csv')

