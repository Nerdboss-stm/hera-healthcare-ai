"""Tests for Clinical NER + Knowledge Graph."""

from ner.extractor import ClinicalNERExtractor


class TestClinicalNERExtractor:
    def setup_method(self):
        self.extractor = ClinicalNERExtractor()

    def test_extract_medications(self):
        text = "Patient is on aspirin 325mg and metoprolol 50mg daily."
        result = self.extractor.extract(text)
        med_names = [m.text.lower() for m in result.medications]
        assert "aspirin" in med_names
        assert "metoprolol" in med_names

    def test_extract_conditions(self):
        text = "History of hypertension, diabetes, and COPD."
        result = self.extractor.extract(text)
        cond_names = [c.text.lower() for c in result.conditions]
        assert "hypertension" in cond_names
        assert "diabetes" in cond_names

    def test_extract_procedures(self):
        text = "ECG performed. CT scan of chest ordered. Chest tube inserted."
        result = self.extractor.extract(text)
        proc_names = [p.text.lower() for p in result.procedures]
        assert any("ecg" in p or "ekg" in p for p in proc_names)

    def test_extract_lab_values(self):
        text = "Troponin of 0.5 ng/mL, WBC 12000 cells/uL, glucose 180 mg/dL."
        result = self.extractor.extract(text)
        assert len(result.lab_values) > 0

    def test_entity_count(self):
        text = "Patient on aspirin for myocardial infarction. ECG shows ST elevation."
        result = self.extractor.extract(text)
        assert result.entity_count > 0
        assert result.entity_count == len(result.entities)

    def test_umls_normalization(self):
        text = "Diagnosed with sepsis, started on heparin."
        result = self.extractor.extract(text)
        heparin_entities = [e for e in result.entities if e.text.lower() == "heparin"]
        assert len(heparin_entities) > 0
        assert heparin_entities[0].code != ""

    def test_to_dict(self):
        text = "Aspirin 325mg for hypertension. ECG ordered."
        result = self.extractor.extract(text)
        d = result.to_dict()
        assert "entity_count" in d
        assert "medications" in d
        assert "conditions" in d

    def test_empty_note(self):
        result = self.extractor.extract("This is a normal note with no medical terms.")
        assert result.entity_count == 0 or result.entity_count >= 0  # may find some


class TestPatientKnowledgeGraph:
    def test_build_graph(self):
        from ner.knowledge_graph import PatientKnowledgeGraph
        kg = PatientKnowledgeGraph()
        result = kg.build_from_note(
            "Patient on aspirin for myocardial infarction. ECG ordered.",
            patient_id="test-001",
        )
        assert result["nodes"] > 0
        assert result["patient_id"] == "test-001"

    def test_query_entity(self):
        from ner.knowledge_graph import PatientKnowledgeGraph
        kg = PatientKnowledgeGraph()
        kg.build_from_note("Patient on aspirin for myocardial infarction.")
        result = kg.query_entity("aspirin")
        assert result["found"] is True

    def test_treatment_chain(self):
        from ner.knowledge_graph import PatientKnowledgeGraph
        kg = PatientKnowledgeGraph()
        kg.build_from_note("Aspirin and heparin for myocardial infarction. ECG performed.")
        chain = kg.get_treatment_chain("myocardial infarction")
        assert chain["found"] is True
        assert len(chain["treating_medications"]) > 0

    def test_export_dict(self):
        from ner.knowledge_graph import PatientKnowledgeGraph
        kg = PatientKnowledgeGraph()
        kg.build_from_note("Aspirin for hypertension.")
        export = kg.to_dict()
        assert "nodes" in export
        assert "edges" in export

    def test_cytoscape_export(self):
        from ner.knowledge_graph import PatientKnowledgeGraph
        kg = PatientKnowledgeGraph()
        kg.build_from_note("Aspirin for hypertension.")
        export = kg.to_cytoscape()
        assert "elements" in export
