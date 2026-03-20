"""Patient Knowledge Graph — Entity relationship graph from clinical notes.

Builds a NetworkX graph where nodes are medical entities (medications,
conditions, procedures) and edges represent clinical relationships
(treats, diagnoses, contraindicated_with, etc.).
"""

from __future__ import annotations
import logging

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ner.extractor import ClinicalNERExtractor, ExtractionResult, ClinicalEntity

logger = logging.getLogger(__name__)

# Predefined clinical relationships
TREATMENT_RELATIONSHIPS = {
    ("aspirin", "myocardial infarction"): "treats",
    ("aspirin", "acute coronary syndrome"): "treats",
    ("heparin", "pulmonary embolism"): "treats",
    ("heparin", "dvt"): "treats",
    ("ceftriaxone", "pneumonia"): "treats",
    ("azithromycin", "pneumonia"): "treats",
    ("furosemide", "heart failure"): "treats",
    ("albuterol", "copd"): "treats",
    ("albuterol", "asthma"): "treats",
    ("insulin", "diabetes"): "treats",
    ("metformin", "diabetes"): "treats",
    ("vancomycin", "sepsis"): "treats",
    ("alteplase", "stroke"): "treats",
    ("prednisone", "copd"): "treats",
    ("nitroglycerin", "myocardial infarction"): "treats",
    ("lisinopril", "hypertension"): "treats",
    ("lisinopril", "heart failure"): "treats",
    ("metoprolol", "hypertension"): "treats",
    ("metoprolol", "atrial fibrillation"): "treats",
    ("warfarin", "atrial fibrillation"): "treats",
    ("enoxaparin", "pulmonary embolism"): "treats",
}

CONTRAINDICATION_RELATIONSHIPS = {
    ("nitroglycerin", "hypotension"): "contraindicated_with",
    ("metformin", "chronic kidney disease"): "caution_with",
    ("warfarin", "aspirin"): "interaction_risk",
    ("heparin", "enoxaparin"): "avoid_combination",
    ("metoprolol", "asthma"): "caution_with",
}

DIAGNOSTIC_RELATIONSHIPS = {
    ("ecg", "myocardial infarction"): "diagnoses",
    ("ecg", "atrial fibrillation"): "diagnoses",
    ("ct scan", "stroke"): "diagnoses",
    ("ct scan", "pulmonary embolism"): "diagnoses",
    ("ct scan", "appendicitis"): "diagnoses",
    ("x-ray", "pneumonia"): "diagnoses",
    ("ultrasound", "cholecystitis"): "diagnoses",
    ("lumbar puncture", "meningitis"): "diagnoses",
    ("echocardiogram", "heart failure"): "diagnoses",
}


class PatientKnowledgeGraph:
    """Builds and queries a patient-specific knowledge graph.

    Nodes represent clinical entities (medications, conditions, procedures).
    Edges represent relationships (treats, diagnoses, contraindicated_with).
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for PatientKnowledgeGraph")
        self._graph = nx.DiGraph()
        self._extractor = ClinicalNERExtractor()

    def build_from_note(self, clinical_note: str, patient_id: str = "unknown") -> dict:
        """Extract entities from a clinical note and build the knowledge graph."""
        extraction = self._extractor.extract(clinical_note)

        # Add entity nodes
        for entity in extraction.entities:
            self._add_entity_node(entity)

        # Add relationship edges
        self._add_relationships(extraction)

        return {
            "patient_id": patient_id,
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "entities": extraction.to_dict(),
        }

    def _add_entity_node(self, entity: ClinicalEntity) -> None:
        node_id = entity.text.lower()
        self._graph.add_node(
            node_id,
            label=entity.label,
            text=entity.text,
            code=entity.code,
            normalized=entity.normalized,
        )

    def _add_relationships(self, extraction: ExtractionResult) -> None:
        all_entities = {e.text.lower(): e for e in extraction.entities}

        # Treatment relationships
        for (med, condition), rel_type in TREATMENT_RELATIONSHIPS.items():
            if med in all_entities and condition in all_entities:
                self._graph.add_edge(med, condition, relationship=rel_type)

        # Contraindication relationships
        for (a, b), rel_type in CONTRAINDICATION_RELATIONSHIPS.items():
            if a in all_entities and b in all_entities:
                self._graph.add_edge(a, b, relationship=rel_type)

        # Diagnostic relationships
        for (proc, condition), rel_type in DIAGNOSTIC_RELATIONSHIPS.items():
            if proc in all_entities and condition in all_entities:
                self._graph.add_edge(proc, condition, relationship=rel_type)

        # Infer co-occurrence relationships for entities not in predefined maps
        meds = [e.text.lower() for e in extraction.medications]
        conditions = [e.text.lower() for e in extraction.conditions]
        procedures = [e.text.lower() for e in extraction.procedures]

        for med in meds:
            for cond in conditions:
                if not self._graph.has_edge(med, cond):
                    self._graph.add_edge(med, cond, relationship="co_mentioned")

        for proc in procedures:
            for cond in conditions:
                if not self._graph.has_edge(proc, cond):
                    self._graph.add_edge(proc, cond, relationship="co_mentioned")

    def query_entity(self, entity_name: str) -> dict:
        """Query all relationships for a given entity."""
        node = entity_name.lower()
        if node not in self._graph:
            return {"entity": entity_name, "found": False, "relationships": []}

        relationships = []
        for _, target, data in self._graph.out_edges(node, data=True):
            relationships.append(
                {
                    "target": target,
                    "relationship": data.get("relationship", "unknown"),
                    "target_label": self._graph.nodes[target].get("label", "unknown"),
                }
            )
        for source, _, data in self._graph.in_edges(node, data=True):
            relationships.append(
                {
                    "source": source,
                    "relationship": data.get("relationship", "unknown"),
                    "source_label": self._graph.nodes[source].get("label", "unknown"),
                }
            )

        return {
            "entity": entity_name,
            "found": True,
            "node_data": dict(self._graph.nodes[node]),
            "relationships": relationships,
        }

    def get_interactions(self, medication: str) -> list[dict]:
        """Find all drug interactions for a given medication."""
        node = medication.lower()
        interactions = []
        for _, target, data in self._graph.out_edges(node, data=True):
            rel = data.get("relationship", "")
            if rel in (
                "contraindicated_with",
                "caution_with",
                "interaction_risk",
                "avoid_combination",
            ):
                interactions.append(
                    {
                        "medication": medication,
                        "interacts_with": target,
                        "severity": rel,
                    }
                )
        return interactions

    def get_treatment_chain(self, condition: str) -> dict:
        """Get the full treatment chain for a condition."""
        node = condition.lower()
        if node not in self._graph:
            return {"condition": condition, "found": False}

        treating_meds = []
        diagnosing_procs = []

        for source, _, data in self._graph.in_edges(node, data=True):
            rel = data.get("relationship", "")
            if rel == "treats":
                treating_meds.append(source)
            elif rel == "diagnoses":
                diagnosing_procs.append(source)

        return {
            "condition": condition,
            "found": True,
            "treating_medications": treating_meds,
            "diagnostic_procedures": diagnosing_procs,
        }

    def to_dict(self) -> dict:
        """Export graph as a serializable dictionary."""
        nodes = []
        for node_id, data in self._graph.nodes(data=True):
            nodes.append({"id": node_id, **data})

        edges = []
        for source, target, data in self._graph.edges(data=True):
            edges.append({"source": source, "target": target, **data})

        return {"nodes": nodes, "edges": edges}

    def to_cytoscape(self) -> dict:
        """Export in Cytoscape.js format for frontend visualization."""
        elements = []
        for node_id, data in self._graph.nodes(data=True):
            elements.append(
                {
                    "data": {"id": node_id, **data},
                    "group": "nodes",
                }
            )
        for source, target, data in self._graph.edges(data=True):
            elements.append(
                {
                    "data": {"source": source, "target": target, **data},
                    "group": "edges",
                }
            )
        return {"elements": elements}
