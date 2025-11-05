"""Entity Extractor - Extracts entities and relationships from text"""

import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class EntityExtractor:
    """Extracts entities and relationships from text using pattern matching and NLP"""

    def __init__(self, use_advanced_nlp: bool = False):
        """
        Initialize entity extractor

        Args:
            use_advanced_nlp: Whether to use spaCy for advanced NLP (requires model download)
        """
        self.use_advanced_nlp = use_advanced_nlp
        self.nlp = None

        if use_advanced_nlp:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("SpaCy model not found. Using basic extraction.")
                    self.use_advanced_nlp = False
            except ImportError:
                print("SpaCy not installed. Using basic extraction.")
                self.use_advanced_nlp = False

    def extract_entities_and_relationships(self, text: str,
                                          document_id: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text

        Args:
            text: Input text to analyze
            document_id: Optional document identifier

        Returns:
            Tuple of (entities, relationships)
        """
        if self.use_advanced_nlp and self.nlp:
            return self._extract_with_spacy(text, document_id)
        else:
            return self._extract_with_patterns(text, document_id)

    def _extract_with_spacy(self, text: str, document_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract using spaCy NLP"""
        doc = self.nlp(text)
        entities = []
        relationships = []

        # Extract named entities
        entity_map = {}
        for ent in doc.ents:
            entity_id = self._normalize_entity(ent.text)
            if entity_id not in entity_map:
                entity = {
                    "id": entity_id,
                    "type": ent.label_,
                    "text": ent.text,
                    "source_document": document_id
                }
                entities.append(entity)
                entity_map[entity_id] = entity

        # Extract relationships using dependency parsing
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr']:
                subject = token.head

                # Find subject entity
                subject_entity = None
                for ent in doc.ents:
                    if ent.start <= subject.i < ent.end:
                        subject_entity = self._normalize_entity(ent.text)
                        break

                # Find object entity
                object_entity = None
                for ent in doc.ents:
                    if ent.start <= token.i < ent.end:
                        object_entity = self._normalize_entity(ent.text)
                        break

                if subject_entity and object_entity and subject_entity != object_entity:
                    relationships.append({
                        "source": subject_entity,
                        "target": object_entity,
                        "type": subject.lemma_,
                        "context": text[max(0, token.sent.start_char):min(len(text), token.sent.end_char)]
                    })

        return entities, relationships

    def _extract_with_patterns(self, text: str, document_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract using simple pattern matching"""
        entities = []
        relationships = []
        entity_map = {}

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Pattern: "X is a Y" or "X is Y"
            is_pattern = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an)?\s*([a-z]+(?:\s+[a-z]+)*)',
                                   sentence)
            for match in is_pattern:
                entity_id = self._normalize_entity(match[0])
                entity_type = match[1].strip()

                if entity_id not in entity_map:
                    entities.append({
                        "id": entity_id,
                        "type": "ENTITY",
                        "text": match[0],
                        "source_document": document_id,
                        "category": entity_type
                    })
                    entity_map[entity_id] = True

            # Pattern: "X VERB Y" (e.g., "John works at Google")
            verb_pattern = re.findall(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+([a-z]+(?:\s+[a-z]+)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                sentence
            )

            for match in verb_pattern:
                source = self._normalize_entity(match[0])
                relation = match[1].strip()
                target = self._normalize_entity(match[2])

                # Add entities if not exist
                for entity_text, entity_id in [(match[0], source), (match[2], target)]:
                    if entity_id not in entity_map:
                        entities.append({
                            "id": entity_id,
                            "type": "ENTITY",
                            "text": entity_text,
                            "source_document": document_id
                        })
                        entity_map[entity_id] = True

                # Add relationship
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": relation,
                    "context": sentence
                })

            # Pattern: "X's Y" (possessive relationships)
            possessive_pattern = re.findall(
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+([a-z]+)",
                sentence
            )

            for match in possessive_pattern:
                source = self._normalize_entity(match[0])
                relation = "has_" + match[1]

                if source not in entity_map:
                    entities.append({
                        "id": source,
                        "type": "ENTITY",
                        "text": match[0],
                        "source_document": document_id
                    })
                    entity_map[source] = True

        # Extract capitalized words as potential entities
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for cap_text in set(capitalized):
            entity_id = self._normalize_entity(cap_text)
            if entity_id not in entity_map:
                entities.append({
                    "id": entity_id,
                    "type": "ENTITY",
                    "text": cap_text,
                    "source_document": document_id
                })
                entity_map[entity_id] = True

        return entities, relationships

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text to create consistent IDs"""
        # Remove extra whitespace and convert to lowercase for ID
        normalized = re.sub(r'\s+', '_', text.strip())
        return normalized.lower()

    def extract_from_structured_data(self, data: Dict[str, Any],
                                     id_field: str = "id") -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from structured data

        Args:
            data: Dictionary or list of dictionaries
            id_field: Field to use as entity ID

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        if isinstance(data, dict):
            data = [data]

        for item in data:
            if not isinstance(item, dict):
                continue

            entity_id = item.get(id_field, str(id(item)))

            # Create entity
            entity = {
                "id": str(entity_id),
                "type": item.get("type", "ENTITY"),
                "properties": {}
            }

            # Extract properties and nested relationships
            for key, value in item.items():
                if key == id_field or key == "type":
                    continue

                if isinstance(value, (str, int, float, bool)):
                    entity["properties"][key] = value
                elif isinstance(value, dict):
                    # Nested object becomes a relationship
                    nested_id = value.get(id_field, f"{entity_id}_{key}")
                    nested_entity = {
                        "id": str(nested_id),
                        "type": key.upper(),
                        "properties": {k: v for k, v in value.items()
                                     if isinstance(v, (str, int, float, bool))}
                    }
                    entities.append(nested_entity)

                    relationships.append({
                        "source": str(entity_id),
                        "target": str(nested_id),
                        "type": f"has_{key}"
                    })
                elif isinstance(value, list):
                    # List of related entities
                    for idx, list_item in enumerate(value):
                        if isinstance(list_item, dict):
                            list_id = list_item.get(id_field, f"{entity_id}_{key}_{idx}")
                            list_entity = {
                                "id": str(list_id),
                                "type": key.upper(),
                                "properties": {k: v for k, v in list_item.items()
                                             if isinstance(v, (str, int, float, bool))}
                            }
                            entities.append(list_entity)

                            relationships.append({
                                "source": str(entity_id),
                                "target": str(list_id),
                                "type": f"has_{key}"
                            })

            entities.append(entity)

        return entities, relationships
