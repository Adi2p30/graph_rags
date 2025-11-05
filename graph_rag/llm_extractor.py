"""LLM-based Entity and Relationship Extractor using Gemini 2.5 Flash"""

import os
from typing import List, Dict, Any, Tuple, Optional
import json
import re


class GeminiExtractor:
    """Extract entities and relationships using Gemini 2.5 Flash"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini extractor

        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
        """
        try:
            import google.generativeai as genai

            self.genai = genai
            self.api_key = api_key or os.getenv('GOOGLE_API_KEY')

            if not self.api_key:
                print("Warning: No Gemini API key provided. Set GOOGLE_API_KEY environment variable.")
                self.model = None
            else:
                genai.configure(api_key=self.api_key)
                # Use Gemini 2.5 Flash for fast, efficient extraction
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

        except ImportError:
            print("Google Generative AI library not installed. Install with: pip install google-generativeai")
            self.genai = None
            self.model = None

    def extract_entities_and_relationships(
        self,
        text: str,
        document_id: str = None,
        entity_types: List[str] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text using Gemini

        Args:
            text: Input text
            document_id: Optional document identifier
            entity_types: Optional list of entity types to extract

        Returns:
            Tuple of (entities, relationships)
        """
        if not self.model:
            print("Gemini model not initialized. Falling back to pattern extraction.")
            return self._fallback_extraction(text, document_id)

        try:
            # Construct prompt for entity and relationship extraction
            prompt = self._build_extraction_prompt(text, entity_types)

            # Generate with Gemini
            response = self.model.generate_content(prompt)

            # Parse the response
            entities, relationships = self._parse_gemini_response(
                response.text,
                document_id
            )

            return entities, relationships

        except Exception as e:
            print(f"Error extracting with Gemini: {e}")
            print("Falling back to pattern extraction.")
            return self._fallback_extraction(text, document_id)

    def _build_extraction_prompt(self, text: str, entity_types: List[str] = None) -> str:
        """Build prompt for Gemini extraction"""
        entity_types_str = ", ".join(entity_types) if entity_types else "PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, PRODUCT"

        prompt = f"""You are an expert knowledge graph extraction system. Extract entities and relationships from the following text.

ENTITY TYPES TO EXTRACT: {entity_types_str}

TEXT:
{text}

INSTRUCTIONS:
1. Identify all important entities in the text
2. Classify each entity into one of the provided types
3. Extract meaningful relationships between entities
4. For each entity, provide: id (lowercase with underscores), name (original text), type, and a brief description
5. For each relationship, provide: source entity id, target entity id, relationship type (uppercase with underscores), and optionally properties

OUTPUT FORMAT (JSON):
{{
  "entities": [
    {{
      "id": "entity_id",
      "name": "Entity Name",
      "type": "ENTITY_TYPE",
      "description": "Brief description"
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "description": "Brief description of the relationship",
      "properties": {{"key": "value"}}
    }}
  ]
}}

Extract comprehensive entities and relationships. Be thorough and capture all important information.

RESPOND WITH VALID JSON ONLY:"""

        return prompt

    def _parse_gemini_response(
        self,
        response_text: str,
        document_id: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Parse Gemini's JSON response"""
        try:
            # Extract JSON from response (may be wrapped in markdown)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_str = response_text.strip()

            data = json.loads(json_str)

            entities = []
            for entity in data.get('entities', []):
                entities.append({
                    'id': entity.get('id', entity.get('name', '').lower().replace(' ', '_')),
                    'type': entity.get('type', 'ENTITY'),
                    'name': entity.get('name', entity.get('id', '')),
                    'description': entity.get('description', ''),
                    'source_document': document_id
                })

            relationships = []
            for rel in data.get('relationships', []):
                relationships.append({
                    'source': rel.get('source'),
                    'target': rel.get('target'),
                    'type': rel.get('type', 'RELATED_TO'),
                    'description': rel.get('description', ''),
                    'properties': rel.get('properties', {}),
                    'source_document': document_id
                })

            return entities, relationships

        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini response as JSON: {e}")
            print(f"Response: {response_text[:500]}")
            return [], []

    def _fallback_extraction(self, text: str, document_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Fallback pattern-based extraction when Gemini is unavailable"""
        from .entity_extractor import EntityExtractor

        extractor = EntityExtractor(use_advanced_nlp=False)
        return extractor.extract_entities_and_relationships(text, document_id)

    def summarize_community(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a summary for a community of entities

        Args:
            entities: List of entities in the community
            relationships: List of relationships in the community

        Returns:
            Summary text
        """
        if not self.model:
            return self._simple_community_summary(entities, relationships)

        try:
            # Build context
            entity_descriptions = "\n".join([
                f"- {e.get('name', e.get('id'))}: {e.get('description', e.get('type', 'Entity'))}"
                for e in entities[:50]  # Limit to avoid token limits
            ])

            relationship_descriptions = "\n".join([
                f"- {r['source']} {r['type']} {r['target']}"
                for r in relationships[:50]
            ])

            prompt = f"""Summarize this community from a knowledge graph:

ENTITIES:
{entity_descriptions}

RELATIONSHIPS:
{relationship_descriptions}

Provide a concise 2-3 paragraph summary that captures:
1. The main theme or topic of this community
2. Key entities and their roles
3. Important relationships and connections
4. Overall significance

SUMMARY:"""

            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            print(f"Error generating community summary: {e}")
            return self._simple_community_summary(entities, relationships)

    def _simple_community_summary(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Simple fallback summary without LLM"""
        entity_types = {}
        for entity in entities:
            etype = entity.get('type', 'ENTITY')
            entity_types[etype] = entity_types.get(etype, 0) + 1

        rel_types = {}
        for rel in relationships:
            rtype = rel.get('type', 'RELATED_TO')
            rel_types[rtype] = rel_types.get(rtype, 0) + 1

        summary = f"This community contains {len(entities)} entities and {len(relationships)} relationships.\n\n"
        summary += "Entity types: " + ", ".join([f"{k} ({v})" for k, v in entity_types.items()]) + "\n\n"
        summary += "Relationship types: " + ", ".join([f"{k} ({v})" for k, v in rel_types.items()])

        return summary

    def answer_query(
        self,
        query: str,
        context_entities: List[Dict[str, Any]],
        context_relationships: List[Dict[str, Any]],
        community_summaries: List[str] = None
    ) -> str:
        """
        Answer a query using graph context

        Args:
            query: User query
            context_entities: Relevant entities
            context_relationships: Relevant relationships
            community_summaries: Optional community summaries

        Returns:
            Answer text
        """
        if not self.model:
            return "Gemini model not available for query answering."

        try:
            # Build context
            entities_context = "\n".join([
                f"- {e.get('name', e.get('id'))}: {e.get('description', '')}"
                for e in context_entities[:30]
            ])

            relationships_context = "\n".join([
                f"- {r['source']} {r['type']} {r['target']}"
                for r in context_relationships[:30]
            ])

            summaries_context = ""
            if community_summaries:
                summaries_context = "\n\nCOMMUNITY SUMMARIES:\n" + "\n\n".join(community_summaries[:5])

            prompt = f"""Answer the following query using the knowledge graph information provided.

QUERY: {query}

RELEVANT ENTITIES:
{entities_context}

RELEVANT RELATIONSHIPS:
{relationships_context}
{summaries_context}

Provide a comprehensive answer based on the graph information. If the information is insufficient, acknowledge what's missing.

ANSWER:"""

            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            print(f"Error answering query with Gemini: {e}")
            return f"Error generating answer: {str(e)}"
