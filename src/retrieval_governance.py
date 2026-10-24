"""Retrieval-Grounded Governance Judge (System B).

Implements a domain-agnostic governance pipeline:
1. Formulate retrieval query from scenario text + parameters.
2. Retrieve top-k policy chunks via FAISS dense similarity search.
3. Prompt an LLM judge with retrieved context to produce a structured
   APPROVE/FLAG/BLOCK decision citing the specific policy chunk.

CRITICAL DESIGN INVARIANT: Domain-specific behavior lives entirely in
the policy documents and domain name config, NEVER in per-domain code
branches. Adding a new domain requires only adding policy documents
and test scenarios — zero code changes.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.index import DomainIndex
from src.llm_client import LLMClient, LLMResponse
from src.config import TOP_K_RETRIEVAL

SYSTEM_PROMPT = """You are an expert autonomous Enterprise Governance Judge.
Your job is to evaluate proposed employee, financial, or IT actions against the provided official enterprise policy context.

You must output a strictly valid JSON object with the following schema:
{
  "decision": "APPROVE" | "FLAG" | "BLOCK",
  "citation_chunk_id": "<exact chunk_id of the most relevant retrieved policy chunk>",
  "citation_text": "<exact phrase or key sentence from the retrieved policy explaining the threshold/rule>",
  "reasoning": "<clear concise explanation including numeric comparison and policy grounding>"
}

Decision Guidelines:
1. "APPROVE": The request completely complies with all stated policy limits, entitlements, and requirements.
2. "FLAG": The request exceeds a routine threshold, requires special manager/VP/director escalation, has missing or pending documentation (e.g. receipt missing, medical certificate pending), or is submitted outside standard advance notice windows.
3. "BLOCK": The request directly violates a hard prohibition, exceeds an absolute ceiling, requests forbidden role combinations (Separation of Duties violation), or is strictly prohibited by policy.

Scalar & Arithmetic Rules:
- If evaluating numerical amounts, dates, or time windows:
  a. Locate the exact threshold in the retrieved policy chunk (e.g., "$50 limit", "5 business days notice", "8 hours break-glass").
  b. Perform an exact comparison with the requested value.
  c. If requested <= limit (compliant), mark APPROVE.
  d. If requested > routine limit or missing required receipt/approver, mark FLAG or BLOCK according to the policy passage.
- Always ground decisions in the retrieved text. Do not invent outside rules.
- Return ONLY the JSON object.
"""

def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Robustly parses JSON from LLM response even if wrapped in markdown codeblocks, thinking tags, or malformed quotes."""
    if not text:
        return {
            "decision": "FLAG",
            "citation_chunk_id": "empty_response",
            "citation_text": "",
            "reasoning": "Empty response from LLM.",
        }

    # Strip thinking tags like <think>...</think> if emitted by reasoning models
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try regex matching json codeblock
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try matching first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except Exception:
            pass

    # Secondary robust fallback: field-level regex extraction
    dec_m = re.search(r'"decision"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    chunk_m = re.search(r'"citation_chunk_id"\s*:\s*"([^"]+)"', cleaned)
    text_m = re.search(r'"citation_text"\s*:\s*"([^"]+)"', cleaned)
    reason_m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', cleaned)

    if dec_m:
        return {
            "decision": dec_m.group(1).upper().strip(),
            "citation_chunk_id": chunk_m.group(1).strip() if chunk_m else "",
            "citation_text": text_m.group(1).strip() if text_m else "",
            "reasoning": reason_m.group(1).strip() if reason_m else "",
        }

    # Fallback structure if parsing completely fails
    return {
        "decision": "FLAG",
        "citation_chunk_id": "parse_failure",
        "citation_text": cleaned[:200],
        "reasoning": "Failed to parse structured JSON from LLM response.",
    }


class RetrievalGovernanceJudge:
    """Domain-agnostic retrieval-grounded governance judge.

    Evaluates action requests by:
    1. Retrieving relevant policy chunks via FAISS vector similarity.
    2. Prompting an LLM to make a grounded governance decision.
    3. Returning structured output with full provenance metadata.
    """

    def __init__(self, domain: str, is_edited: bool = False, llm_client: Optional[LLMClient] = None):
        self.domain = domain
        self.is_edited = is_edited
        self.index_manager = DomainIndex(domain=domain, is_edited=is_edited)
        self.index_manager.build_or_load()
        self.llm_client = llm_client or LLMClient()

    def evaluate_scenario(
        self,
        scenario: Dict[str, Any],
        top_k: int = TOP_K_RETRIEVAL,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes System B governance evaluation:
        1. Formulates retrieval query from scenario text and parameters.
        2. Retrieves top-k policy chunks.
        3. Prompts LLM with retrieved context.
        4. Returns structured decision, citation, reasoning, and provenance.
        """
        scenario_text = scenario.get("text", "")
        details_str = json.dumps(scenario.get("details", {}), indent=2)
        query = f"{scenario_text} {details_str}"

        retrieved_chunks = self.index_manager.retrieve(query=query, top_k=top_k)

        context_blocks = []
        for i, chk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"--- [CHUNK ID: {chk['chunk_id']}] (Source: {chk['source_file']} | Header: {chk['header']}) ---\n"
                f"{chk['text']}\n"
            )

        policy_context = "\n".join(context_blocks)

        user_prompt = f"""EVALUATE THE FOLLOWING PROPOSED ACTION:

Domain: {self.domain}
Scenario Description:
{scenario_text}

Scenario Parameters / Structured Details:
{details_str}

RETRIEVED OFFICIAL POLICY PASSAGES:
{policy_context}

Please evaluate the action and respond with the required JSON object containing 'decision' (APPROVE/FLAG/BLOCK), 'citation_chunk_id', 'citation_text', and 'reasoning'.
"""

        # Use metadata-aware generation for full provenance tracking
        llm_response: LLMResponse = self.llm_client.generate_with_metadata(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            seed=seed,
        )

        parsed_decision = extract_json_from_response(llm_response.content)

        # Standardize decision label
        decision = parsed_decision.get("decision", "FLAG").upper().strip()
        if decision not in ["APPROVE", "FLAG", "BLOCK"]:
            decision = "FLAG"

        return {
            "decision": decision,
            "citation_chunk_id": parsed_decision.get(
                "citation_chunk_id",
                retrieved_chunks[0]["chunk_id"] if retrieved_chunks else "none"
            ),
            "citation_text": parsed_decision.get("citation_text", ""),
            "reasoning": parsed_decision.get("reasoning", ""),
            "retrieved_chunks": [c["chunk_id"] for c in retrieved_chunks],
            "retrieved_scores": [c.get("similarity_score", 0.0) for c in retrieved_chunks],
            "raw_response": llm_response.content,
            # Provenance metadata for research integrity
            "response_source": llm_response.response_source,
            "model_used": llm_response.model_used,
            "latency_ms": llm_response.latency_ms,
            "prompt_tokens": llm_response.prompt_tokens,
            "completion_tokens": llm_response.completion_tokens,
            "total_tokens": llm_response.total_tokens,
            "finish_reason": llm_response.finish_reason,
            "api_attempts": llm_response.attempts,
            "api_errors": llm_response.errors,
        }
