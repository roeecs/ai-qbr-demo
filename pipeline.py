# pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import SETTINGS
# from assets import PIPELINE_INPUT_FIELDS, PIPELINE_COMPUTED_FIELDS


# ======================
# Pipeline main function
# ======================

def main(company_name: str, perform_research: bool) -> Optional[Dict[str, Any]]:
    """
    Run the full pipeline for a single company.
    
    Args:
        company_name: Name of the company to process (must match account_name in data)
        perform_research: Whether to run web research for the company
    
    Returns:
        Final output dictionary, or None if company not found
    """
    raw_customers = load_customers()
    
    # Find the matching company
    raw = None
    for customer in raw_customers:
        if customer.get("account_name") == company_name:
            raw = customer
            break
    
    if raw is None:
        print(f"Error: Company '{company_name}' not found in raw customers data.")
        return None
    
    # --- Pipeline steps ---

    # Compute features
    features = compute_features(raw)
    
    # Run research only if requested
    research = None
    if perform_research:
        research = run_research(raw, features)
    
    # Generate insights 
    insights = generate_reasoning(raw, features, research)

    # Format QBR
    final_output = format_qbr(raw, features, research, insights)

    # --- End of pipeline steps ---
    
    # Save output
    save_output(company_name, perform_research, final_output)
    
    # Run validation checks
    validate_research_behavior(final_output, research)
    
    return final_output


# ======================
# Step functions
# ======================


def load_customers() -> List[Dict[str, Any]]:
    """
    Load raw customer records from the input Excel file.

    - One row per customer
    - Column names are preserved as-is
    - No transformation, validation, or interpretation
    """

    file_path = SETTINGS.raw_dir / SETTINGS.raw_filename

    df = pd.read_excel(file_path)

    # Convert each row to a plain dict
    customers: List[Dict[str, Any]] = df.to_dict(orient="records")

    return customers


import json
import requests
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from assets import (
    PIPELINE_COMPUTED_FIELDS,
    PIPELINE_INPUT_FIELDS,
    InputField,
    InsightsPackage,
    REASONING_PROMPT,
    FormattedOutput,
    FORMATTING_PROMPT,
    ResearchDigest,
    RESEARCH_SUMMARIZATION_PROMPT,
)
from pydantic import ValidationError


# ======================
# LLM Helper Functions
# ======================

def get_llm() -> ChatOpenAI:
    """
    Create and return a configured ChatOpenAI LLM instance.
    
    Returns:
        ChatOpenAI instance configured with settings from config
    """
    return ChatOpenAI(
        model=SETTINGS.openai_model,
        temperature=SETTINGS.temperature,
        api_key=SETTINGS.openai_api_key,
    )


def compute_features(raw: Dict[str, Any]) -> List[InputField]:
    """
    Compute deterministic fields based on PIPELINE_COMPUTED_FIELDS.

    Rules:
    - Use raw values only
    - If any required input is missing or non-numeric → skip that computed field
    - No normalization beyond the operation definition
    
    Returns:
    - Combined list of InputField objects from both raw and computed features
    """

    # Collect raw input fields that exist in the raw data
    raw_fields: List[InputField] = []
    for field_dict in PIPELINE_INPUT_FIELDS:
        if field_dict["key"] in raw:
            raw_fields.append(InputField(**field_dict))

    # Compute computed fields
    computed_fields: List[InputField] = []
    
    for field_dict in PIPELINE_COMPUTED_FIELDS:
        values = []

        for key in field_dict["fields"]:
            val = raw.get(key)
            if val is None or not isinstance(val, (int, float)):
                values = []
                break
            values.append(val)

        if not values:
            continue

        if field_dict["op"] == "ratio_mean_product":
            result = 1.0
            for v in values:
                result *= v
        elif field_dict["op"] == "ratio_mean_diff":
            result = values[0] - values[1]
        else:
            raise ValueError(f"Unknown op: {field_dict['op']}")

        # Convert computed field to InputField
        computed_fields.append(InputField(
            key=field_dict["key"],
            name=field_dict["name"],
            desc=field_dict["desc"],
            benchmark_stats=None  # Computed fields don't have benchmark stats
        ))

    # Combine raw and computed fields
    return raw_fields + computed_fields


def run_research(
    snapshot: Dict[str, Any],
    features: List[InputField],
) -> Optional[Dict[str, Any]]:
    """
    Run web research for the account using Tavily API.
        
    Args:
        snapshot: Raw customer data dictionary
        features: List of InputField objects (raw + computed)
    
    Returns:
        Dict with research_digest (ResearchDigest object), or None if research fails
    """
    # Check if Tavily API key is available
    if not SETTINGS.tavily_api_key:
        print("Warning: TAVILY_API_KEY not set. Skipping research.")
        return None
    
    # Extract company name from snapshot
    company_name = snapshot.get("account_name")
    if not company_name:
        print("Warning: No account_name found in snapshot. Skipping research.")
        return None
    
    # Build query similar to the example but with dynamic company name
    # Escape quotes in company name for the query
    company_escaped = company_name.replace('"', '\\"')
    query = (
        f'"{company_escaped}" '
        "(layoffs OR restructuring OR reorganization OR earnings OR revenue OR guidance "
        "OR funding OR acquisition OR outage OR incident OR status OR pricing "
        'OR "new plan" OR deprecation) '
        "2024 2025"
    )
    
    def truncate(text: str, max_chars: int) -> str:
        """Truncate text to max_chars."""
        if not text:
            return ""
        return text[:max_chars].rstrip()
    
    # Prepare request
    payload = {
        "query": query,
        "recency_days": SETTINGS.tavily_recency_days,
        "max_results": SETTINGS.tavily_max_results,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SETTINGS.tavily_api_key}",
    }
    
    try:
        # Make API call
        resp = requests.post(SETTINGS.tavily_url, json=payload, headers=headers, timeout=SETTINGS.tavily_timeout)
        resp.raise_for_status()
        data = resp.json()
        
        # Clean and format results
        cleaned_results = []
        for r in data.get("results", []):
            cleaned_results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": truncate(r.get("content", ""), SETTINGS.tavily_max_content_chars),
                "score": r.get("score"),
            })
        
        if not cleaned_results:
            return None
        
        # Summarize results using LLM
        try:
            # Format search results for the prompt
            search_results_json = json.dumps(cleaned_results, indent=2, default=str)
            
            # Format prompt using LangChain ChatPromptTemplate
            messages = RESEARCH_SUMMARIZATION_PROMPT.format_messages(
                search_results=search_results_json
            )
            
            # Initialize LLM with structured output
            llm = get_llm()
            
            # Create output parser for explicit parsing (documentation/fallback)
            output_parser = PydanticOutputParser(pydantic_object=ResearchDigest)
            
            # Get structured output using with_structured_output (most reliable method)
            result: ResearchDigest = llm.with_structured_output(ResearchDigest).invoke(messages)
            
            # Apply post-filter to enforce "none" status for weak results
            bullets = result.bullets if result.bullets else []
            status = result.status
            
            # Post-filter: if bullets are empty, force status="none"
            if not bullets:
                status = "none"
                bullets = []
            else:
                # Check for obvious generic patterns (simple heuristic)
                generic_keywords = ["overview", "about", "profile", "linkedin", "crunchbase", "company page"]
                event_keywords = ["announced", "reported", "launched", "acquired", "funding", "earnings", "restructuring", "layoff", "hiring", "partnership"]
                
                # If all bullets contain generic keywords but no event keywords, mark as none
                all_generic = True
                for bullet in bullets:
                    bullet_lower = bullet.lower()
                    has_generic = any(keyword in bullet_lower for keyword in generic_keywords)
                    has_event = any(keyword in bullet_lower for keyword in event_keywords)
                    if has_event or not has_generic:
                        all_generic = False
                        break
                
                if all_generic:
                    status = "none"
                    bullets = []
            
            # Create final ResearchDigest with post-filter applied
            research_digest = ResearchDigest(
                status=status,
                bullets=bullets,
                timeframe_months=result.timeframe_months
            )
            
        except Exception as e:
            print(f"Warning: LLM summarization failed: {e}. Returning none status.")
            research_digest = ResearchDigest(
                status="none",
                bullets=[],
                timeframe_months=3
            )
        
        # Return research_digest - this is what downstream functions need
        return {
            "research_digest": research_digest,
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Warning: Tavily API request failed: {e}. Skipping research.")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error during research: {e}. Skipping research.")
        return None


def validate_and_log_insights(
    raw_json_str: str,
    account_identifier: str,
    output_dir: Path,
) -> InsightsPackage:
    """
    Validate raw LLM JSON output against InsightsPackage schema.
    
    Args:
        raw_json_str: Raw JSON string from LLM
        account_identifier: Account name for logging
        output_dir: Directory to save raw output for debugging
    
    Returns:
        Validated InsightsPackage
    
    Raises:
        ValidationError: If JSON doesn't match schema
        ValueError: If JSON is invalid
    """
    try:
        # Parse JSON
        parsed = json.loads(raw_json_str)
    except json.JSONDecodeError as e:
        # Save raw output for debugging
        os.makedirs(output_dir, exist_ok=True)
        debug_file = output_dir / f"{account_identifier}_raw_llm_output.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(raw_json_str)
        raise ValueError(f"Invalid JSON from LLM. Raw output saved to {debug_file}") from e
    
    try:
        # Validate with Pydantic
        validated = InsightsPackage(**parsed)
        return validated
    except ValidationError as e:
        # Save raw output for debugging
        os.makedirs(output_dir, exist_ok=True)
        debug_file = output_dir / f"{account_identifier}_raw_llm_output.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        
        # Extract first error path for clearer error message
        errors = e.errors()
        if errors:
            first_error = errors[0]
            error_path = " -> ".join(str(loc) for loc in first_error.get("loc", []))
            error_msg = first_error.get("msg", "Validation error")
            raise ValidationError(
                f"Schema validation failed at path '{error_path}': {error_msg}. "
                f"Raw output saved to {debug_file}"
            ) from e
        raise


def generate_reasoning(
    snapshot: Dict[str, Any],
    features: List[InputField],
    research: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
    """
    Generate reasoning/insights using LLM with structured output.
    
    Args:
        snapshot: Raw customer data dictionary
        features: List of InputField objects (raw + computed)
        research: Optional research data (dict with 'research_digest' ResearchDigest or None)
    
    Returns:
        InsightsPackage as a dictionary
    """
    # Build lookup for computed field definitions
    computed_field_by_key = {f["key"]: f for f in PIPELINE_COMPUTED_FIELDS}
    
    # Convert InputField list to feature dicts with values from snapshot
    feature_dicts = []
    for field in features:
        # Check if this is a computed field
        if field.key in computed_field_by_key:
            # Recompute the value
            computed_def = computed_field_by_key[field.key]
            values = []
            for key in computed_def["fields"]:
                val = snapshot.get(key)
                if val is None or not isinstance(val, (int, float)):
                    values = []
                    break
                values.append(val)
            
            if values:
                if computed_def["op"] == "ratio_mean_product":
                    value = 1.0
                    for v in values:
                        value *= v
                elif computed_def["op"] == "ratio_mean_diff":
                    value = values[0] - values[1]
                else:
                    value = None
            else:
                value = None
        else:
            # Raw field - get from snapshot
            value = snapshot.get(field.key)
        
        feature_dict = {
            "key": field.key,
            "name": field.name,
            "desc": field.desc,
            "benchmark_stats": field.benchmark_stats,
            "value": value,
        }
        feature_dicts.append(feature_dict)
    
    # Get account identifier and timeframe
    account_identifier = snapshot.get("account_name", "Unknown Account")
    timeframe = "Q3 2025"  # TODO: Make this configurable or extract from data
    
    # Format research section
    if research is not None and research.get("research_digest"):
        research_digest = research["research_digest"]
        # Handle both Pydantic model and dict
        if hasattr(research_digest, "status"):
            status = research_digest.status
            bullets = research_digest.bullets
        else:
            status = research_digest.get("status", "none")
            bullets = research_digest.get("bullets", [])
        
        if status == "relevant" and bullets:
            research_bullets_str = json.dumps(bullets, indent=2)
            research_section = f"""- research (optional): {research_bullets_str}
  - This is a list of up to 3 short strings.
  - Each string summarizes a recent external signal or news item about the company.
  - Use research ONLY to validate or challenge insights (see RESEARCH USAGE RULES)."""
            research_data_section = f"\nOPTIONAL RESEARCH (status: relevant, {len(bullets)} bullets):\n{research_bullets_str}"
            research_signals_section = """6) Research Signals (ONLY IF research.status="relevant")
   - For each research bullet, produce:
     - title
     - short_summary
     - source: always set to "research"
   - Link relevant research signals as evidence in risks, opportunities,
     or actions when appropriate.
   - Use research to validate or challenge insights, not to create new ones."""
        else:
            # status == "none" or no bullets
            research_section = """- research (optional): null
  - Research was performed but no meaningful company-specific news was found.
  - Ignore research entirely and do NOT fabricate sources."""
            research_data_section = "\nOPTIONAL RESEARCH: none (no meaningful news found)"
            research_signals_section = ""
    else:
        research_section = """- research (optional): null
  - Research is not available for this account.
  - Ignore research entirely and do NOT fabricate sources."""
        research_data_section = "\nOPTIONAL RESEARCH: null (not provided)"
        research_signals_section = ""
    
    # Format features as JSON
    features_json = json.dumps(feature_dicts, indent=2, default=str)
    
    # Format prompt using LangChain ChatPromptTemplate
    messages = REASONING_PROMPT.format_messages(
        timeframe=timeframe,
        account_identifier=account_identifier,
        research_section=research_section,
        research_signals_section=research_signals_section,
        features=features_json,
        research_data_section=research_data_section,
    )
    
    # Initialize LLM with structured output
    llm = get_llm()
    
    # Create output parser for explicit parsing (documentation/fallback)
    output_parser = PydanticOutputParser(pydantic_object=InsightsPackage)
    
    # Get structured output using with_structured_output (most reliable method)
    result: InsightsPackage = llm.with_structured_output(InsightsPackage).invoke(messages)
    
    # Log validated output for debugging (run-scoped artifact)
    try:
        # Create debug directory
        debug_dir = SETTINGS.processed_dir / "debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save validated output as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = debug_dir / f"{account_identifier}_{timestamp}_reasoning_output.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Don't fail the pipeline if logging fails
        print(f"Warning: Failed to log reasoning output: {e}")
    
    # Convert to dict
    return result.model_dump()


def format_qbr(
    snapshot: Dict[str, Any],
    features: List[InputField],
    research: Optional[Dict[str, Any]],
    insights: Dict[str, Any],
    ) -> Dict[str, Any]:
    """
    Format insights into internal CSM brief and external QBR summary using LLM with structured output.
    
    Args:
        snapshot: Raw customer data dictionary
        features: List of InputField objects (raw + computed)
        research: Optional research data (dict with 'research_digest' ResearchDigest or None)
        insights: InsightsPackage as a dictionary from generate_reasoning
    
    Returns:
        FormattedOutput as a dictionary
    """
    # Build lookup for computed field definitions
    computed_field_by_key = {f["key"]: f for f in PIPELINE_COMPUTED_FIELDS}
    
    # Convert InputField list to feature dicts with values from snapshot
    feature_dicts = []
    for field in features:
        # Check if this is a computed field
        if field.key in computed_field_by_key:
            # Recompute the value
            computed_def = computed_field_by_key[field.key]
            values = []
            for key in computed_def["fields"]:
                val = snapshot.get(key)
                if val is None or not isinstance(val, (int, float)):
                    values = []
                    break
                values.append(val)
            
            if values:
                if computed_def["op"] == "ratio_mean_product":
                    value = 1.0
                    for v in values:
                        value *= v
                elif computed_def["op"] == "ratio_mean_diff":
                    value = values[0] - values[1]
                else:
                    value = None
            else:
                value = None
        else:
            # Raw field - get from snapshot
            value = snapshot.get(field.key)
        
        feature_dict = {
            "key": field.key,
            "name": field.name,
            "desc": field.desc,
            "benchmark_stats": field.benchmark_stats,
            "value": value,
        }
        feature_dicts.append(feature_dict)
    
    # Get account identifier and timeframe from insights
    account_identifier = insights.get("account_identifier", snapshot.get("account_name", "Unknown Account"))
    timeframe = insights.get("timeframe", "Q3 2025")
    
    # Format features and insights as JSON
    features_json = json.dumps(feature_dicts, indent=2, default=str)
    extracted_insights_json = json.dumps(insights, indent=2, default=str)
    
    # Format research_digest section for formatter
    if research is not None and research.get("research_digest"):
        research_digest = research["research_digest"]
        # Handle both Pydantic model and dict
        if hasattr(research_digest, "status"):
            status = research_digest.status
            bullets = research_digest.bullets
        else:
            status = research_digest.get("status", "none")
            bullets = research_digest.get("bullets", [])
        
        if status == "relevant" and bullets:
            research_digest_section = f"""
RESEARCH DIGEST (for company news snapshot):
- status: "relevant"
- bullets: {json.dumps(bullets, indent=2)}
- Use these bullets to create the company_news_snapshot in the internal brief.
"""
        else:
            research_digest_section = """
RESEARCH DIGEST (for company news snapshot):
- status: "none"
- bullets: []
- Set company_news_snapshot to: "No meaningful company-specific news detected in the last few months."
"""
    else:
        research_digest_section = """
RESEARCH DIGEST (for company news snapshot):
- status: "none" (research not performed)
- bullets: []
- Set company_news_snapshot to: "No meaningful company-specific news detected in the last few months."
"""
    
    # Format prompt using LangChain ChatPromptTemplate
    messages = FORMATTING_PROMPT.format_messages(
        timeframe=timeframe,
        account_identifier=account_identifier,
        features=features_json,
        extracted_insights=extracted_insights_json,
        research_digest_section=research_digest_section,
    )
    
    # Initialize LLM with structured output
    llm = get_llm()
    
    # Create output parser for explicit parsing (documentation/fallback)
    output_parser = PydanticOutputParser(pydantic_object=FormattedOutput)
    
    # Get structured output with error handling
    try:
        result: FormattedOutput = llm.with_structured_output(FormattedOutput).invoke(messages)
    except ValidationError as e:
        # Save raw output for debugging
        try:
            debug_dir = SETTINGS.processed_dir / "debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"{account_identifier}_{timestamp}_formatter_output.json"
            
            # Try to get raw response for debugging
            try:
                raw_response = llm.invoke(messages)
                raw_content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                # Reconstruct prompt preview from messages for debugging
                prompt_preview = "\n".join([msg.content[:500] if hasattr(msg, 'content') else str(msg)[:500] for msg in messages])
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "error": "ValidationError",
                        "validation_errors": str(e.errors() if hasattr(e, 'errors') else e),
                        "raw_llm_response": raw_content,
                        "prompt_preview": prompt_preview[:1000] + "..." if len(prompt_preview) > 1000 else prompt_preview
                    }, f, indent=2, ensure_ascii=False)
            except Exception as save_error:
                # If we can't get raw response, at least save the error
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "error": "ValidationError",
                        "validation_errors": str(e.errors() if hasattr(e, 'errors') else e),
                        "save_error": str(save_error)
                    }, f, indent=2, ensure_ascii=False)
            
            # Extract first error path for clearer error message
            errors = e.errors() if hasattr(e, 'errors') else []
            if errors:
                first_error = errors[0]
                error_path = " -> ".join(str(loc) for loc in first_error.get("loc", []))
                error_msg = first_error.get("msg", "Validation error")
                raise ValidationError(
                    f"Schema validation failed at path '{error_path}': {error_msg}. "
                    f"Raw output saved to {debug_file}"
                ) from e
        except Exception as log_error:
            # Don't fail the pipeline if logging fails
            print(f"Warning: Failed to log formatter output: {log_error}")
        
        # Re-raise the original validation error
        raise
    
    # Convert to dict
    return result.model_dump()


def save_output(company_name: str, perform_research: bool, final_output: Dict[str, Any]) -> None:
    """
    Save final output to the appropriate directory based on research flag.
    
    Args:
        company_name: Name of the company
        perform_research: Whether research was performed
        final_output: The final output dictionary to save
    """
    # Determine output directory
    if perform_research:
        base_dir = SETTINGS.researched_output_dir
    else:
        base_dir = SETTINGS.baseline_output_dir
    
    # Create company-specific directory
    company_dir = base_dir / company_name
    os.makedirs(company_dir, exist_ok=True)
    
    # Remove all existing files in the company directory
    for existing_file in company_dir.glob("*.json"):
        existing_file.unlink()
    
    # Generate timestamp in format: dd/mm/yy - hh:mm
    now = datetime.now()
    timestamp = now.strftime("%d/%m/%y - %H:%M")
    # Replace slashes with dashes for filename safety
    safe_timestamp = timestamp.replace("/", "-")
    
    # Create filename
    filename = f"{safe_timestamp}.json"
    filepath = company_dir / filename
    
    # Save output as JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to: {filepath}")


def validate_research_behavior(final_output: Dict[str, Any], research: Optional[Dict[str, Any]]) -> None:
    """
    Basic validation checks for research behavior.
    
    Validates:
    - Case A: When research_digest.status="none", internal snapshot contains "No meaningful"
    - Case B: When research_digest.status="relevant", internal snapshot contains bullets
    - External QBR never has market_context key
    
    Args:
        final_output: Final output dictionary from format_qbr
        research: Research data dict (with research_digest) or None
    """
    internal_brief = final_output.get("internal_csm_brief", {})
    external_qbr = final_output.get("external_qbr_summary", {})
    company_news_snapshot = internal_brief.get("company_news_snapshot", "")
    
    # Check that external QBR has no market_context
    if "market_context" in external_qbr:
        print("WARNING: External QBR contains market_context key (should be removed)")
    
    # Determine research status
    research_status = None
    if research and research.get("research_digest"):
        research_digest = research["research_digest"]
        if hasattr(research_digest, "status"):
            research_status = research_digest.status
        else:
            research_status = research_digest.get("status", "none")
    
    # Case A: status="none" or no research
    if research_status == "none" or research_status is None:
        if "No meaningful" not in company_news_snapshot:
            print(f"WARNING: Research status is 'none' but company_news_snapshot doesn't contain 'No meaningful': {company_news_snapshot[:100]}")
        else:
            print("✓ Validation passed: 'none' status correctly reflected in company_news_snapshot")
    
    # Case B: status="relevant"
    elif research_status == "relevant":
        if not company_news_snapshot or "No meaningful" in company_news_snapshot:
            print(f"WARNING: Research status is 'relevant' but company_news_snapshot indicates no news: {company_news_snapshot[:100]}")
        else:
            print("✓ Validation passed: 'relevant' status correctly reflected in company_news_snapshot")


if __name__ == "__main__":
    import sys
    
    # Example usage - can be modified to accept command-line arguments
    if len(sys.argv) >= 3:
        company = sys.argv[1]
        do_research = sys.argv[2].lower() in ("true", "1", "yes")
    else:
        # Default values for testing
        company = "Altura Systems"
        do_research = False
    
    print(f"Running pipeline for {company} with research: {do_research}")
    main(company, do_research)
