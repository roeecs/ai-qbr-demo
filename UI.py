# UI.py
"""
Streamlit UI for Monday QBR Copilot
Main executor and presentation layer for the QBR pipeline.
"""

import streamlit as st

# Set page config - must be first Streamlit command
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* slightly larger body text (markdown only) */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        font-size: 1.25rem;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True
)
left, center, right = st.columns([1, 3, 1])  # 3/5 ≈ 60%


import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pipeline
from config import SETTINGS


# ======================
# Helper Functions
# ======================

def load_customer_names() -> List[str]:
    """Extract account names from Excel via pipeline.load_customers()."""
    try:
        customers = pipeline.load_customers()
        account_names = [customer.get("account_name", "") for customer in customers if customer.get("account_name")]
        return sorted(set(account_names))  # Remove duplicates and sort
    except Exception as e:
        st.error(f"Error loading customers: {e}")
        return []


def parse_timestamp(filename: str) -> str:
    """
    Convert "dd-mm-yy - hh:mm.json" to "Jan 3, 12:14" format.
    
    Example: "03-01-26 - 13:05.json" -> "Jan 3, 13:05"
    """
    try:
        # Remove .json extension
        base = filename.replace(".json", "")
        # Parse: "dd-mm-yy - hh:mm"
        parts = base.split(" - ")
        if len(parts) != 2:
            return base
        
        date_part = parts[0]  # "dd-mm-yy"
        time_part = parts[1]   # "hh:mm"
        
        # Parse date
        day, month, year = date_part.split("-")
        # Convert 2-digit year to 4-digit (assuming 20xx)
        full_year = 2000 + int(year)
        
        # Create datetime object
        dt = datetime(int(full_year), int(month), int(day))
        
        # Format as "Jan 3, 12:14"
        month_name = dt.strftime("%b")
        day_num = dt.day
        formatted = f"{month_name} {day_num}, {time_part}"
        
        return formatted
    except Exception as e:
        # If parsing fails, return the original filename without .json
        return filename.replace(".json", "")


def scan_drafts() -> List[Dict[str, Any]]:
    """
    Scan baseline and researched directories, return list of draft metadata.
    
    Returns list of dicts with keys: company, variant, timestamp, filepath, display_name
    """
    drafts = []
    
    # Scan baseline directory
    baseline_dir = SETTINGS.baseline_output_dir
    # Ensure parent directory exists (but don't create if it doesn't - just handle gracefully)
    if baseline_dir.exists():
        for company_dir in baseline_dir.iterdir():
            if company_dir.is_dir():
                company_name = company_dir.name
                for json_file in company_dir.glob("*.json"):
                    timestamp_str = parse_timestamp(json_file.name)
                    drafts.append({
                        "company": company_name,
                        "variant": "Default",
                        "timestamp": timestamp_str,
                        "filepath": str(json_file),
                        "display_name": f"{company_name} — Draft (Default) — {timestamp_str}"
                    })
    
    # Scan researched directory
    researched_dir = SETTINGS.researched_output_dir
    # Ensure parent directory exists (but don't create if it doesn't - just handle gracefully)
    if researched_dir.exists():
        for company_dir in researched_dir.iterdir():
            if company_dir.is_dir():
                company_name = company_dir.name
                for json_file in company_dir.glob("*.json"):
                    timestamp_str = parse_timestamp(json_file.name)
                    drafts.append({
                        "company": company_name,
                        "variant": "Researched",
                        "timestamp": timestamp_str,
                        "filepath": str(json_file),
                        "display_name": f"{company_name} — Draft (Researched) — {timestamp_str}"
                    })
    
    # Sort by timestamp (newest first) - parse timestamp for sorting
    def sort_key(draft):
        try:
            # Extract date part from display_name for sorting
            parts = draft["timestamp"].split(", ")
            if len(parts) == 2:
                date_part = parts[0]  # "Jan 3"
                time_part = parts[1]  # "13:05"
                # Parse month and day
                month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                month_str, day_str = date_part.split()
                month = month_map.get(month_str, 1)
                day = int(day_str)
                hour, minute = map(int, time_part.split(":"))
                return datetime(2026, month, day, hour, minute)  # Assuming 2026
        except:
            pass
        return datetime.min
    
    drafts.sort(key=sort_key, reverse=True)
    
    return drafts


def load_draft(filepath: str) -> Optional[Dict[str, Any]]:
    """Load and parse JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading draft: {e}")
        return None


def format_internal_display(data: Dict[str, Any], is_researched: bool = False) -> None:
    """Format internal_csm_brief for display.
    
    Args:
        data: The draft data dictionary
        is_researched: Whether this draft was created with research enabled
    """
    if "internal_csm_brief" not in data:
        st.warning("No internal CSM brief found in data.")
        return
    
    brief = data["internal_csm_brief"]
    
    # Executive Summary
    st.subheader("Executive Summary")
    st.info(brief.get("executive_snapshot", "No snapshot available."))
    
    # Health & Confidence - Centered with colors
    col1, col2 = st.columns(2)
    with col1:
        health = brief.get("health_label", "unknown")
        health_upper = health.upper()
        if health == "green":
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><span style="background-color: #d4edda; color: #155724; padding: 0.5rem 1rem; border-radius: 0.25rem; display: inline-block;"><strong>Health:</strong> {health_upper}</span></div>',
                unsafe_allow_html=True
            )
        elif health == "yellow":
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><span style="background-color: #fff3cd; color: #856404; padding: 0.5rem 1rem; border-radius: 0.25rem; display: inline-block;"><strong>Health:</strong> {health_upper}</span></div>',
                unsafe_allow_html=True
            )
        elif health == "red":
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><span style="background-color: #f8d7da; color: #721c24; padding: 0.5rem 1rem; border-radius: 0.25rem; display: inline-block;"><strong>Health:</strong> {health_upper}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><strong>Health:</strong> {health_upper}</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        confidence = brief.get("confidence", "unknown")
        confidence_upper = confidence.upper()
        if confidence == "high":
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><span style="background-color: #d4edda; color: #155724; padding: 0.5rem 1rem; border-radius: 0.25rem; display: inline-block;"><strong>Confidence:</strong> {confidence_upper}</span></div>',
                unsafe_allow_html=True
            )
        elif confidence == "med":
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><span style="background-color: #fff3cd; color: #856404; padding: 0.5rem 1rem; border-radius: 0.25rem; display: inline-block;"><strong>Confidence:</strong> {confidence_upper}</span></div>',
                unsafe_allow_html=True
            )
        elif confidence == "low":
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><span style="background-color: #f8d7da; color: #721c24; padding: 0.5rem 1rem; border-radius: 0.25rem; display: inline-block;"><strong>Confidence:</strong> {confidence_upper}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align: center; padding: 1rem;"><strong>Confidence:</strong> {confidence_upper}</div>',
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # Company News Snapshot (only show for researched drafts)
    if is_researched:
        company_news = brief.get("company_news_snapshot", "")
        company_news_confidence = brief.get("company_news_confidence")
        if company_news:
            with st.expander("📰 Company News Snapshot", expanded=True):
                st.write(company_news)
                if company_news_confidence:
                    confidence_upper = company_news_confidence.upper()
                    if company_news_confidence == "high":
                        st.markdown(
                            f'<div style="text-align: left; padding: 0.5rem 0;"><span style="background-color: #d4edda; color: #155724; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block; font-size: 0.875rem;"><strong>Confidence:</strong> {confidence_upper}</span></div>',
                            unsafe_allow_html=True
                        )
                    elif company_news_confidence == "med":
                        st.markdown(
                            f'<div style="text-align: left; padding: 0.5rem 0;"><span style="background-color: #fff3cd; color: #856404; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block; font-size: 0.875rem;"><strong>Confidence:</strong> {confidence_upper}</span></div>',
                            unsafe_allow_html=True
                        )
                    elif company_news_confidence == "low":
                        st.markdown(
                            f'<div style="text-align: left; padding: 0.5rem 0;"><span style="background-color: #f8d7da; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block; font-size: 0.875rem;"><strong>Confidence:</strong> {confidence_upper}</span></div>',
                            unsafe_allow_html=True
                        )
    
    # Wins
    if brief.get("wins"):
        with st.expander("🏆 Wins", expanded=True):
            for i, win in enumerate(brief["wins"], 1):
                st.markdown(f"**{i}. {win.get('title', 'Untitled')}**")
                st.write(win.get("detail", ""))
                if i < len(brief["wins"]):
                    st.write("")
    
    # Risks
    if brief.get("risks"):
        with st.expander("⚠️ Risks", expanded=True):
            for i, risk in enumerate(brief["risks"], 1):
                st.markdown(f"**{i}. {risk.get('title', 'Untitled')}**")
                st.write(risk.get("detail", ""))
                if i < len(brief["risks"]):
                    st.write("")
    
    # Opportunities
    if brief.get("opportunities"):
        with st.expander("💡 Opportunities", expanded=True):
            for i, opp in enumerate(brief["opportunities"], 1):
                st.markdown(f"**{i}. {opp.get('title', 'Untitled')}**")
                st.write(opp.get("detail", ""))
                if i < len(brief["opportunities"]):
                    st.write("")
    
    # Action Plan
    if brief.get("action_plan"):
        with st.expander("📋 Action Plan", expanded=True):
            for action in brief["action_plan"]:
                priority = action.get("priority", "?")
                title = action.get("title", "Untitled")
                owner = action.get("owner", "Unknown")
                
                st.markdown(f"**Priority {priority}: {title}**")
                st.markdown(f"*Owner: {owner}*")
                st.markdown(f"**Next Step:** {action.get('next_step', 'N/A')}")
                st.markdown(f"**Expected Outcome:** {action.get('expected_outcome', 'N/A')}")
                st.write("")


def format_external_markdown(data: Dict[str, Any]) -> str:
    """
    Format external_qbr_summary as beautiful markdown.
    Returns formatted markdown string ready for display and download.
    """
    if "external_qbr_summary" not in data:
        return "No external QBR summary available."
    
    summary = data["external_qbr_summary"]
    account_name = data.get("account_identifier", "Account")
    timeframe = data.get("timeframe", "")
    
    lines = []
    
    # Title
    lines.append(f"# {account_name} - Quarterly Business Review")
    if timeframe:
        lines.append(f"**{timeframe}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Opening narrative
    opening = summary.get("opening_narrative", "")
    if opening:
        lines.append(opening)
        lines.append("")
    
    # Highlights
    if summary.get("highlights"):
        lines.append("## Highlights")
        lines.append("")
        for highlight in summary["highlights"]:
            lines.append(f"- {highlight}")
        lines.append("")
    
    # Risks & Watch-outs
    if summary.get("risks_and_watchouts"):
        lines.append("## Risks & Watch-outs")
        lines.append("")
        for risk in summary["risks_and_watchouts"]:
            lines.append(f"- {risk}")
        lines.append("")
    
    # Recommendations/Next Steps
    if summary.get("recommendations_next_steps"):
        lines.append("## Recommendations & Next Steps")
        lines.append("")
        for rec in summary["recommendations_next_steps"]:
            lines.append(f"- {rec}")
        lines.append("")
    
    # Closing (forward-looking)
    closing = summary.get("closing", "")
    if closing:
        lines.append(closing)
        lines.append("")
    
    return "\n".join(lines)


# ======================
# Main UI
# ======================

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if "selected_draft_data" not in st.session_state:
        st.session_state.selected_draft_data = None
    with center:
        # 1. Title Section (centered)
        center.markdown("<h1 style='text-align: center;'>Monday.com QBR copilot</h1>", unsafe_allow_html=True)
        center.divider()
        
        # 2. Pipeline Execution Area
        st.subheader("Build QBR")
        
        # Load customer names
        customer_names = load_customer_names()
        
        if not customer_names:
            st.error("No customers found. Please check the data file.")
            return
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_company = st.selectbox(
                "Select Customer",
                options=customer_names,
                index=0 if customer_names else None
            )
        
        with col2:
            perform_research = st.checkbox(
                "Perform online research",
                value=False,
                help="Enable web research to gather recent company news and events that may impact the QBR."
            )
        
        with col3:
            build_button = st.button("Build", type="primary", use_container_width=True)
        
        # Handle Build button click
        if build_button:
            if not selected_company:
                st.error("Please select a customer.")
            else:
                with st.spinner(f"Building QBR for {selected_company}..."):
                    try:
                        result = pipeline.main(selected_company, perform_research)
                        if result:
                            st.session_state.selected_draft_data = result
                            st.session_state.draft_variant = "Researched" if perform_research else "Default"
                            st.success(f"QBR built successfully for {selected_company}!")
                            st.rerun()
                        else:
                            st.error(f"Failed to build QBR for {selected_company}.")
                    except Exception as e:
                        st.error(f"Error building QBR: {e}")
        
        st.divider()
        
        # 3. Draft Selection Section
        st.subheader("Select a draft")
        
        drafts = scan_drafts()
        
        if not drafts:
            st.info("No drafts found. Build a QBR to create your first draft.")
        else:
            draft_options = [d["display_name"] for d in drafts]
            draft_filepaths = {d["display_name"]: d["filepath"] for d in drafts}
            
            selected_draft_name = st.selectbox(
                "Choose a draft to view",
                options=draft_options,
                index=0,
                label_visibility="collapsed",
                key="draft_selector"
            )
            
            # Load selected draft
            if selected_draft_name:
                filepath = draft_filepaths[selected_draft_name]
                # Find the variant for this draft
                draft_variant = None
                for draft in drafts:
                    if draft["filepath"] == filepath:
                        draft_variant = draft["variant"]
                        break
                
                # Only reload if this is a different draft than currently loaded
                current_filepath = st.session_state.get("current_draft_filepath")
                if current_filepath != filepath:
                    draft_data = load_draft(filepath)
                    if draft_data:
                        st.session_state.selected_draft_data = draft_data
                        st.session_state.current_draft_filepath = filepath
                        st.session_state.draft_variant = draft_variant
                elif draft_variant:
                    # Update variant if filepath matches but variant might have changed
                    st.session_state.draft_variant = draft_variant
        
        st.divider()
        
        # 4. Internal Results Presentation
        if st.session_state.selected_draft_data:
            st.subheader("Internal CSM Brief")
            draft_variant = st.session_state.get("draft_variant", "Default")
            format_internal_display(st.session_state.selected_draft_data, is_researched=(draft_variant == "Researched"))
            st.divider()
        
        # 5. External QBR Presentation
        if st.session_state.selected_draft_data:
            st.subheader("External QBR Summary")
            
            # Format and display as markdown
            external_markdown = format_external_markdown(st.session_state.selected_draft_data)
            st.markdown(external_markdown)
            
            # Download button
            account_name = st.session_state.selected_draft_data.get("account_identifier", "Account")
            # Sanitize filename (remove invalid characters)
            safe_filename = "".join(c for c in account_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_filename = safe_filename.replace(' ', '_')
            filename = f"{safe_filename}_QBR.md"
            
            st.download_button(
                label="📥 Download External QBR Summary",
                data=external_markdown,
                file_name=filename,
                mime="text/markdown",
                type="primary"
            )


if __name__ == "__main__":
    main()

