# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# config.py (main)

load_dotenv()  # Loads OPENAI_API_KEY (and anything else) from .env


@dataclass(frozen=True)
class Settings:
    """
    Global project settings.

    Notes:
    - Secrets are loaded from environment variables (via .env).
    - Paths are resolved relative to this file, so the project is portable.
    """

    # LLM
    openai_model: str = "gpt-5.2"
    temperature: float = 0.2

    # Paths
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    runs_dir: Path = processed_dir / "runs"
    researched_output_dir: Path = processed_dir / "researched"
    baseline_output_dir: Path = processed_dir / "baseline"

    # Filenames
    raw_filename: str = "sample_customers_q3_2025.xlsx"
    # enriched_filename: str = "customers_enriched.jsonl"

    # Secrets (loaded from env)
    openai_api_key: str = ""
    tavily_api_key: str = ""

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        object.__setattr__(self, "openai_api_key", api_key)

        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Put it in a .env file (OPENAI_API_KEY=...) "
                "or export it in your environment."
            )
        
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        object.__setattr__(self, "tavily_api_key", tavily_key)
        
        # Note: Tavily key is optional - only required if research_enabled is True

    # Pipeline settings
    research_enabled: bool = True
    
    # Tavily API settings
    tavily_url: str = "https://api.tavily.com/search"
    tavily_max_results: int = 8
    tavily_max_content_chars: int = 400
    tavily_recency_days: int = 180
    tavily_timeout: int = 20

SETTINGS = Settings()


# Computed fields (blank for now; we’ll define these together later)
COMPUTED_FIELDS = []
