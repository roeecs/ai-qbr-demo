# AI QBR Copilot (Monday.com Home Assignment)

**Live Demo:** [Link](https://ai-qbr-demo-pgyako2lbesql3dk4hzpzb.streamlit.app/)

This project is a proof-of-concept AI copilot that generates **Quarterly Business Review (QBR)** summaries for Customer Success Managers, based on structured account data and optional external research.

The app produces:
- **Internal CSM summary** (candid, diagnostic)
- **Client-facing QBR narrative** (polished, safe, ready to share)

Both are generated from the same grounded insights pipeline.

---

## What this demo shows
- Turning **raw + computed account metrics** into structured insights
- Separating **reasoning** from **presentation** (internal vs external)
- Optional use of **online research as validation / challenger**, not as a source of new facts
- Output that is **QBR-ready** (Markdown, downloadable)

This is intentionally a **POC** focused on correctness, structure, and explainability rather than UI polish.

---

## How to use the demo
1. Open the **Live Demo** link above
2. Select a company from the dropdown
3. Click **Generate QBR**
4. Review:
   - Internal CSM summary
   - Client-facing QBR summary
5. Download the client-facing summary as a `.md` file

---

## Run locally
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
export TAVILY_API_KEY=your_tavily_key_here
streamlit run ui.py
