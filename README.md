# 1915(c) HCBS Waiver Extraction

Extracts structured data from CMS 1915(c) Home and Community-Based Services waiver application PDFs (267 active waivers in `data/pdfs_267_latest/`). Full system description: [`data/archive/docs/SYSTEM_OVERVIEW.md`](data/archive/docs/SYSTEM_OVERVIEW.md).

## Layout
Top level is three agents plus one `data/` folder:

| Path | What it is |
|---|---|
| `waiver_level_extraction_agent/` | Waiver-level (overview) extraction: critic, AcroForm resolver, re-extraction (entry: `run_pipeline.py`, or `adk web`) |
| `services_extraction_agent/` | Service-level extraction, Appendix C / E / J (entry: `run_services_pipeline.py`, or `adk web`) |
| `analytics_agent/` | Run logs and error analysis |
| `data/` | Everything else - see below |

All file locations come from one module, `waiver_level_extraction_agent/shared_libraries/paths.py`.

## The data folder
| Path | What it is |
|---|---|
| `data/outputs/` | New extraction results: `waiver_level_extraction.csv`, `service_level_extraction.csv`, `waivers.db`, one folder per waiver |
| `data/pdfs_267_latest/` | Source PDFs |
| `data/stages/`, `data/chroma/`, `data/logs/` | ADK stage checkpoints, vector store, per-waiver event logs |
| `data/human_overrides.jsonl`, `services_human_overrides.jsonl` | Audit records, created when you run the tools |
| `data/archive/scripts/export_to_csv.py` | The only archived file kept in this repo; the tools load it for the CSV column order |

## Choosing the input and the output
- CLI: `python run_pipeline.py --file <any.pdf>` / `--waiver <KEY>` / `--dir <any folder>`; services: `--file`, `--waiver`, `--pdf-dir`. In `adk web`, give a PDF path or a folder (`ADK_PDF_DIRS` adds search folders).
- Output CSV defaults to `data/outputs/waiver_level_extraction.csv` and `service_level_extraction.csv`. Change with `--output-csv <name or path>` (CLI), the `output_csv` argument of the ADK write-to-CSV tools, or the `WAIVER_LEVEL_CSV` / `SERVICE_LEVEL_CSV` environment variables. In ADK mode nothing is written to a CSV until you ask.

Configuration lives in `waiver_level_extraction_agent/.env` (see `.env.example`). The pipelines call Gemini through Vertex AI. Do not start a run without checking cost and the PDF-cache behavior first.
