# Open-Source RAG Platform Architecture & Guidelines

## Project Overview
A production-grade Retrieval-Augmented Generation (RAG) system utilizing fully open-source components. The core architecture comprises an embedding pipeline, a vector database for semantic retrieval, and an open-source LLM orchestration layer.

## Core Engineering Workflows (CRITICAL LOGGING)
Before writing code or executing changes, you must strictly follow this interactive, file-documented workflow. 

### Phase 1: Interactive Architecture Planning & Logging
1. **Analyze & Blueprint:** Analyze the task. Before printing it in the chat, append or update the file `architecture_log.md` in the root directory. 
2. **Log Structure:** The entry in `architecture_log.md` must contain:
   - **Feature/Task Name** and Date.
   - **Proposed Changes:** Exact files to create/modify.
   - **Architectural Trade-offs:** Pros, cons, and performance impacts (e.g., memory vs. speed).
3. **Await Explicit Approval:** Once the file is updated, output this text in the main chat: `[SYSTEM] Architecture plan written to architecture_log.md. WAITING FOR USER APPROVAL: Please review the file and type 'Proceed' to execute.` Stop execution immediately. Do not write code until the user approves.

### Phase 2: Implementation & Step Documentation
1. **Incremental Execution:** Implement changes modularly. 
2. **Log the Implementation Steps:** As you write code, update `architecture_log.md` with a "Steps Taken" section detailing what was successfully created, what failed/was corrected, and the testing results.
3. **Verify:** Run formatting and verification tools instantly after generating code to patch syntax or type issues.

---

## Technical Stack & Commands
- **Language/Runtime:** Python 3.11+ / Poetry
- **Frameworks:** LangChain/LlamaIndex, FastAPI
- **Vector DB:** Qdrant / Chroma / Milvus (Local Docker)
- **Models:** Hugging Face / Ollama (Local execution)

### Operational Commands
- **Environment Setup:** `poetry install`
- **Spin Up Vector DB (Docker):** `docker-compose up -d`
- **Run Development Server:** `poetry run fastapi dev main.py`
- **Execute Test Suite:** `poetry run pytest`
- **Type Checking (Strict):** `poetry run mypy .`
- **Code Linting & Formatting:** `poetry run ruff check . --fix && poetry run ruff format .`

---

## Code Quality & Engineering Standards
- **Strict Type Hinting:** Every function definition must include explicit Python type hints. No implicit `Any`.
- **Modular RAG Separation:** Absolute separation between `ingestion/`, `retrieval/`, and `generation/`.
- **Fail-Safe Design:** Wrap external I/O operations (Vector DB, LLM calls) in explicit `try-except` blocks with granular logging.
- **Async by Default:** Use asynchronous functions (`async def`) for all network-bound API operations.
