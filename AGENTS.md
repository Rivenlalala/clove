# AI-DLC and Spec-Driven Development

Kiro-style Spec Driven Development implementation on AI-DLC (AI Development Life Cycle)

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- Check `.kiro/specs/` for active specifications
- Use `/kiro-spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, generate responses in English. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## Minimal Workflow
- Phase 0 (optional): `/kiro-steering`, `/kiro-steering-custom`
- Phase 1 (Specification):
  - `/kiro-spec-init "description"`
  - `/kiro-spec-requirements {feature}`
  - `/kiro-validate-gap {feature}` (optional: for existing codebase)
  - `/kiro-spec-design {feature} [-y]`
  - `/kiro-validate-design {feature}` (optional: design review)
  - `/kiro-spec-tasks {feature} [-y]`
- Phase 2 (Implementation): `/kiro-spec-impl {feature} [tasks]`
  - `/kiro-validate-impl {feature}` (optional: after implementation)
- Progress check: `/kiro-spec-status {feature}` (use anytime)

## Development Rules
- 3-phase approval workflow: Requirements → Design → Tasks → Implementation
- Human review required each phase; use `-y` only for intentional fast-track
- Keep steering current and verify alignment with `/kiro-spec-status`
- Follow the user's instructions precisely, and within that scope act autonomously: gather the necessary context and complete the requested work end-to-end in this run, asking questions only when essential information is missing or the instructions are critically ambiguous.

## Steering Configuration
- Load entire `.kiro/steering/` as project memory
- Default files: `product.md`, `tech.md`, `structure.md`
- Custom files are supported (managed via `/kiro-steering-custom`)

---

## Build / Lint / Test Commands

### Package Manager
This project uses **uv** for Python package management and **hatchling** as the build backend.

```bash
# Install dependencies (including dev extras)
uv sync --extra dev

# Install in development mode
make install-dev        # or: uv pip install -e .
```

### Running Tests
```bash
# Run all tests
make test               # or: uv run pytest tests/
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_claude_request_models.py -v

# Run a single test class or function
uv run pytest tests/test_claude_request_models.py::MessagesAPIRequestToolParsingTests -v
uv run pytest tests/test_claude_request_models.py -k "test_accepts_custom_tool_payload" -v

# Run with verbose output and show print statements
uv run pytest tests/ -v -s
```

### Linting & Formatting (Ruff)
```bash
# Lint
uv run ruff check app/ tests/

# Format
uv run ruff format app/ tests/

# Lint with auto-fix
uv run ruff check app/ tests/ --fix
```

### Building
```bash
# Build frontend + wheel
make build

# Build wheel only
make build-wheel        # or: uv run python scripts/build_wheel.py --skip-frontend

# Build frontend only
make build-frontend     # cd front && pnpm install && pnpm run build
```

### Running the App
```bash
make run                # or: uv run python -m app.main
clove                   # if installed
```

### Prerequisites
- `app/static/` must exist (frontend build artifact). Create if missing:
  ```bash
  mkdir -p app/static
  ```
- Python 3.13 (specified in `.python-version`)

---

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local (`app.*`)
- Group imports with a blank line between groups
- Use absolute imports from `app.` (e.g., `from app.core.config import settings`)
- Prefer specific imports over wildcards

### Formatting
- Follow **Ruff** defaults (compatible with Black)
- Line length: default (88 chars)
- Use double quotes for strings
- Trailing commas in multi-line structures

### Types
- Use type hints on all function signatures and class attributes
- Import from `typing`: `Optional`, `List`, `Dict`, `Any`, `Union`, `Literal`
- Use `Optional[T]` instead of `T | None` for consistency with existing code
- Pydantic models use `model_config = ConfigDict(extra="allow")` for flexibility

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `MessagesAPIRequest`, `AppError`)
- **Functions/Methods**: `snake_case` (e.g., `configure_logger`, `parse_comma_separated`)
- **Variables**: `snake_case` (e.g., `api_keys`, `data_folder`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `USER`, `ASSISTANT` in enums)
- **Private methods**: prefix with `_` (e.g., `_json_config_settings`)
- **Pydantic models**: suffix with context (e.g., `*Content`, `*Error`, `*Request`)

### Error Handling
- All exceptions inherit from `AppError` (in `app/core/exceptions.py`)
- `AppError` fields: `error_code` (int), `message_key` (str for i18n), `status_code` (int), `context` (dict), `retryable` (bool)
- Create specific exception classes per error type (e.g., `ClaudeRateLimitedError`, `NoAccountsAvailableError`)
- Error codes follow pattern: `{http_status}{sequence}` (e.g., `429120`, `503100`)
- Message keys use dot notation for i18n: `"section.errorType"` (e.g., `"claudeClient.httpError"`)
- Use `context.copy()` when extending context in subclasses
- Register global exception handler via `app.add_exception_handler(AppError, app_exception_handler)`

### Async / Concurrency
- Use `async/await` throughout for I/O operations
- Background tasks managed via `asyncio.create_task()` with cleanup in lifespan
- Use `@asynccontextmanager` for lifespan management (see `app/main.py`)

### Logging
- Use **loguru** (`from loguru import logger`)
- Configure via `app/utils/logger.py` and `app/utils/content_logger.py`
- Do not use `print()` for output; use `logger.info()`, `logger.error()`, etc.

### Pydantic Models
- Use `Field()` with `default`, `env`, `ge`, `le`, `description` parameters
- Use `@field_validator` for custom validation (e.g., comma-separated parsing)
- Use `@model_validator(mode="after")` for cross-field validation
- Enum classes inherit from `str, Enum` for JSON serialization

### Architecture
- `app/api/` - FastAPI routers and route handlers
- `app/core/` - Core config, exceptions, HTTP client, session management
- `app/models/` - Pydantic request/response models
- `app/processors/` - Request/response processing pipeline
- `app/services/` - Business logic services (accounts, sessions, cache, etc.)
- `app/utils/` - Utility functions (logging, etc.)
- `app/dependencies/` - FastAPI dependency injection

### Tests
- Use `unittest.TestCase` style (not pytest fixtures)
- Test files named `test_*.py` in `tests/` directory
- Test classes named `*Tests` (e.g., `MessagesAPIRequestToolParsingTests`)
- Test methods prefixed with `test_` and descriptive names
- Use `self.assertEqual()`, `self.assertTrue()`, etc.
