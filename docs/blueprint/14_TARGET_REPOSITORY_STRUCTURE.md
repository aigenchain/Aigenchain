# Target Repository Structure

## Objective

This document defines the long-term repository layout for Aigenchain.

Every refactoring decision must move the project closer to this target structure.

---

# Top Level

```
aigenchain/

core/
runtime/
engine/
services/
routes/
integrations/
mcp_servers/

config/
docs/
static/
tests/

scripts/
docker/

data/
licenses/

app.py
```

---

# Core

Responsible for platform fundamentals.

Examples:

* authentication
* middleware
* database abstraction
* session management
* platform compatibility
* constants

Core must not depend on Engine.

---

# Runtime

Responsible for execution lifecycle.

Submodules:

```
runtime/

lifecycle/
scheduler/
executor/
supervisor/
resource_manager/
```

Responsibilities:

* startup
* shutdown
* orchestration
* background jobs
* scheduling
* monitoring

---

# Engine

Responsible for intelligence.

Submodules:

```
engine/

reasoning/
planning/
context/
memory/
tools/
execution/
```

Responsibilities:

* planning
* reasoning
* context management
* tool orchestration
* autonomous execution

Engine should remain infrastructure-independent.

---

# Services

Business capabilities.

Examples:

```
services/

memory/
search/
research/
youtube/
tts/
stt/
docs/
shell/
hwfit/
```

Each service exposes stable interfaces.

---

# Routes

HTTP API layer only.

Routes should never contain business logic.

Responsibilities:

* validation
* serialization
* request handling

Business execution belongs to services or engine.

---

# Integrations

External platform adapters.

Examples:

* Claude
* Codex
* third-party providers

No core business logic should live here.

---

# MCP Servers

Dedicated MCP implementations.

Independent from runtime orchestration.

---

# Config

Configuration only.

No executable business logic.

---

# Static

Frontend assets.

No backend implementation.

---

# Docs

Single source of architectural truth.

Blueprint takes precedence over implementation.

---

# Tests

Mirror production architecture.

Tests should follow:

```
tests/

engine/
runtime/
services/
routes/
integration/
```

---

# Migration Policy

During migration:

* src may temporarily exist.

Final objective:

* eliminate duplicated implementations.
* migrate functionality into runtime, engine, or services.
* reduce src into compatibility wrappers.
* eventually remove src entirely.

---

# Architectural Principle

Every module must have exactly one responsibility.

No duplicated ownership.

No circular dependencies.

Documentation must always precede implementation.
