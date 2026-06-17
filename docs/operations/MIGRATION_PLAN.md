# Aigenchain Migration Plan

## Purpose

This document defines the long-term migration strategy for evolving Aigenchain into a fully modular AI-native platform.

Migration must prioritize stability, backward compatibility, and incremental refactoring.

---

# Current Assessment

Current repository contains legacy architecture mixed with newer modular components.

Several implementations exist in duplicated locations and should eventually converge into a single authoritative implementation.

Examples include:

* Search
* Memory
* Research
* YouTube
* Runtime utilities

No destructive migration should occur without validation.

---

# Migration Principles

1. Refactor incrementally.

2. Never break production functionality.

3. Prefer moving responsibilities instead of copying code.

4. Maintain compatibility during transition.

5. One authoritative implementation per subsystem.

6. Documentation precedes implementation.

---

# Phase 0

## Blueprint Freeze

Objectives

* Freeze governance.
* Freeze platform strategy.
* Freeze architecture.
* Freeze engineering principles.

Status

Completed.

---

# Phase 1

## Documentation Foundation

Objectives

* Organize blueprint.
* Organize specifications.
* Organize ADR.
* Organize references.

Status

Completed.

---

# Phase 2

## Search Consolidation

Target

Replace duplicated implementations.

Future source of truth:

services/search/

Legacy modules under src/search become compatibility wrappers before removal.

---

# Phase 3

## Memory Consolidation

Target

Unify:

* memory.py
* memory_vector.py
* memory_provider.py

into:

services/memory/

Memory subsystem should expose a stable service interface.

---

# Phase 4

## Research Consolidation

Target

Move all research execution logic into:

services/research/

Eliminate duplicate handlers.

---

# Phase 5

## Runtime Separation

Introduce:

runtime/

Future responsibilities:

* execution
* orchestration
* scheduler
* lifecycle
* supervision

This becomes the foundation of AigenRuntime.

---

# Phase 6

## Engine Separation

Introduce:

engine/

Responsibilities:

* reasoning
* planning
* context
* inference
* orchestration

The engine must remain independent from infrastructure.

---

# Phase 7

## Infrastructure Cleanup

Separate infrastructure responsibilities:

* storage
* postgres
* vector database
* filesystem
* logging
* configuration

Infrastructure should become replaceable.

---

# Phase 8

## Legacy Deprecation

After validation:

* remove obsolete src modules
* remove compatibility layers
* simplify imports

No duplicated implementations should remain.

---

# Phase 9

## Native Runtime Evolution

Long-term objective:

Aigenchain evolves from an application into its own AI-native execution platform.

Docker, Ollama, and external runtimes become pluggable adapters instead of architectural dependencies.

Internal orchestration is handled by AigenRuntime.

---

# Success Criteria

Migration is considered complete when:

* no duplicated subsystem exists
* architecture is modular
* runtime is internally orchestrated
* infrastructure is replaceable
* business logic is isolated
* documentation matches implementation
