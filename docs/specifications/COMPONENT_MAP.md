# Component Map

## Core Components

### Interface

* Desktop UI
* Web UI
* API Gateway

---

### Agent

* Planner
* Executor
* Coordinator
* Session Manager

---

### Intelligence

* Local Model Manager
* Context Engine
* Memory Manager
* Knowledge Retrieval

---

### Runtime

* AigenRuntime
* Scheduler
* Resource Monitor
* Process Controller

---

### Storage

* PostgreSQL
* Vector Storage
* Local Files
* Configuration Store

---

### Security

* Authentication
* Authorization
* Encryption
* Secret Management
* Audit Logging

---

### Production

* Code Generator
* Testing Engine
* Documentation Generator
* Deployment Generator

---

# Dependency Principle

Dependencies should flow downward only.

Higher-level components must not be tightly coupled to infrastructure implementations.
