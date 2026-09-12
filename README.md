# dreamgate

**A gate that opens during consolidation to let new memory into the core — without destroying what's already there.**

---

## The Problem

Every learning system today fails one of three tests:

- **RAG** adds memory. It doesn't learn. The model never changes.
- **Fine-tuning** adds learning. It forgets. Old capabilities collapse.
- **LoRA** is cheaper. It still forgets. Just more slowly.

None of them **close the loop**.

A system that learns must:
1. Remember experience (memory)
2. Generalise across contexts (transfer)
3. Update from outcomes (feedback)

**None of the three exist together in any current architecture.**

---

## The Idea

Biologically, memory works in three layers:

| Layer | Role | Timescale |
|-------|------|-----------|
| **Working** | Current context | Seconds |
| **Episodic** | Recent experience | Hours to days |
| **Semantic** | Consolidated knowledge | Years |

Consolidation happens **offline** — during sleep, when the brain replays the day and merges new patterns into old ones without overwriting them.

**Dreamgate is that process, in silicon.**

---

## The Architecture

