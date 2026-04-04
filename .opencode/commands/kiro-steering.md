---
description: Manage .kiro/steering/ as persistent project knowledge
agent: kiro/steering
subtask: true
---

# Kiro Steering Management

## Mode Detection

**Perform detection before invoking Subagent**:

Check `.kiro/steering/` status:
- **Bootstrap Mode**: Empty OR missing core files (product.md, tech.md, structure.md)
- **Sync Mode**: All core files exist

Use Glob to check for existing steering files.

## Subagent Context

Mode: {bootstrap or sync based on detection}

File patterns to read:
- .kiro/steering/*.md (if sync mode)
- .kiro/settings/templates/steering/*.md
- .kiro/settings/rules/steering-principles.md

JIT Strategy: Fetch codebase files when needed, not upfront

## Display Result

Show Subagent summary to user:

### Bootstrap:
- Generated steering files: product.md, tech.md, structure.md
- Review and approve as Source of Truth

### Sync:
- Updated steering files
- Code drift warnings
- Recommendations for custom steering

## Notes

- All `.kiro/steering/*.md` loaded as project memory
- Templates and principles are external for customization
- Focus on patterns, not catalogs
- "Golden Rule": New code following patterns shouldn't require steering updates
- Avoid documenting agent-specific tooling directories (e.g. `.cursor/`, `.gemini/`, `.claude/`)
- `.kiro/settings/` content should NOT be documented in steering files (settings are metadata, not project knowledge)
- Light references to `.kiro/specs/` and `.kiro/steering/` are acceptable; avoid other `.kiro/` directories
