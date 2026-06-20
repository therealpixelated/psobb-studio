# Quest Editor — Reference Docs

These are **FACTS-based reference docs** for the psobb-studio quest editor and
quest DSL toolchain. Their content was mined from two community sources:

- **phantasmal-world** (MIT licensed) — the primary clean-room port source for
  file-format codecs and quest data structures.
- **Alisaryn/qedit** (LGPL v2.1) — read **understand-only** for facts and
  behavior. No qedit code is copied or ported into this tree; LGPL is
  incompatible with relicensing here. It is consulted purely to corroborate
  opcode semantics, ID tables, and binary layouts.

newserv (MIT) and live server implementations (PSOBB.IO / Ephinea) are used as
additional cross-checks throughout.

## Index

| Doc | Summary |
|--|--|
| [OPCODE_CROSSVALIDATION.md](OPCODE_CROSSVALIDATION.md) | Our quest opcode table aligns 99.8% with qedit (535→541 opcodes); twelve missing entries are unused BB stubs, eighteen extra are newserv extensions or no-ops, and all arg-type mappings check out — production-ready with optional housekeeping. |
| [REFERENCE_DATA.md](REFERENCE_DATA.md) | Authoritative ID/enum tables for the quest DSL: NPC skins, enemy/monster types, object types, floor/area IDs per episode, fog entries, and quest-register slot conventions (R250=difficulty, R251=mode, R255=quest-clear). |
| [DAT_MAP_FORMAT.md](DAT_MAP_FORMAT.md) | Byte-level spec of the `.dat` quest map format (objects 68B, enemies 72B, events 20B+) and the `.qst` chunked transport container (1048B/1056B chunks), plus requirements for the `quest_map.py` codec. |
| [EDITOR_UX_BLUEPRINT.md](EDITOR_UX_BLUEPRINT.md) | P3 Quest Editor UX and architecture blueprint: panel-by-panel layout, `/api/quest/*` REST routes, language services, 3D placement, and the LLM-assist layer that only ever emits DSL text for deterministic validation. |
