# Quest Editor — UX & Architecture Blueprint

**Version:** P3 (Frontend Design Phase)  
**Audience:** Quest DSL implementation team  
**Status:** Specification (implement per the PSPerspectives + REST API pattern)

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Architecture Summary](#architecture-summary)
4. [Frontend: Perspectives & Panels](#frontend-perspectives--panels)
5. [Backend: REST API Routes](#backend-rest-api-routes)
6. [Panel-by-Panel Blueprint](#panel-by-panel-blueprint)
7. [3D Placement View & Selection](#3d-placement-view--selection)
8. [Code Editor & Language Services](#code-editor--language-services)
9. [LLM/AI Layer Integration](#llmai-layer-integration)
10. [Workflow Sketches](#workflow-sketches)
11. [Quality Gates & Testing](#quality-gates--testing)

---

## Overview

The Quest Editor is a **dual-frontend (JS + Godot) companion** to the bytecode compiler stack (Layer 0–1, quest_opcodes.py + quest_asm.py + quest_dsl.py). It provides:

- **Code editing** for DSL (.quest) + assembler (.qasm) with opcode-driven language services
- **3D map visualization** with entity placement (NPCs, objects, enemies, waves)
- **Timeline view** for wave/event orchestration
- **Quest metadata editor** (name, description, floor assignments)
- **Compilation diagnostics** (inline errors, warnings, go-to-label navigation)
- **Optional LLM assist** layer that emits DSL text (never bytecode) for validation

It follows the studio's established **PSPerspectives pattern** (unified viewport + interchangeable panels) and reuses existing infrastructure: the model viewer (Three.js), the REST API gateway, and the panel lifecycle hooks.

---

## Design Principles

1. **Layer the views vertically:** code → assembly → bytecode; users author at the DSL level (optional), compile to assembly for inspection, validate against opcodes.

2. **Reuse existing viewport infrastructure:** The 3D scene (model_viewer.js) already handles Three.js lifecycle, camera controls, and rendering. The Quest Editor "borrows" the canvas via PSPerspectives mount/unmount (same pattern as map_panel.js).

3. **Keep compilation deterministic:** The compiler (`quest_dsl.py` + `quest_asm.py`) is the single source of truth. UI surfaces compile errors, never hides them. Disassembler best-effort, never required.

4. **Persistence: editor state, not artifacts.** Unsaved edits live in browser localStorage (DSL + .dat changes) or in a temporary sidecar until the user explicitly builds/exports. Never auto-save bytecode; always compile on demand.

5. **Language services are optional polish:** Syntax highlighting, hover docs, and go-to-label are driven by `quest_opcodes.py` (facts only). These fail gracefully if the server is slow; they never block editing.

6. **Operator-friendly:** Place/rotate entities with visual feedback, not numeric entry dialogs. Timeline shows wave/event sequence at a glance. Quest properties (name, episode, max players) are editable inline.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                 FRONTEND (JS + Godot)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  PSPerspectives viewport (unified stage + inspector)   │
│  ├─ Quest Editor perspective                           │
│  │  ├─ [toolbar]                                       │
│  │  │  └─ File picker, build/validate, deploy...      │
│  │  │                                                  │
│  │  ├─ [body]                                          │
│  │  │  ├─ Code editor (left pane, ~40% width)         │
│  │  │  │  └─ Monaco/CodeMirror with language services │
│  │  │  │                                               │
│  │  │  ├─ 3D viewport (center, ~50%)                  │
│  │  │  │  ├─ Borrowed Three.js canvas                  │
│  │  │  │  ├─ Entity markers (click-to-place)          │
│  │  │  │  └─ Toolbar: grid, reset camera, layer vis   │
│  │  │  │                                               │
│  │  │  └─ [footer]                                     │
│  │  │     └─ Coord readout, compile status            │
│  │  │                                                  │
│  │  └─ [inspector (right pane, ~15%)]                 │
│  │     ├─ Entity properties (selected entity)          │
│  │     ├─ Quest metadata                               │
│  │     ├─ Wave/event timeline                          │
│  │     └─ Compile diagnostics                          │
│  │                                                     │
│  └─ View toggles:                                     │
│     ├─ DSL editor ↔ Assembly view (bidirectional)    │
│     ├─ 3D placement ↔ Timeline                        │
│     └─ Minimize inspector                             │
│                                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               BACKEND (FastAPI + Python)                │
├─────────────────────────────────────────────────────────┤
│  Compiler layer (formats/quest_*)                      │
│  ├─ quest_opcodes.py    (opcode definitions)          │
│  ├─ quest_bin.py        (.bin/.qst codec)             │
│  ├─ quest_asm.py        (assembler + disassembler)    │
│  ├─ quest_dsl.py        (high-level DSL → asm)        │
│  └─ quest_map.py        (.dat file parsing)           │
│                                                        │
│  REST routes: /api/quest/*                            │
│  ├─ GET  /api/quest/list                              │
│  ├─ GET  /api/quest/<id>         (load .qst)          │
│  ├─ POST /api/quest/<id>/parse   (DSL → AST)          │
│  ├─ POST /api/quest/<id>/compile (compile to .bin)    │
│  ├─ POST /api/quest/<id>/disasm  (bytecode → asm)     │
│  ├─ POST /api/quest/<id>/validate(lint + diagnostics) │
│  ├─ POST /api/quest/<id>/build   (pack .qst)          │
│  ├─ POST /api/quest/<id>/deploy  (to game dir)        │
│  ├─ POST /api/quest/<id>/preview (validate + errors)  │
│  └─ POST /api/quest/<id>/lift    (asm → DSL, best-eff)│
│                                                        │
└─────────────────────────────────────────────────────────┘
```

---

## Frontend: Perspectives & Panels

### Perspective Registration

The Quest Editor registers with `window.PSOPerspectives` (see `perspectives.js`):

```javascript
PSOPerspectives.register("quest-editor", {
  label: "Quest Editor",
  
  // Match logic: score > 0 activates this perspective
  match(entry, fileName) {
    // .quest, .qasm, .qst files
    if (/\.(quest|qasm|qst)$/i.test(fileName)) return 100;
    return 0;
  },
  
  // Mount on stage + inspector elements; set up event listeners
  async mount(stageEl, inspectorEl, ctx) {
    await questEditorMount(stageEl, inspectorEl, ctx);
  },
  
  // Teardown: save edits to localStorage, restore viewport
  async unmount(stageEl, inspectorEl) {
    await questEditorUnmount(stageEl, inspectorEl);
  },
});
```

The perspective shares the **unified stage + inspector** pattern with existing panels (map_panel.js, model_viewer.js, etc.).

### Lifecycle & State

**Stage elements:**
- `#vpStage` — main work area (code + 3D + props)
- `#vpInspector` — side panel (entity details, quest properties, diagnostics)

**Data persistence:**
- Browser localStorage: editor state (open file, viewport camera, last DSL text)
- Server (sidecar JSON): the compiled `.qst` + `.dat` pair
- Temporary compile cache in `/tmp/psoharness_quest_*` (per-session)

**Edits lifecycle:**
1. User types in DSL editor or places entity in 3D view
2. On-keystroke compile → emit diagnostics to inspector
3. Save button → POST to `/api/quest/<id>/build` → receive bytecode + warnings
4. Deploy button → POST to `/api/quest/<id>/deploy` → copy to game data dir

---

## Backend: REST API Routes

All routes under `/api/quest/`:

### Metadata & Discovery

**GET `/api/quest/list`**
- Returns: `{quests: [{id, name, episode, path, mtime}]}`
- Lists all .qst files in the quest data directory

**GET `/api/quest/<id>`**
- Returns: `{id, name, episode, max_players, floor_assignments, qst_path, ...}`
- Loads quest metadata + decompressed `.bin` + `.dat` snapshots

### Compilation & Language Services

**POST `/api/quest/<id>/parse`**
- Body: `{text: "...DSL source..."}`
- Returns: `{ast: {...}, errors: [{line, col, message}]}`
- Parse DSL to AST without code generation (for incremental edits)

**POST `/api/quest/<id>/compile`**
- Body: `{text: "...DSL source...", format: "dsl"|"asm"}`
- Returns: `{bin_b64: "...", bin_size: 123, asm_text: "...", errors: [], warnings: []}`
- Compile DSL or assembler to bytecode; include decompressed binary + human-readable assembly

**POST `/api/quest/<id>/disasm`**
- Body: `{bin_b64: "..."}` (optional; else use current quest)
- Returns: `{asm_text: "...", labels: {addr: name, ...}}`
- Disassemble bytecode to newserv-compatible assembler text

**POST `/api/quest/<id>/validate`**
- Body: `{text: "...source...", format: "dsl"|"asm"|"bin"}`
- Returns: `{valid: true|false, errors: [...], warnings: [...]}`
- Lint + diagnostics WITHOUT code generation (fast, non-blocking)

### Build & Deployment

**POST `/api/quest/<id>/build`**
- Body: `{dsl_text: "...", dat_json: {...}, deploy: false|true}`
- Returns: `{qst_path: "...", qst_size: 123, export_token: "abc", bytecode_md5: "..."}`
- Compile DSL → bytecode, pack .qst (DSL + `.dat`), optionally deploy
- If `deploy: false`, mint an export token for download (see app.js pattern)

**POST `/api/quest/<id>/deploy`**
- Body: `{qst_path: "...", target: "newserv"|"live"}`
- Returns: `{deployed: true, backup_path: "...", target_path: "..."}`
- Atomic: back up, copy, validate checksums

### Decompilation & Reconstruction

**POST `/api/quest/<id>/lift`**
- Body: `{bin_b64: "..."}` (optional)
- Returns: `{dsl_text: "...", lifted_rate: 0.87, unlifted_regions: [...]}`
- Best-effort bytecode → DSL disassembly. Regions that can't be lifted emit raw assembly.
- Lifted rate (% of opcodes matched to high-level constructs) informs the UI
  ("87% lifted to DSL; 13% raw asm")

---

## Panel-by-Panel Blueprint

### 1. Code Editor (Left Pane, ~40% width)

**Features:**
- **Editor type:** Monaco (VS Code API) or CodeMirror (lighter, more portable)
- **Language modes:** quest DSL (.quest), quest assembler (.qasm), output (read-only)
- **Syntax highlighting:** Token-driven, keyed to language
- **Line numbers, code folding, minimap** (standard editor features)

**Language Services (quest DSL):**
- **Completion:** on `{` or `(` or identifier-start, offer:
  - Built-in functions (npc, floor_handler, wave, thread, message, ...)
  - Registers (r0–r255, f0–f255)
  - Opcode names (auto-filled via quest_opcodes.py hover text)
  - Quest flag names (from loaded quest metadata)
- **Hover:** opcode signature + argument types + brief doc
- **Go-to-label:** click label ref → jump to definition (Ctrl+Shift+G)
- **Diagnostics:** on-keystroke compile, surface errors inline (red squiggle) + in gutter
- **Format on save:** optional (Prettier-style, configured per-quest)

**Assembly Mode (.qasm):**
- Syntax highlighting for opcodes (keyed to quest_opcodes.py opcode_name)
- Completion: opcode names, register names, label names
- Hover: argument count, stack/register semantics
- No auto-generate (users edit raw asm intentionally)

**Key Bindings:**
- `Ctrl+Shift+B` — build (compile to bytecode)
- `Ctrl+L` — open Go-to-Label dialog
- `Ctrl+Shift+A` — toggle DSL ↔ Assembly view
- `Ctrl+Shift+D` — disassemble current bytecode to editor

**Output View (read-only pane):**
- Shows compiled bytecode as hex dump + disassembly (side-by-side)
- Synchronized scroll with code editor (jump to bytecode offset of cursor line)
- Copy-to-clipboard for bytecode export

### 2. 3D Viewport (Center, ~50% width)

**Setup:**
- Reuses `window.PSOModelViewer` (the shared Three.js canvas from model_viewer.js)
- On mount: PSPerspectives relocates the canvas into the quest editor's viewport pane
- On unmount: restores canvas to model_viewer's default parent

**Rendering:**
- Load the quest's map/floor (via `/api/map/<quest_id>?floor=N`)
- Render all NPCs, objects, enemies as **colored markers** (cubes or spheres):
  - Red = NPC (enemy spawner)
  - Blue = object (warp, chest, etc.)
  - Green = enemy wave member (placed static enemy)
  - Yellow = waypoint (path node)
- Camera: free-look (mouse drag) + zoom (wheel); optional grid snap

**Interaction:**
- **Click in viewport:** if place-mode is ON (button in toolbar), drop an entity marker
  - Prompts: entity type (npc/object/enemy), ID, rotation
  - Auto-fills default position from viewport ray-cast
- **Drag marker:** move entity in 3D space (X-Z plane, fixed Y)
- **Drag-rotate:** hold Shift + drag to rotate entity around Y axis
- **Double-click marker:** open entity detail editor (right pane)
- **Right-click → Delete:** remove entity from placement list
- **Grid toggle:** snap to 0.5-unit grid (visual + movement)

**Toolbar in viewport:**
- Floor selector (dropdown)
- "Place mode" toggle + entity-type picker (npc/object/enemy)
- Grid toggle
- Reset camera (home view)
- Snap settings (distance: 0.1 / 0.5 / 1.0)

**Coordinate display (footer):**
- Mouse position: `(X: 12.5, Y: 0.0, Z: -8.3)`
- Selected entity: `ID: 0x1A, Section: 5, Rot: 45°`

### 3. Wave/Event Timeline (Bottom-right inspector expansion)

**Layout:**
- Horizontal timeline bar (0ms to quest completion)
- Rows for each wave/event (mouse-scrollable)
- Each row shows: wave number, spawn condition, enemy type, time range

**Interactions:**
- Click on a timeline block → select wave, open wave editor
- Drag timeline block → reschedule wave (if quest uses time-based triggers)
- Right-click → edit/delete wave
- Toolbar: "Add wave", "Auto-sequence" (layout waves by floor/section)

**Data bound to:**
- Quest metadata: floor_assignments, wave definitions
- `.dat` MapFile: enemy sets, wave event tables

### 4. Entity Inspector (Right Pane, ~15% width)

**Tabs:**

**Tab: Entity Properties**
- Shows selected entity from 3D view (or sidebar list)
- Fields:
  - **Type** (NPC, object, enemy): dropdown
  - **ID** (0x00–0xFF): spinner or hex input
  - **Skin** (variant): linked to database (qedit.info IDs)
  - **Section** (0–9): spinner
  - **Position** (X, Y, Z): float inputs (sync with 3D view on change)
  - **Rotation** (0–360°): slider
  - **Params** (type-specific, e.g. chest item ID): expandable
- Actions: copy entity, delete, duplicate, focus in 3D view

**Tab: Quest Metadata**
- **Name**: text input
- **Episode** (1=Forest, 2=Caves, 3=Mines, 4=Ruins): dropdown
- **Max players** (1–4): spinner
- **Joinable**: checkbox
- **Floor assignments** (per floor: object set ID, NPC set ID, enemy set ID): table
- **Short description**: text area
- **Long description**: text area

**Tab: Compile Diagnostics**
- Real-time list of errors + warnings from last compile
- Click error → jump to source line in code editor + highlight
- Severity color-coded (red=error, yellow=warning, blue=info)

**Tab: Entity List**
- Tree view of all placed entities (grouped by floor / type)
- Search box (filter by ID, name, section)
- Column: type, ID, name, position, actions (edit/delete/focus)

### 5. Toolbar (Top of Stage)

- **File picker**: dropdown, shows recent quests + "New quest"
- **Build** button: compile DSL → bytecode (Ctrl+Shift+B)
- **Deploy** button: save to game data dir
- **Preview** button: parse + validate, show diagnostics
- **Undo/Redo**: standard (replay entity placements + code edits)
- **View toggles**:
  - DSL ↔ Assembly (switches code editor mode)
  - 3D view ↔ Timeline (swaps inspector view)
  - Minimize inspector (collapse to icon)

---

## 3D Placement View & Selection

### Entity Placement Workflow

1. **Initiate placement:**
   - Click toolbar "Place mode" + pick entity type (NPC, object, enemy)
   - Cursor changes (crosshair)
   - 3D view shows a "ghost" preview entity under cursor

2. **Place entity:**
   - Click in viewport → position locked
   - Dialog pops: "New NPC" form
     - ID: text/spinner (pre-filled from hovered database)
     - Skin: dropdown (if NPC, list PSOBB NPC skins)
     - Section: spinner (default from placement options)
     - Rotation: slider (0–360°)
     - [Place] [Cancel]

3. **Entity appears in viewport** (marker + label)
4. **Adjust if needed:**
   - Drag marker to move
   - Shift+drag to rotate
   - Double-click → open detail editor

### Selection & Multi-Select

- **Click marker** → single-select (highlight, show in inspector)
- **Ctrl+Click** → toggle select (accumulate)
- **Shift+Click** → range-select (from last to this)
- **Ctrl+A** → select all on current floor
- **Double-click marker** → focus inspector on entity (open properties tab)
- **Right-click marker** → context menu (Edit, Delete, Duplicate, Focus camera)

### Undo/Redo

- Each entity placement, rotation, property edit is an undoable action
- Undo stack: max 50 entries
- On save/build, snapshot the undo stack to localStorage

---

## Code Editor & Language Services

### Syntax & Highlighting

**DSL example (.quest file):**
```quest
// Quest metadata (optional, compiled into .bin header)
quest {
  name: "The First Quest"
  episode: 1
  max_players: 4
  difficulty: 7 // 0-9
}

// Define a thread (runs when quest starts)
thread entry {
  sync  // yield once (mandatory first instruction in any thread)
  
  // Show a message to all players
  window_msg "Welcome, adventurers!"
  
  // NPCs (furniture; not combatants)
  npc {
    id: 0x42
    skin: "Paganini"  // NPC skin name
    floor: 0  // Pioneer 2
    section: 5
    pos: (100, 0, -50)
    dir: 45  // degrees
    dialogue {
      on_talk: "greet_dialog"
    }
  }
  
  // Floor handler (runs when player enters floor 1)
  floor_handler 1 {
    message "Floor 1 unlocked!"
  }
  
  // Spawn a wave when player reaches section 8
  wave {
    trigger: on_enter_section 8
    enemies: [0x4A, 0x4B, 0x4C]  // enemy IDs
    spawn_pos: (50, 0, 0)
  }
  
  // Menu/dialogue example
  on_message "greet_dialog" {
    choose {
      option "Hi" -> {
        message "Hello!"
        quest_flag_set 0x01
      }
      option "Bye" -> {
        message "See you!"
      }
    }
  }
}
```

**Tokens:** keyword (thread, npc, wave, ...), identifier, number, string, operator, comment

### Language Server Features

#### Completion (Ctrl+Space)

- **On keyword start** (e.g., `npc {`):
  - Show all builtin blocks: npc, floor_handler, wave, thread, message, ...
  - Each completion includes snippet (auto-indent, auto-close braces)

- **On register reference** (e.g., `r1`):
  - Suggest r0–r255, f0–f255
  - Show register aliases (r250=slot, r251=difficulty, r255=success)

- **On opcode name** (raw asm mode):
  - Full opcode list from quest_opcodes.py
  - Signature + doc in completion details

#### Hover

- **On opcode name (asm mode):**
  ```
  jmp_eq
  ─────
  Branch if equal. Syntax: jmp_eq <reg_a> <reg_b> <label>
  Pops <reg_a>, <reg_b> from arg stack; if equal, branches to <label>.
  ```

- **On identifier (DSL mode):**
  - Show definition location (file + line)
  - For opcodes, show parameter types

#### Diagnostics

- **On-keystroke** (debounced 500ms):
  - Quick parse → catch syntax errors
  - Emit squiggle + message in gutter
  - Example: `undefined identifier: 'greet_dialog'` (red at line 45)

- **On Save** (full compile):
  - Full type-check + semantic analysis
  - Surface bytecode size, label count, estimated execution

#### Go-to-Label (Ctrl+Shift+G or click)

- Dialog: "Go to label"
- Type label name → fuzzy search through all defined labels
- Click → jump to definition in editor

### Assembly-Mode Editor

When user clicks "View as Assembly" (Ctrl+Shift+A), the code editor switches to show the compiled `.qasm` text:

```asm
; quest: "The First Quest"
; episode: 1
; compiled: 2026-06-20 14:32:18 UTC

  ; Entry thread
  .label entry
  sync 0x02
  
  ; window_msg call with string push
  arg_push_s "Welcome, adventurers!"
  window_msg 0x10
  
  ; NPC placement (map data, not script)
  ; [NPC 0x42 on floor 0, section 5, pos (100, 0, -50)]
  
  ; floor_handler on entry
  .label floor_1_handler
  arg_push_s "Floor 1 unlocked!"
  message 0x08
  ret
```

Language services apply here too (opcode completion, hover docs, etc.).

---

## LLM/AI Layer Integration

**Optional Layer 2 feature** (implement after Layer 0–1 bytecode is proven).

### UI: "Describe the Quest" Panel

In the inspector, add a card:
```
┌─────────────────────────────────────────┐
│  Quest Idea (AI Assist)                 │
├─────────────────────────────────────────┤
│ [Text area]                             │
│ "Create a simple quest where..."        │
│                                         │
│ [Generate]  [Clear]                    │
│                                         │
│ [⟳ Generating...] (status)             │
│                                         │
│ Compiled to DSL: 156 lines              │
│ Errors: 0 | Warnings: 2                │
└─────────────────────────────────────────┘
```

### Workflow

1. User types a natural-language description
2. Click [Generate]
3. Frontend: POST to `/api/aigen` (or similar AI provider route)
   - Prompt includes quest context (loaded quest metadata + opcode reference)
   - Explicitly request DSL output format
4. Server: call LLM, parse response
5. Return: `{dsl_text: "...", confidence: 0.85, warnings: [...]}`
6. Frontend:
   - Insert generated DSL into code editor
   - Compile it (POST `/api/quest/<id>/compile`)
   - Show compile result (errors/warnings) immediately
   - If errors, highlight in editor + inspector diagnostics tab

### Safety Guarantees

- **Model never emits bytecode.** Always DSL or assembly text.
- **Deterministic validation.** Compiler is the hard guardrail; rejected text is shown to user with precise error.
- **No silent hallucination.** If compilation fails, error is surfaced prominently (not hidden in logs).

---

## Workflow Sketches

### Workflow A: Authoring a Quest from Scratch (DSL)

1. Open Quest Editor → "New Quest"
2. Fill out metadata (name, episode, max players)
3. In code editor, write DSL (`thread entry { ... }`)
4. On keystroke, diagnostics appear in inspector (errors/warnings)
5. Click [Build] → compile DSL → bytecode
6. Click [Preview] → show compiled asm in side-by-side view
7. Adjust map placement in 3D view (add NPCs, enemies, waypoints)
8. Click [Deploy] → pack .qst, copy to game data dir
9. In-game: launch quest from counter, verify

### Workflow B: Porting an Existing Quest (Reverse-Engineer)

1. Upload .qst file → load into editor
2. Disassemble to ASM (POST `/api/quest/<id>/disasm`)
3. Show bytecode in output view (hex + asm, synchronized scroll)
4. Lift to DSL (best-effort, POST `/api/quest/<id>/lift`)
5. If lifted successfully:
   - Show DSL in code editor (editable)
   - Compiler validates (should be byte-identical round-trip)
6. If partially lifted:
   - Show DSL + raw asm inline (delineated)
   - Author can manually clean up unlifted sections
7. Save revised DSL → re-compile → verify byte-match

### Workflow C: Iterative Tuning (Place → Test → Refine)

1. Load existing quest in editor
2. In 3D view: click [Place mode], add a new NPC
3. Press [Build] → compile (DSL unchanged)
4. In-game: walk to NPC, confirm placement
5. Back to editor: adjust NPC position via inspector (drag in 3D or numeric input)
6. [Build] again → test
7. Repeat until satisfied
8. [Deploy] when done

---

## Quality Gates & Testing

### Frontend Testing

1. **Smoke test (app.js loads):**
   - Quest Editor perspective registers without error
   - Mount/unmount cycle succeeds (no exceptions)

2. **Unit tests (quest_opcodes.py fixtures):**
   - Completion list includes all opcode names
   - Hover doc retrieves correct signature
   - Go-to-label finds matching labels

3. **Integration tests (compile loop):**
   - Edit DSL → keystroke compile → diagnostic appears ✓
   - Edit assembly → validate → error surfaced ✓
   - Disassemble bytecode → edit → recompile → byte-match ✓

### End-to-End Tests (Live Proof, §5 of autonomous prompt)

1. **Author a minimal quest in DSL:**
   ```quest
   thread entry {
     sync
     window_msg "Hello"
   }
   ```

2. **Compile + build .qst**
3. **Deploy to newserv**
4. **Launch in psoharness client**
5. **Screenshot: confirm message appears**
6. **Log evidence** (screenshot + compile diagnostics)

### Acceptance Criteria for P3

- [ ] Quest Editor perspective mounts/unmounts cleanly
- [ ] Code editor loads with syntax highlighting (DSL + ASM modes)
- [ ] Language services (completion, hover, diagnostics) active on demo opcodes
- [ ] 3D view loads map, renders entity markers (click-to-place works)
- [ ] Entity detail editor (right pane) reflects 3D changes
- [ ] Build button compiles DSL → bytecode (via quest_dsl.py + quest_asm.py)
- [ ] Diagnostics tab surfaces compile errors with line/col precision
- [ ] Deploy button packages .qst + copies to game dir
- [ ] Disassemble view shows bytecode + synchronized asm
- [ ] Lift (bytecode → DSL) shows % confidence + unlifted regions
- [ ] localStorage persists editor state across reload
- [ ] No console errors on perspective mount/unmount

---

## Implementation Checklist

### Backend (Routes + Compiler Wiring)

- [ ] Add `/api/quest/list` endpoint (scan quest data dir)
- [ ] Add `/api/quest/<id>` endpoint (load .qst metadata)
- [ ] Add `/api/quest/<id>/parse` (DSL → AST, no codegen)
- [ ] Add `/api/quest/<id>/compile` (DSL/ASM → .bin)
- [ ] Add `/api/quest/<id>/disasm` (bytecode → ASM)
- [ ] Add `/api/quest/<id>/validate` (lint without codegen)
- [ ] Add `/api/quest/<id>/build` (compile + pack .qst)
- [ ] Add `/api/quest/<id>/deploy` (atomic copy to game dir)
- [ ] Add `/api/quest/<id>/lift` (bytecode → DSL, best-effort)
- [ ] Quest opcodes loaded at startup (quest_opcodes.py → JSON cache)

### Frontend (JS Perspective)

- [ ] Register quest-editor perspective with PSPerspectives
- [ ] mount() routine: build DOM, init editor + 3D view
- [ ] Code editor: Monaco or CodeMirror with quest language support
- [ ] Syntax highlighting rules (DSL + ASM keywords)
- [ ] Language server client (completion + hover + go-to-label)
- [ ] 3D viewport: relocate model viewer canvas, set up event handlers
- [ ] Entity marker rendering (colored cubes by type)
- [ ] Click-to-place workflow (place-mode toggle → entity creation)
- [ ] Drag/rotate/delete entity interactions
- [ ] Inspector tabs: entity props, metadata, diagnostics, timeline
- [ ] Compile button → POST /api/quest/<id>/compile
- [ ] Deploy button → POST /api/quest/<id>/deploy
- [ ] Output view: hex dump + disassembly (synchronized)
- [ ] localStorage: persist editor state (DSL text, camera, selections)

### Tests

- [ ] Round-trip test: DSL → compile → disasm → matches reference asm
- [ ] Opcode coverage: all quest_opcodes.py entries exercised in fixtures
- [ ] Place-and-move: entity lifecycle (create, move, rotate, delete) end-to-end
- [ ] Compile diagnostics: error position correct (line/col match source)
- [ ] Lift rate: heuristic validates against bytecode corpus

---

## Godot Frontend (Parallel Implementation)

The Godot 4.6.2 frontend mirrors the JS layout:

```
┌─ MainScene
│  ├─ ViewportContainer (central stage)
│  │  ├─ SubViewport (3D world)
│  │  └─ CodeEdit panel (left, split)
│  │
│  ├─ VBoxContainer (inspector, right, split)
│  │  ├─ TabContainer
│  │  │  ├─ Tab: Entity Properties (ItemList + PanelContainer)
│  │  │  ├─ Tab: Metadata (LineEdit + TextEdit)
│  │  │  ├─ Tab: Diagnostics (RichTextLabel)
│  │  │  └─ Tab: Timeline (HBoxContainer, wave blocks)
│  │  │
│  │  └─ HBoxContainer (toolbar: Place, Delete, Build, Deploy)
│  │
│  └─ HTTPClient (backend REST calls)
```

Language services via LSP client (optional; can ship without hover/completion in Godot MVP).

---

## Summary

The Quest Editor blueprint integrates tightly with the studio's existing **PSPerspectives + REST API architecture**, minimizing new infrastructure. It layers the DSL → assembly → bytecode hierarchy visually (code → output → 3D placement), with language services and optional LLM assist providing quality-of-life features that never bypass the deterministic compiler gate.

The backend is straightforward: wrap quest_opcodes.py + quest_dsl.py + quest_asm.py behind `/api/quest/*` routes. The frontend reuses the unified viewport + inspector model from map_panel.js and model_viewer.js, reducing duplication.

Implement backend + JS frontend first; Godot can follow once the core workflow is validated.
