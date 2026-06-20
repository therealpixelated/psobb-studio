# PSO Quest Map Data Format (.DAT / .MAP)

## Overview

The `.dat` file contains static map data for a PSO quest: object placements, NPC definitions, enemy spawns, and wave/event triggers. It is one of three components packed into a `.qst` container (alongside `.bin` bytecode and metadata), and is PRS-compressed on disk.

This document specifies:
1. The binary on-disk structure (decompressed)
2. The object, NPC, and enemy entity layouts
3. Event and wave/trigger semantics
4. Multi-floor quest structure
5. The `.qst` container format (how `.dat` + `.bin` are transported)
6. Requirements for `formats/quest_map.py` codec

**Sources:**
- **newserv** (MIT): `src/Map.hh`, `src/Map.cc`, `src/Quest.cc`, `src/QuestScript.cc` — canonical byte layouts and round-trip parsing
- **phantasmal-world** (MIT): `psolib/fileFormats/quest/Dat.kt`, `psolib/fileFormats/quest/Qst.kt` — working codec reference
- **qedit** (closed, understand-only): format validation and semantic corroboration

---

## 1. DECOMPRESSED .DAT FILE STRUCTURE

### Global Structure

A quest `.dat` file is a **sequence of sections**, each prefixed with a 16-byte header. A section can be:
1. **Object Sets** (type 1)
2. **Enemy Sets** (type 2)
3. **Events** (type 3)
4. **Random Enemy Locations** (type 4, challenge mode only)
5. **Random Enemy Definitions** (type 5, challenge mode only)
6. **End marker** (type 0)

Each section can span multiple floors (one or more areas per floor), and sections are **grouped by floor**, with floors appearing in ascending order within each section type.

```
[Section Header] [Section Data]
[Section Header] [Section Data]
...
[Section Header=0x00] [Empty]  ← End of file marker
```

### Section Header (16 bytes, Little-Endian)

```c
struct SectionHeader {
  le_uint32_t type;         // 0=END, 1=OBJECT_SETS, 2=ENEMY_SETS, 3=EVENTS, 4=RANDOM_LOCATIONS, 5=RANDOM_DEFS
  le_uint32_t section_size; // Total size including this header (16 bytes)
  le_uint32_t floor;        // Floor index (cumulative: 0-17; Ep2 starts at floor 18, etc.)
  le_uint32_t data_size;    // Size of section data (section_size - 16)
};
```

**Note:** Floors are **cumulative across episodes**:
- Episode 1: floors 0-17
- Episode 2: floors 18-35 (further floor 16-17 are special: 0xFF = unused in Episode 4 Pioneer 2)
- Episode 3: floors 0-1, rest unused
- Episode 4: floors 0-10, rest unused

---

## 2. OBJECT SETS (Section Type 1)

### Layout

Objects in a section are stored as a **continuous array** of `ObjectSetEntry` structs. Multiple areas can coexist in one section; the `floor` field in the header identifies which floor, and the `areaId` field in each entry identifies the area within that floor.

### ObjectSetEntry (68 bytes, Little-Endian)

```c
struct ObjectSetEntry {
  /* 00 */ le_uint16_t base_type;       // Object type ID (e.g., 0x0001 = door, 0x0002 = box, etc.)
  /* 02 */ le_uint16_t set_flags;       // Runtime state flags; unused in DAT (populated by engine at load)
  /* 04 */ le_uint16_t index;           // Runtime index; unused in DAT
  /* 06 */ le_uint16_t floor;           // Floor ID (redundant with section header)
  /* 08 */ le_uint16_t entity_id;       // = index + 0x4000 (computed at runtime; unused in DAT)
  /* 0A */ le_uint16_t group;           // Placement group (used for area layout grouping)
  /* 0C */ le_uint16_t room;            // Room index within the floor
  /* 0E */ le_uint16_t unknown_a3;      // Reserved/unknown (usually 0)
  
  /* 10 */ le_float   pos_x;            // Position relative to room origin (X)
  /* 14 */ le_float   pos_y;            // Position relative to room origin (Y)
  /* 18 */ le_float   pos_z;            // Position relative to room origin (Z)
  
  /* 1C */ le_uint16_t angle_x;         // Rotation around X axis (16-bit: 0 = 0°, 0xFFFF ≈ 360°)
  /* 1E */ le_uint16_t angle_y;         // Rotation around Y axis
  /* 20 */ le_uint16_t angle_z;         // Rotation around Z axis
  /* 22 */ le_uint16_t _pad1;           // Padding/alignment
  
  /* 24 */ le_float   param1;           // Object-specific float parameter
  /* 28 */ le_float   param2;           // Object-specific float parameter
  /* 2C */ le_float   param3;           // Object-specific float parameter
  /* 30 */ le_int32_t param4;           // Object-specific int parameter
  /* 34 */ le_int32_t param5;           // Object-specific int parameter
  /* 38 */ le_int32_t param6;           // Object-specific int parameter
  /* 3C */ le_uint32_t unused_obj_ptr;  // Reserved for client memory pointer; unused in file
  /* 40 */                              // Total: 0x44 = 68 bytes
};
```

### Key Details

- **Position:** Relative to the **room's local origin** (not world coords). The room's position and rotation are stored separately in `.rel` area-layout files.
- **Rotation angles:** 16-bit values where `0xFFFF` represents ~360° (or just under; exact conversion is `angle_radians = (angle_u16 / 65536.0) * 2π`).
- **Parameters:** Type-specific. Common examples:
  - Doors: `param1` = target floor, `param4` = door switch flag ID
  - Boxes: `param1` = box type (health, monomate, etc.)
  - Walls: `param1` = wall type ID
- **group field:** Used to batch objects for area layouts (e.g., all doors in group 0 form one logical unit).

**Important:** The **`index` field is NOT the storage index** — it is assigned at runtime. The `entity_id = index + 0x4000` is also runtime-computed and reflects the global object slot ID in the current game session. **When parsing, ignore `index` and `entity_id`; assign them during engine load.**

---

## 3. ENEMY SETS (Section Type 2)

### Layout

Similar to objects: a continuous array of `EnemySetEntry` structs. The `wave_number` and `wave_number2` fields link enemies to **event triggers**.

### EnemySetEntry (72 bytes, Little-Endian)

```c
struct EnemySetEntry {
  /* 00 */ le_uint16_t base_type;       // Enemy type ID (e.g., 0x0001 = Hildebear, etc.)
  /* 02 */ le_uint16_t set_flags;       // Runtime state flags; unused in DAT
  /* 04 */ le_uint16_t index;           // Runtime index; unused in DAT
  /* 06 */ le_uint16_t num_children;    // Number of child enemies (0 = use default from constructor table)
  /* 08 */ le_uint16_t floor;           // Floor ID (redundant with section header)
  /* 0A */ le_uint16_t entity_id;       // = index + 0x1000 (runtime-computed; unused in DAT)
  /* 0C */ le_uint16_t room;            // Room index
  /* 0E */ le_uint16_t wave_number;     // Wave ID for event triggering (primary)
  /* 10 */ le_uint16_t wave_number2;    // Wave ID variant or secondary wave (rarely used)
  /* 12 */ le_uint16_t unknown_a1;      // Reserved/unknown (usually 0)
  
  /* 14 */ le_float   pos_x;            // Position relative to room origin (X)
  /* 18 */ le_float   pos_y;            // Position relative to room origin (Y)
  /* 1C */ le_float   pos_z;            // Position relative to room origin (Z)
  
  /* 24 */ le_uint16_t angle_x;         // Rotation around X axis
  /* 26 */ le_uint16_t angle_y;         // Rotation around Y axis
  /* 28 */ le_uint16_t angle_z;         // Rotation around Z axis
  /* 2A */ le_uint16_t _pad1;           // Padding/alignment
  
  /* 2C */ le_float   param1;           // Enemy-specific float parameter
  /* 30 */ le_float   param2;           // Enemy-specific float parameter
  /* 34 */ le_float   param3;           // Enemy-specific float parameter
  /* 38 */ le_float   param4;           // Enemy-specific float parameter
  /* 3C */ le_float   param5;           // Enemy-specific float parameter
  /* 40 */ le_int16_t param6;           // Enemy-specific int parameter
  /* 42 */ le_int16_t param7;           // Enemy-specific int parameter
  /* 44 */ le_uint32_t unused_obj_ptr;  // Reserved for client memory pointer; unused in file
  /* 48 */                              // Total: 0x48 = 72 bytes
};
```

### Key Details

- **Wave coupling:** An `Event` (see §4) specifies a `room` and `wave_number`. When that event is triggered, the engine spawns all enemies whose `room` and `wave_number` match.
- **num_children:** If 0, the engine consults a static constructor table to determine how many child enemies this spawn-point creates (e.g., one Hildebear typically spawns with 0 children, but some rare variants spawn with multiple). If non-zero, overrides the default.
- **wave_number2:** A secondary or alternate wave ID, rarely used; most quests leave it at 0 or equal to `wave_number`.

---

## 4. EVENTS (Section Type 3)

### Layout

The **Events section** is more complex. It consists of three subsections:
1. **Event Header** (16 bytes)
2. **Event Entry Array** (size depends on format + event count)
3. **Action Stream** (variable size; bytecode-like script for post-wave actions)

### Event Section Header (16 bytes)

```c
struct EventsSectionHeader {
  /* 00 */ le_uint32_t action_stream_offset;  // Absolute offset from start of this header to the action stream
  /* 04 */ le_uint32_t entries_offset;        // Absolute offset from start of this header to first event entry
  /* 08 */ le_uint32_t entry_count;           // Number of event entries
  /* 0C */ be_uint32_t format;                // 0x00000000 (old) or 0x65767432 ('evt2' in ASCII = challenge mode)
};
```

**Format variants:**
- **format == 0x00:** Event1Entry (old format; standard quests)
- **format == 0x65767432 ('evt2'):** Event2Entry (challenge mode; rare)

### Event1Entry (20 bytes, Little-Endian) — Standard Format

```c
struct Event1Entry {
  /* 00 */ le_uint32_t event_id;              // Event identifier (can be non-unique; all events with same ID trigger together)
  /* 04 */ le_uint16_t flags;                 // Runtime state flags (unused in DAT; engine sets these at load)
  /* 06 */ le_uint16_t event_type;            // Event constructor type (0 = no-op, 1 = spawn wave, other values undefined behavior)
  /* 08 */ le_uint16_t room;                  // Room index to match against EnemySetEntry.room
  /* 0A */ le_uint16_t wave_number;           // Wave ID to match against EnemySetEntry.wave_number
  /* 0C */ le_uint32_t delay;                 // Frames to wait after trigger before spawning (1 frame = ~1/30th sec)
  /* 10 */ le_uint32_t action_stream_offset;  // Relative offset into the action stream (from action_stream_offset in header)
  /* 14 */                                     // Total: 20 bytes
};
```

### Event2Entry (24 bytes, Little-Endian) — Challenge Mode Format

```c
struct Event2Entry {
  /* 00 */ le_uint32_t event_id;
  /* 04 */ le_uint16_t flags;
  /* 06 */ le_uint16_t event_type;
  /* 08 */ le_uint16_t room;
  /* 0A */ le_uint16_t wave_number;
  /* 0C */ le_uint16_t min_delay;              // Minimum delay (randomized)
  /* 0E */ le_uint16_t max_delay;              // Maximum delay (randomized)
  /* 10 */ uint8_t     min_enemies;            // Minimum enemies to spawn (randomized)
  /* 11 */ uint8_t     max_enemies;            // Maximum enemies to spawn (randomized)
  /* 12 */ le_uint16_t max_waves;              // Max number of this event type to trigger
  /* 14 */ le_uint32_t action_stream_offset;
  /* 18 */                                     // Total: 24 bytes
};
```

### Action Stream (Variable, Little-Endian)

The action stream is a **bytecode-like script** executed when all enemies in a wave are killed. It is NOT the same as the main quest `.bin` bytecode; it's a simple opcode sequence for post-wave actions.

**Action codes:**
- `0x08` — Spawn NPCs: reads `(le_uint16_t sectionId, le_uint16_t appearFlag)` (total 4 bytes)
- `0x0A` — Unlock door: reads `(le_uint16_t doorId)` (total 2 bytes)
- `0x0B` — Lock door: reads `(le_uint16_t doorId)` (total 2 bytes)
- `0x0C` — Trigger event: reads `(le_uint32_t eventId)` (total 4 bytes)
- `0x01` — End of actions (sentinel)
- `0xFF` — Padding/null byte (ignored until end)

**Example action stream (hex):**
```
08 00 00 00 00        ← Spawn NPCs, section 0, appear flag 0
0A 04 00              ← Unlock door ID 4
01                    ← End
FF FF FF FF           ← Padding to 4-byte boundary
```

---

## 5. RANDOM ENEMY LOCATIONS & DEFINITIONS (Section Types 4 & 5)

These sections are used only in **challenge mode** quests and enable randomized enemy placement.

### RandomEnemyLocationsHeader (12 bytes)

```c
struct RandomEnemyLocationsHeader {
  /* 00 */ le_uint32_t room_table_offset;     // Offset to RandomEnemyRoom array (from start of header)
  /* 04 */ le_uint32_t entries_offset;        // Offset to RandomEnemyLocation array (from start of header)
  /* 08 */ le_uint32_t num_rooms;             // Number of room entries
  /* 0C */
};
```

### RandomEnemyRoom (8 bytes)

```c
struct RandomEnemyRoom {
  /* 00 */ le_uint16_t room_id;               // Room index
  /* 02 */ le_uint16_t count;                 // Number of location entries for this room
  /* 04 */ le_uint32_t offset;                // Byte offset into the location entries table
  /* 08 */
};
```

### RandomEnemyLocation (28 bytes)

```c
struct RandomEnemyLocation {
  /* 00 */ le_float   pos_x;                  // Position X (relative to room)
  /* 04 */ le_float   pos_y;                  // Position Y
  /* 08 */ le_float   pos_z;                  // Position Z
  /* 0C */ le_uint16_t angle_x;               // Rotation X
  /* 0E */ le_uint16_t angle_y;               // Rotation Y
  /* 10 */ le_uint16_t angle_z;               // Rotation Z
  /* 12 */ le_uint16_t _pad1;                 // Padding
  /* 14 */ le_uint16_t unknown_a9;            // Reserved/unknown
  /* 16 */ le_uint16_t unknown_a10;           // Reserved/unknown
  /* 18 */                                     // Total: 0x1C = 28 bytes
};
```

### RandomEnemyDefinitionsHeader (16 bytes)

```c
struct RandomEnemyDefinitionsHeader {
  /* 00 */ le_uint32_t entries_offset;            // Offset to RandomEnemyDefinition array
  /* 04 */ le_uint32_t weight_entries_offset;    // Offset to RandomEnemyWeight array
  /* 08 */ le_uint32_t entry_count;              // Number of definitions
  /* 0C */ le_uint32_t weight_entry_count;       // Number of weight entries
  /* 10 */
};
```

### RandomEnemyDefinition (32 bytes)

```c
struct RandomEnemyDefinition {
  /* 00 */ le_float   param1;
  /* 04 */ le_float   param2;
  /* 08 */ le_float   param3;
  /* 0C */ le_float   param4;
  /* 10 */ le_float   param5;
  /* 14 */ le_int16_t param7;       // Note: order is reversed vs EnemySetEntry
  /* 16 */ le_int16_t param6;       // (param6 and param7 swapped)
  /* 18 */ le_uint16_t entry_index; // Index into a static enemy constructor table
  /* 1A */ le_uint16_t unknown_a1;
  /* 1C */ le_uint16_t min_children;
  /* 1E */ le_uint16_t max_children;
  /* 20 */                           // Total: 0x20 = 32 bytes
};
```

### RandomEnemyWeight (4 bytes)

```c
struct RandomEnemyWeight {
  /* 00 */ uint8_t  base_type_index;
  /* 01 */ uint8_t  def_entry_index;
  /* 02 */ uint8_t  weight;         // Relative probability (higher = more likely)
  /* 03 */ uint8_t  unknown_a4;
  /* 04 */
};
```

---

## 6. THE .QST CONTAINER FORMAT

A `.qst` file is a **transport container** holding the encrypted/compressed `.dat` and `.bin` files. It is not itself compressed; each contained file is individually compressed with PRS.

### High-Level Structure

```
[Header for .dat or .bin] 64 bytes
[Header for .bin or .dat] 64 bytes
[Interleaved chunks for both files]
```

The `.qst` contains **two file headers** (one per contained file), followed by **interleaved 1056-byte (or 1048-byte) chunks** that are demultiplexed by filename.

### File Header Variants

File headers vary by **version** (DC / GC / PC / BB):

#### DC/GC/PC Header (60 bytes)

```c
struct QstHeader_DC_GC_PC {
  /* 00 */ uint8_t   online_flag;          // 0x44 (online) or 0xA6 (download)
  /* 01 */ uint8_t   quest_id;             // Quest ID (0-255)
  /* 02 */ le_uint16_t header_size;        // Header size (should be 0x3C = 60)
  /* 04 */ char      quest_name[32];       // UTF-8 (DC) or ASCII; null-terminated
  /* 24 */ [varying padding]               // Platform-specific alignment
  /* 34 */ char      filename[16];         // Contained filename (e.g., "q001.bin" or "q001.dat"); null-terminated
  /* 44 */ le_uint32_t uncompressed_size;  // Size of decompressed file
  /* 48 */ [padding to 60]
};
```

**DC layout:**
```c
struct QstHeader_DC {
  /* 00 */ uint8_t   online_flag;
  /* 01 */ uint8_t   quest_id;
  /* 02 */ le_uint16_t header_size;
  /* 04 */ char      quest_name[32];
  /* 24 */ uint8_t   pad1[3];
  /* 27 */ char      filename[16];
  /* 37 */ uint8_t   pad2;
  /* 38 */ le_uint32_t uncompressed_size;
};
```

**GC layout:**
```c
struct QstHeader_GC {
  /* 00 */ uint8_t   online_flag;
  /* 01 */ uint8_t   quest_id;
  /* 02 */ le_uint16_t header_size;
  /* 04 */ char      quest_name[32];
  /* 24 */ le_uint32_t pad1;
  /* 28 */ char      filename[16];
  /* 38 */ le_uint32_t uncompressed_size;
};
```

**PC layout:**
```c
struct QstHeader_PC {
  /* 00 */ le_uint16_t header_size;
  /* 02 */ uint8_t   online_flag;
  /* 03 */ uint8_t   quest_id;
  /* 04 */ char      quest_name[32];
  /* 24 */ le_uint32_t pad1;
  /* 28 */ char      filename[16];
  /* 38 */ le_uint32_t uncompressed_size;
};
```

#### BB Header (88 bytes)

```c
struct QstHeader_BB {
  /* 00 */ le_uint16_t header_size;        // 0x58 = 88
  /* 02 */ le_uint16_t online_flag;        // 0x0044 (online) or 0x00A6 (download)
  /* 04 */ le_uint16_t quest_id;           // Quest ID (0-65535)
  /* 06 */ [38 bytes reserved/padding]
  /* 38 */ char       filename[16];        // UTF-8; null-terminated
  /* 48 */ le_uint32_t uncompressed_size;
  /* 4C */ char       quest_name[24];      // UTF-16 LE; null-terminated
  /* 64 */ [padding to 88]
};
```

### Chunks (Interleaved File Data)

After headers, files are chunked for reliable transport (1024-byte payload + overhead). Chunk size depends on version:

**DC/GC/PC:**
- Chunk header: 20 bytes
- Data payload: 1024 bytes
- Trailer: 4 bytes
- **Total chunk: 1048 bytes**

**BB:**
- Chunk header: 24 bytes
- Data payload: 1024 bytes
- Trailer: 8 bytes
- **Total chunk: 1056 bytes**

#### DC/GC/PC Chunk Header (20 bytes)

```c
struct Chunk_DC_GC_PC {
  /* 00 */ uint8_t   pad1;
  /* 01 */ uint8_t   chunk_no;             // Chunk sequence number (0, 1, 2, ...)
  /* 02 */ le_uint16_t pad2;
  /* 04 */ char      filename[16];         // Identifies which file this chunk belongs to
  /* 14 */ [1024 bytes of file data]
  /* 414 */ le_uint32_t data_size;         // Actual bytes written in this chunk (≤ 1024)
};
```

#### BB Chunk Header (24 bytes)

```c
struct Chunk_BB {
  /* 00 */ uint8_t   pad1;          // 0x1C
  /* 01 */ uint8_t   pad2;          // 0x04
  /* 02 */ uint8_t   pad3;          // 0x13
  /* 03 */ uint8_t   pad4;          // 0x00
  /* 04 */ le_uint32_t chunk_no;
  /* 08 */ char      filename[16];
  /* 18 */ [1024 bytes of file data]
  /* 418 */ le_uint32_t data_size;
  /* 41C */ le_uint32_t trailer;     // Usually 0
};
```

**Key points:**
- Chunks for the same file are **not necessarily adjacent** — they are interleaved with chunks from other contained files.
- Each chunk specifies its **filename** in the header, so demultiplexing is straightforward.
- The **data_size field** in the trailer indicates the actual number of bytes used in the 1024-byte payload; trailing bytes are padding (usually 0x00).

---

## 7. SPECIFICATION FOR `formats/quest_map.py`

The `quest_map.py` module must implement:

### 1. Decompression / Compression

```python
def parse_dat(data: bytes) -> DatFile:
    """
    Decompress (if PRS-compressed) and parse .dat file.
    
    Returns an object with attributes:
      - objects: List[DatObject] — parsed object entries
      - npcs: List[DatNpc] — parsed enemy/NPC entries
      - events: List[DatEvent] — parsed events with action streams
      - unknowns: List[DatUnknown] — (sections 4/5; challenge mode)
    """
    pass

def serialize_dat(dat: DatFile) -> bytes:
    """
    Emit binary .dat (decompressed).
    Round-trip invariant: parse(serialize(x)) == x.
    """
    pass

def compress_dat(decompressed: bytes) -> bytes:
    """Use prs.py to compress."""
    pass

def decompress_dat(compressed: bytes) -> bytes:
    """Use prs.py to decompress."""
    pass
```

### 2. Entity Parsing

```python
class DatEntity:
    """Raw 68-byte (object) or 72-byte (enemy) buffer + parsed fields."""
    area_id: int
    base_type: int
    floor: int
    group: int  # (objects only)
    room: int
    pos: Tuple[float, float, float]
    angle: Tuple[int, int, int]  # 16-bit rotations
    params: List[Union[float, int]]  # param1-6

class DatObject(DatEntity):
    pass

class DatNpc(DatEntity):
    wave_number: int
    wave_number2: int
    num_children: int  # (enemies only)
```

### 3. Event Parsing

```python
class DatEvent:
    id: int
    room: int
    wave_number: int
    delay_frames: int  # (or min_delay, max_delay for evt2)
    event_type: int  # 0=no-op, 1=spawn wave
    actions: List[DatEventAction]
    area_id: int

class DatEventAction:
    pass

class SpawnNpcs(DatEventAction):
    section_id: int
    appear_flag: int

class UnlockDoor(DatEventAction):
    door_id: int

class LockDoor(DatEventAction):
    door_id: int

class TriggerEvent(DatEventAction):
    event_id: int
```

### 4. Section-by-Section Serialization

```python
def serialize_objects(objects: List[DatObject], floor: int) -> bytes:
    """Emit a single OBJECT_SETS section (with header)."""
    pass

def serialize_enemies(enemies: List[DatNpc], floor: int) -> bytes:
    """Emit a single ENEMY_SETS section (with header)."""
    pass

def serialize_events(events: List[DatEvent], floor: int) -> bytes:
    """
    Emit a single EVENTS section (with header + event entries + action stream).
    Must handle both Event1Entry (standard) and Event2Entry (challenge mode).
    """
    pass
```

### 5. Multi-Floor Indexing

```python
def parse_dat_by_floor(data: bytes) -> Dict[int, FloorData]:
    """
    Return a dict keyed by floor index, with each value containing:
      - objects: List[DatObject] for that floor
      - enemies: List[DatNpc] for that floor
      - events: List[DatEvent] for that floor
      - random_locations: (if challenge mode)
      - random_definitions: (if challenge mode)
    """
    pass
```

### 6. Round-Trip Validation

```python
def test_round_trip_parity():
    """
    For every shipped BB quest in the reference corpus (~/PSOBB.IO/data/quest_data/):
    1. Decompress .qst → extract .dat
    2. parse_dat() → serialize_dat() → recompressed
    3. Decompress the recompressed version
    4. Assert byte-for-byte equality of the decompressed payload
    
    Use newserv's disassemble-quest (independent oracle) as a secondary check.
    """
    pass
```

### 7. .QST Container Handling

The module should optionally provide:

```python
def extract_qst(qst_bytes: bytes) -> Tuple[bytes, bytes, Dict[str, str]]:
    """
    Parse .qst container; return (decompressed_dat, decompressed_bin, metadata).
    """
    pass

def pack_qst(dat: bytes, bin: bytes, version: str, online: bool, quest_id: int, quest_name: str) -> bytes:
    """
    Pack .dat + .bin into a .qst container (with PRS compression applied).
    version: 'DC', 'GC', 'PC', or 'BB'
    """
    pass
```

### 8. Error Handling

- **Malformed sections:** Log and skip/pad gracefully (following phantasmal-world's "warning" pattern).
- **Chunk corruption in .qst:** Detect missing/duplicate chunks and report.
- **Action stream truncation:** Warn if action stream ends abruptly (no 0x01 terminator).
- **Mismatched version headers:** Validate both contained files have the same version.

---

## 8. KEY INVARIANTS & NOTES

1. **No position transform needed for parsing** — store positions as-is (room-relative). The room's world transform is stored in separate `.rel` area-layout files; the engine applies it at load time.

2. **Index/entity_id fields are runtime-only** — do NOT assign them during serialization. The engine computes them sequentially during load.

3. **Wave triggering is by (room, wave_number) pair** — an event with room=3 and wave_number=5 triggers all enemies where room==3 AND wave_number==5.

4. **Action streams are position-independent** — offsets in the stream are **relative to the action stream base**, not absolute file offsets. This simplifies patching/editing.

5. **Challenge mode (sections 4 & 5)** is out of scope for the initial MVP but the structure should be understood for future-proofing. Only skip if the section type is 4 or 5 and log a warning.

6. **Multi-floor quests:** A single `.dat` file can describe all floors of a quest (Episode 1 = 18 floors). Sections appear per-floor, so the section header's `floor` field is the primary key. When building a quest DSL, the map codec groups entities by floor for easier editing.

7. **Rotations are in 16-bit fixed-point:** `angle_radians = (angle_u16 / 65536.0) * 2π`. There is no sign — all values are unsigned; rotations wrap naturally.

---

## 9. REFERENCES

- **newserv (MIT):** `src/Map.hh` (lines 138–500 for complete struct definitions), `src/Map.cc` (parsing logic)
- **newserv (MIT):** `src/Quest.cc` (`.qst` packing/unpacking)
- **phantasmal-world (MIT):** `psolib/fileFormats/quest/Dat.kt` (working codec; use for validation)
- **phantasmal-world (MIT):** `psolib/fileFormats/quest/Qst.kt` (`.qst` interleaving logic)
- **qedit (closed, understand-only):** Format and semantics corroboration via RE

---

## 10. VALIDATION CHECKLIST

Before shipping `quest_map.py`:

- [ ] Parse and emit all five section types (0-5) without truncation or corruption
- [ ] Round-trip parity test passes on the full BB quest corpus (disassemble → assemble = byte-identical decompressed `.dat`)
- [ ] Multi-floor quest handling validated (e.g., Episode 1 quest with 18 floors)
- [ ] Action stream parsing handles all four action types + terminator + padding
- [ ] `.qst` container extraction and repacking (with optional compression via PRS)
- [ ] Error recovery: malformed sections logged, file still parseable
- [ ] No hardcoded paths; uses `Path.home()` or env vars for test corpus
- [ ] Lint clean (`ruff`)
- [ ] All gate-3 parity tests pass
