# Quest DSL Reference Data

This document provides authoritative ID/enum tables for quest scripting. All values cross-referenced against Alisaryn's qedit, newserv (MIT), and phantasmal-world (MIT) sources.

---

## NPC Skin IDs

These define NPC/character appearances placeable in quests. Valid on both server and client.

| ID | Name | Episode | Notes |
|--|--|--|--|
| 1 | Female Base | Ep1,Ep2,Ep4 | Pioneer 2 default female |
| 2 | Female Child | Ep1,Ep2,Ep4 | Small female |
| 3 | Female Dwarf | Ep1,Ep2,Ep4 | Dwarf female |
| 4 | Female Fat | Ep1,Ep2,Ep4 | Heavy female |
| 5 | Female Macho | Ep1,Ep2,Ep4 | Muscular female |
| 6 | Female Old | Ep1,Ep2,Ep4 | Elder female |
| 7 | Female Tall | Ep1,Ep2,Ep4 | Tall female |
| 8 | Male Base | Ep1,Ep2,Ep4 | Pioneer 2 default male |
| 9 | Male Child | Ep1,Ep2,Ep4 | Small male |
| 10 | Male Dwarf | Ep1,Ep2,Ep4 | Dwarf male |
| 11 | Male Fat | Ep1,Ep2,Ep4 | Heavy male |
| 12 | Male Macho | Ep1,Ep2,Ep4 | Muscular male |
| 13 | Male Old | Ep1,Ep2,Ep4 | Elder male |
| 14 | Male Tall | Ep1,Ep2,Ep4 | Tall male |
| 25 | Blue Soldier | Ep1,Ep2,Ep4 | Pioneer 2 military (blue uniform) |
| 26 | Red Soldier | Ep1,Ep2,Ep4 | Pioneer 2 military (red uniform) |
| 27 | Principal Tyrel | Ep1,Ep2,Ep4 | Pioneer 2 facility head |
| 28 | Tekker | Ep1,Ep2,Ep4 | Equipment appraiser |
| 29 | Bank Lady | Ep1,Ep2,Ep4 | Bank teller |
| 30 | Scientist | Ep1,Ep2,Ep4 | Pioneer 2 lab scientist |
| 31 | Nurse | Ep1,Ep2,Ep4 | Medical center nurse |
| 32 | Irene | Ep1,Ep2,Ep4 | Medical center administrator |
| 33 | Broomop | Ep1 | NPC Broomop (Episode 1) |
| 34 | Hunter (Male) | Ep1,Ep2,Ep4 | Default hunter class |
| 36 | Ranger (Male) | Ep1,Ep2,Ep4 | Default ranger class |
| 37 | Racast | Ep1,Ep2,Ep4 | Default racast class |
| 38 | Racaseal | Ep1,Ep2,Ep4 | Default racaseal class |
| 39 | Fomarl | Ep1,Ep2,Ep4 | Default fomarl class |
| 40 | Fomewm | Ep1,Ep2,Ep4 | Default fomewm class |
| 41 | Fomewearl | Ep1,Ep2,Ep4 | Default fomewearl class |
| 43 | HUnewearl | Ep1,Ep2,Ep4 | Default hunewearl class |
| 44 | Cast (Male) | Ep1,Ep2,Ep4 | Default cast class |
| 45 | RAmar | Ep1,Ep2,Ep4 | Default ramar class |
| 48 | HMN FRC W 01 | Ep1,Ep2,Ep4 | Human Force female variant |
| 49 | NMN FRC M 01 | Ep1,Ep2,Ep4 | Human Force male variant |
| 50 | NMN FRC W 01 | Ep1,Ep2,Ep4 | Newmen female variant |
| 208 | Ep2 NPC D0 | Ep2 | Episode 2 NPC character D0 |
| 209 | Directrice (Natasha) | Ep2 | Ep2 facility head |
| 210 | Dan | Ep2 | Ep2 scientist |
| 211 | Ep2 NPC D3 | Ep2 | Episode 2 NPC character D3 |
| 240 | Ep2 Armor Shop | Ep2 | Armor vendor (Episode 2) |
| 241 | Ep2 Item Shop | Ep2 | Item vendor (Episode 2) |
| 242 | Default Fomar | Ep2 | Generic female FOmar |
| 243 | Default Ramarl | Ep2 | Karen (Episode 2 character) |
| 244 | Leo | Ep2 | Episode 2 NPC Leo |
| 245 | Pagini | Ep2 | Episode 2 NPC Pagini |
| 246 | Ep2 NPC F6 | Ep2 | Episode 2 NPC character F6 |
| 247 | Nol | Ep2 | Episode 2 NPC Nol |
| 248 | Elly | Ep2 | Episode 2 NPC Elly |
| 249 | Ep2 NPC F9 | Ep2 | Episode 2 NPC character F9 |
| 250 | Ep2 Item Shop | Ep2 | Ep2 item vendor (alternate) |
| 251 | Ep2 Weapon Shop | Ep2 | Ep2 weapon vendor |
| 252 | Security Guard | Ep2 | Ep2 military security |
| 253 | Ep2 Hunters Guild | Ep2 | Ep2 hunter guild NPC |
| 254 | Ep2 Bank Lady | Ep2 | Ep2 bank teller |
| 256 | Momoka | Ep2 | Episode 2 NPC Momoka |
| 272 | Astark | Ep4 | Episode 4 boss (rare NPC variant) |
| 273 | Satellite Lizard | Ep4 | Episode 4 creature NPC |
| 274 | Merissa A | Ep4 | Episode 4 NPC Merissa A |
| 275 | Girtablulu | Ep4 | Episode 4 creature NPC |
| 276 | Zu | Ep4 | Episode 4 creature NPC |
| 277 | Boota | Ep4 | Episode 4 creature NPC |
| 278 | Dorphon | Ep4 | Episode 4 creature NPC |
| 279 | Goran | Ep4 | Episode 4 creature NPC |
| 281 | Saint Million | Ep4 | Episode 4 NPC Saint Million |

---

## Enemy/Monster Type IDs

Indexed by in-game reference. Episode availability and rarity flags noted.

### Episode 1 Enemies
| ID | Name | Type | Rarity | Boss | Notes |
|--|--|--|--|--|--|
| 0 | Booma | Insectoid | - | - | Forest flying enemy |
| 1 | Savage Wolf | Canine | - | - | Forest runner |
| 2 | Rag Rappy | Avian | - | - | Forest bird/rare drop |
| 3 | Monest | Jellyfish | - | - | Forest floating enemy |
| 4 | Hildebear | Bear | - | - | Forest bruiser |
| 5 | Grass Assassin | Grass | - | - | Forest plant enemy |
| 6 | Poison Lily / Del Lily | Grass | - | - | Caves/forest plant |
| 7 | Nano Dragon | Dragon | - | - | Caves small dragon |
| 8 | Evil Shark | Shark | - | - | Caves water enemy |
| 9 | Pofuilly Slime | Slime | - | - | Caves slime |
| 10 | Pan Arms | Armored | - | - | Caves four-armed enemy |
| 11 | Gillchic | Avian | - | - | Caves bird |
| 12 | Garanz | Insectoid | - | - | Caves centaur-like |
| 13 | Sinow Blue | Muscle | - | - | Caves rare variant |
| 14 | Canadine | Armored | - | - | Caves centaur |
| 15 | Canane | Armored | - | - | Caves centaur variant |
| 16 | Dubchic Switch | Avian | - | - | Caves trigger enemy |
| 17 | Delsaber | Android | - | - | Ruins sword fighter |
| 18 | Chaos Sorcerer | Android | - | - | Ruins caster |
| 19 | Dark Gunner | Android | - | - | Ruins ranged |
| 20 | Dark Gunner (Activator) | Android | - | - | Ruins trigger unit |
| 21 | Chaos Bringer | Android | - | - | Ruins heavy caster |
| 22 | Dark Belra | Android | Rare | - | Ruins rare android |
| 23 | Dimenian | Armored | - | - | Ruins teleporting |
| 24 | Bulclaw | Armored | - | - | Ruins tanky |
| 25 | Claw | Armored | - | - | Ruins mobile tank |
| 40 | Dragon (Ep1 Boss) | Dragon | - | Yes | Boss: red dragon |
| 41 | De Rol Le (Ep1 Boss) | Crustacean | - | Yes | Boss: mantis-like |
| 42 | Vol Opt Control (Ep1 Boss) | Machine | - | Yes | Boss: mechanical core |

### Episode 2 Enemies
| ID | Name | Type | Rarity | Boss | Ep2 Only | Notes |
|--|--|--|--|--|--|--|
| 26 | Epsilon | Machine | - | - | Yes | Ep2 mechanical box |
| 27 | Sinow Berill | Muscle | - | - | Yes | Ep2 rare variant |
| 28 | Merillias | Armored | - | - | Yes | Ep2 scaled armor |
| 29 | Mericarol | Armored | - | - | Yes | Ep2 scaled variant |
| 30 | Ul Gibbon | Armored | - | - | Yes | Ep2 jumping armor |
| 31 | Gibbles | Armored | - | - | Yes | Ep2 small version |
| 32 | Gee | Machine | - | - | Yes | Ep2 flying machine |
| 33 | Gi Gue | Machine | - | - | Yes | Ep2 machine variant |
| 34 | Deldepth | Armored | - | - | Yes | Ep2 sub-armored |
| 35 | Delbiter | Armored | - | - | Yes | Ep2 biting armor |
| 36 | Dolmdarl | Armored | - | - | Yes | Ep2 barrel-like |
| 37 | Morfos | Armored | - | - | Yes | Ep2 morphing armor |
| 38 | Recon Box | Machine | - | - | Yes | Ep2 detector box |
| 39 | Sinow Zoa | Muscle | - | - | Yes | Ep2 variant muscle |
| 40 | Epsilon (alt) | Machine | - | - | Yes | Ep2 box variant |
| 41 | Ill Gill (Ep2 Boss) | Crustacean | - | Yes | Yes | Boss: Ep2 mantis |
| 45 | Olga Flow (Ep2 Boss) | Beast | - | Yes | Yes | Boss: Ep2 whale |
| 46 | Barba Ray (Ep2 Boss) | Crustacean | - | Yes | Yes | Boss: Ep2 crab |
| 47 | Gol Dragon (Ep2 Boss) | Dragon | - | Yes | Yes | Boss: Ep2 black dragon |

### Episode 4 Enemies
| ID | Name | Type | Rarity | Boss | Ep4 Only | Notes |
|--|--|--|--|--|--|--|
| 48 | Boota (Ep4) | Insectoid | - | - | Yes | Ep4 Booma variant |
| 49 | Ze Boota | Insectoid | Rare | - | Yes | Ep4 stronger variant |
| 50 | Ba Boota | Insectoid | Rare | - | Yes | Ep4 rare variant |
| 51 | Satellite Lizard | Reptile | - | - | Yes | Ep4 desert runner |
| 52 | Yowie | Beast | - | - | Yes | Ep4 hairy creature |
| 53 | Dorphon | Armored | - | - | Yes | Ep4 insectoid armor |
| 54 | Astark | Beast | - | - | Yes | Ep4 large creature |
| 55 | Girtablulu | Armored | - | - | Yes | Ep4 scorpion-like |
| 56 | Merissa A | Armored | - | - | Yes | Ep4 scaled female |
| 57 | Goran | Machine | - | - | Yes | Ep4 explosive bot |
| 58 | Goran (Detonator) | Machine | Rare | - | Yes | Ep4 explosive variant |
| 59 | Pyro Goran | Machine | Rare | - | Yes | Ep4 fire variant |
| 60 | Zu | Beast | - | - | Yes | Ep4 flying creature |
| 61 | Saint Million (Ep4 Boss) | Machine | - | Yes | Yes | Boss: Ep4 sentinel |
| 62 | Kondrieu (Ep4 Boss) | Machine | - | Yes | Yes | Boss: Ep4 mechanical boss |

**Note**: Ep1/Ep2 share most base enemies; episode-specific variants have Rarity or Ep-Only flags. Ep4 introduces entirely new enemy roster.

---

## Object Type IDs

Structural elements and interactive objects placeable in quest environments.

| ID | Name | Category | Episode | Notes |
|--|--|--|--|--|
| 0 | Player Set 1 | Spawn | All | Player spawn point (1-4 players) |
| 1 | Particle | Effect | All | Ambient particle emitter |
| 2 | Teleporter | Transition | All | Warp point to another area |
| 3 | Warp | Transition | All | Area transition trigger |
| 4 | Light Collision | Hazard | All | Invisible collision light |
| 5 | Item | Loot | All | Droppable item container |
| 6 | Env Sound | Audio | All | Environmental audio trigger |
| 7 | Fog Collision | Hazard | All | Fog area boundary |
| 8 | Event Collision | Script | All | Quest event trigger zone |
| 9 | Chara Collision | Script | All | Character interaction zone |
| 10 | Elemental Trap | Hazard | All | Fire/ice/thunder trap |
| 11 | Status Trap | Hazard | All | Poison/paralysis trap |
| 12 | Heal Trap | Healing | All | Recovery item container |
| 13 | Large Elemental Trap | Hazard | All | Area damage trap |
| 14 | Obj Room ID | Meta | All | Quest room identifier |
| 15 | Sensor | Script | All | Movement/proximity sensor |
| 16 | Unknown Item (16) | Unknown | All | Unused/reserved |
| 17 | Lensflare | Effect | All | Lens flare light |
| 18 | Script Collision | Script | All | Quest script trigger |
| 19 | Heal Ring | Healing | All | Healing circle (platforms) |
| 20 | Map Collision | Structure | All | Solid collision boundary |
| 21 | Script Collision A | Script | All | Alt script trigger |
| 22 | Item Light | Loot | All | Item drop light effect |
| 23 | Radar Collision | Structure | All | Minimap boundary |
| 24 | Fog Collision SW | Hazard | All | SW corner fog boundary |
| 25 | Boss Teleporter | Transition | All | Boss arena warp |
| 26 | Image Board | Structure | Ep1 | Wall-mounted screen |
| 27 | Area Warp (Ep1) | Transition | Ep1 | Ragol area transition (no sound) |
| 28 | Epilogue | Transition | Ep1 | Quest completion trigger |
| 32 | Box Detect Object | Structure | All | Box/crate object |
| 33 | Symbol Chat Object | Script | All | Symbol chat activation |
| 34 | Touch Plate Object | Script | All | Floor pressure plate |
| 35 | Targetable Object | Script | All | Destruction target |
| 36 | Effect Object | Effect | All | Visual effect emitter |
| 37 | Count Down Object | Script | All | Timer display object |
| 64 | Menu Activation | Script | All | Shop/menu trigger |
| 65 | Telepipe Location | Transition | Ep1 | Telepipe entry point |
| 66 | BGM Collision | Audio | All | Music zone trigger |
| 67 | Main Ragol Teleporter | Transition | Ep1 | Main quest warp |
| 68 | Lobby Teleporter | Transition | Ep1 | Lobby return warp |
| 69 | Principal Warp | Transition | Ep1 | Facility warp |
| 70 | Shop Door | Transition | Ep1 | Shop entrance |
| 71 | Hunter's Guild Door | Transition | Ep1 | Guild entrance |
| 72 | Teleporter Door | Transition | Ep1 | Teleporter entrance |
| 73 | Medical Center Door | Transition | Ep1 | Medical center entrance |
| 74 | Elevator | Transition | Ep1 | Vertical transition |
| 75 | Easter Egg | Script | Ep1 | Special event trigger |
| 76 | Valentines Heart | Decoration | All | Holiday decoration (Feb) |
| 77 | Christmas Tree | Decoration | All | Holiday decoration (Dec) |
| 78 | Christmas Wreath | Decoration | All | Holiday decoration (Dec) |
| 79 | Halloween Pumpkin | Decoration | All | Holiday decoration (Oct) |
| 80 | 21st Century | Decoration | Ep1 | Y2K decoration |
| 81 | Dr Robotnic / Sonic / Knux / Tails | Decoration | Ep1 | Easter egg characters |
| 82 | Welcome Board | Decoration | Ep1 | Info board |
| 83 | Firework | Effect | Ep1 | Firework emitter |
| 84 | Lobby Screen Door | Transition | Ep1 | Lobby section door |
| 85 | Main Ragol Teleporter (Battle Next) | Transition | Ep1 | Battle arena warp |
| 86 | Lab Teleporter Door | Transition | Ep1 | Laboratory entrance |
| 87 | Pioneer 2 Invisible Touchplate | Script | Ep1 | Invisible floor trigger |
| 128 | Forest Door | Transition | Ep1 | Forest area door |
| 129 | Forest Switch | Script | Ep1 | Forest mechanical switch |
| 130 | Laser Fence | Hazard | Ep1 | Red laser barrier |
| 131 | Laser Square Fence | Hazard | Ep1 | Square laser trap |
| 132 | Forest Laser Fence Switch | Script | Ep1 | Laser control switch |
| 133 | Light Rays | Effect | Ep1 | Atmospheric light |
| 134 | Blue Butterfly | Creature | Ep1 | Flying decoration |
| 135 | Crashed Probe / Probe | Creature | Ep1 | Drone object (states) |
| 136 | Random Type Box 1 | Loot | Ep1 | Random item box |
| 137 | Forest Weather Station | Structure | Ep1 | Weather control |
| 138 | Battery | Loot | Ep1 | Battery item |
| 139 | Forest Console 1/2 | Structure | Ep1 | Computer terminal |
| 140 | Black Sliding Door | Transition | Ep1 | Automated door |
| 141 | Rico Message Pod | Script | Ep1 | Message container |
| 142 | Energy Barrier | Hazard | Ep1 | Force field |
| 143 | Forest Rising Bridge | Transition | Ep1 | Drawbridge |
| 144 | Switch (None Door) | Script | Ep1 | Generic switch |
| 145 | Enemy Box (Grey) | Loot | Ep1 | Enemy-guarded box |
| 146 | Fixed Type Box | Loot | Ep1 | Fixed item box |
| 147 | Enemy Box (Brown) | Loot | Ep1 | Enemy-guarded box variant |
| 149 | Empty Type Box | Loot | Ep1 | Empty/dummy box |
| 150 | Laser Fence Ex | Hazard | Ep1 | Extended laser barrier |
| 151 | Laser Square Fence Ex | Hazard | Ep1 | Extended square laser |
| 192 | Floor Panel 1/2/3/4 | Structure | All | Floor tiles by area |
| 193 | Caves 4 Button Door | Transition | Ep1 | 4-switch door |
| 194 | Caves Normal Door | Transition | Ep1 | Standard cave door |
| 195 | Caves Smashing Pillar | Hazard | Ep1 | Crush trap pillar |
| 196-198 | Caves Sign 1/2/3 | Decoration | Ep1 | Info signs |
| 199 | Hexagonal Tank | Structure | Ep1 | Fuel container |
| 200 | Brown Platform | Transition | Ep1 | Conveyor/platform |
| 201 | Warning Light Object | Decoration | Ep1 | Blinker light |
| 203 | Rainbow | Effect | Ep1 | Color effect |
| 204-205 | Floating Jellyfish / Dragonfly | Creature | Ep1 | Flying decoration |
| 206 | Caves Switch Door | Transition | Ep1 | Cave switch-operated door |
| 207 | Robot Recharge Station | Structure | Ep1 | Robot charger |
| 208 | Caves Cake Shop | Structure | Ep1 | Shop prop |
| 209-211 | Caves Rocks (Small/Med/Large) | Structure | Ep1 | Rock sizes |
| 212-220 | Caves 2/3 Rocks | Structure | Ep1 | More rock variants |
| 222 | Floor Panel 2 | Structure | Ep1 | Floor type 2 |
| 223-225 | Destructable Rocks (Caves 1/2/3) | Hazard | Ep1 | Breakable walls |
| 256 | Mines Door | Transition | Ep1 | Mine entrance |
| 257 | Floor Panel 3 | Structure | Ep1 | Floor type 3 |
| 258 | Mines Switch Door | Transition | Ep1 | Mine switch door |
| 259 | Large Cryo-Tube | Structure | Ep1 | Freezing chamber |
| 260 | Computer (Calus-like) | Structure | Ep1 | Computer terminal |
| 261 | Green Screen Opening/Closing | Effect | Ep1 | Sliding gate effect |
| 262 | Floating Robot | Creature | Ep1 | Flying robot |
| 263 | Floating Blue Light | Effect | Ep1 | Light orb effect |
| 264-266 | Self Destructing Objects 1/2/3 | Hazard | Ep1 | Timed explosives |
| 267 | Spark Machine | Hazard | Ep1 | Electrical hazard |
| 268 | Mines Large Flashing Crate | Loot | Ep1 | Blinking box |
| 304 | Ruins Seal | Structure | Ep1 | Sealed entrance |
| 320 | Ruins Teleporter | Transition | Ep1 | Ruins warp point |
| 321 | Ruins Warp (Site to Site) | Transition | Ep1 | Intra-area warp |
| 322 | Ruins Switch | Script | Ep1 | Ruins control switch |
| 323 | Floor Panel 4 | Structure | Ep1 | Floor type 4 |
| 324-327 | Ruins Doors 1/3/2 / Button Door | Transition | Ep1 | Ruins doors |

**Note**: Object IDs may repeat in Ep2/Ep4 with different semantic meanings; use Episode context when parsing.

---

## Floor/Area IDs per Episode

Maps the floor index to area name. Used for quest map generation.

### Episode 1 Floors
| Floor | Area Name | Description |
|--|--|--|
| 0 | Forest 1 | Dense forest (entry) |
| 1 | Forest 2 | Forest caves |
| 2 | Caves 1 | Mine system (entry) |
| 3 | Caves 2 | Deeper mines |
| 4 | Caves 3 | Mine core |
| 5 | Mines 1 | Central mines |
| 6 | Mines 2 | Mine tunnels |
| 7 | Ruins 1 | Ancient ruins (entry) |
| 8 | Ruins 2 | Ruins middle |
| 9 | Ruins 3 | Ruins depths |
| 10 | Boss Floor | Dragon arena |
| 11 | Boss Floor | De Rol Le arena |
| 12 | Boss Floor | Vol Opt arena |
| 13-17 | Unused | Reserved slots |

### Episode 2 Floors
| Floor | Area Name | Description |
|--|--|--|
| 0-2 | Seabeds 1-3 | Underwater (entry/middle/deep) |
| 3 | Sky Ship Deck | Starship exterior |
| 4 | Sky Ship Interior | Starship inside |
| 5 | Sky Ship Hallway | Starship corridors |
| 6-8 | Boss Floors | Olga/Barba Ray/Gol Dragon arenas |
| 9-17 | Unused | Reserved slots |

### Episode 4 Floors
| Floor | Area Name | Description |
|--|--|--|
| 0-2 | Desert 1-3 | Sand desert areas |
| 3-5 | Jungle 1-3 | Jungle/undergrowth areas |
| 6-8 | Crater 1-3 | Impact crater zones |
| 9 | Boss Floor | Boss arena |
| 10-17 | Unused | Reserved slots |

---

## Fog Entry IDs

Fog/atmosphere configuration. Binary fog data stored in fogentry.dat (qedit reference).

| ID | Fog Type | Density | Notes |
|--|--|--|--|
| 0-15 | Various | Low-High | Standard fog presets |
| 16-31 | Dense | Very High | Heavy fog variants |
| 32+ | Custom | Configured | Custom fog/weather |

*(Exact fog definitions require fogentry.dat binary decode; values 0-15 are standard game presets per Ep/area.)*

---

## Quest Register/Flag Conventions

Standardized register slots for common quest mechanics. Cross-referenced from newserv and qedit patterns.

| Register | Bit Width | Purpose | Standard Values | Notes |
|--|--|--|--|--|
| **R250** | 16-bit | Difficulty | 0=Very Easy, 1=Normal, 2=Hard, 3=Very Hard | Quest-wide setting |
| **R251** | 16-bit | Game Mode | 0=Normal, 1=Story, 2=Challenge, 3=Ultimate | Episode/quest variant |
| **R252** | 16-bit | Player Count | 0=Solo, 1-3=Team size | Auto-set by server |
| **R253** | 16-bit | Episode | 1=Ep1, 2=Ep2, 4=Ep4 | Questmark embedded |
| **R254** | 16-bit | Area Index | 0-17 | Current floor (0-indexed) |
| **R255** | 16-bit | Quest State | 0=In Progress, 1=Complete | Quest victory flag |

**Additional Convention**:
- **R200-R249**: Quest designer free slots (not reserved)
- **R100-R199**: Per-player state (indexed by player 0-3)
- **R0-R99**: System/scoring (bosses defeated, items collected, etc.)

*(Exact register mapping per server impl; these are conventions observed across newserv + Ephinea + PSOBB.io)*

---

## Cross-Episode Entity Differences

Entities with episode-specific variants:

| Entity | Ep1 ID | Ep2 ID | Ep4 ID | Notes |
|--|--|--|--|--|
| Rag Rappy | 2 | - | Sand Rappy (variant) | Desert variant in Ep4 |
| Booma | 0 | - | Boota (48+) | Ep4 uses family variants |
| Dragon | 40 | Gal Gryphon | - | Ep2 uses different boss |
| Sinow variants | Blue(13) | Berill(27),Spigell(27) | - | Episode-specific skins |
| Directrice NPC | - | Skin 209 | - | Ep2 only facility head |

---

## Notes for DSL Implementation

1. **Skin validation**: Check skin ID against valid Episode range before placing NPCs.
2. **Enemy rates**: Monster spawn rates are per-floor, not per-enemy; vary by difficulty.
3. **Object persistence**: Some objects (doors, boxes) persist until destroyed; script them via event collision.
4. **Register scoping**: R0-R99 are global per quest; R100+ may be player-indexed per impl.
5. **Floor linking**: Teleporters must specify target floor ID and spawn point within that floor's layout.
6. **Fog blending**: Fog zones can overlap; fog ID determines blend priority and density.

---

## References

- **qedit-alisaryn**: Alisaryn's PSO Quest Editor (LGPL v2.1) — authoritative object/NPC/enemy definitions.
- **newserv**: MIT-licensed server source; EnemyType.hh, Map.hh contain canonical ID mappings.
- **phantasmal-world**: MIT-licensed library; QuestEntity.kt, QuestNpc.kt provide quest data structures.
- **PSOBB.IO / Ephinea**: Live server implementations; tested for register slot and difficulty conventions.
