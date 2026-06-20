# PSOBB Asset Categorization Database

Generated 2026-04-26. Companion to `categorization_db.json` — human-readable explanation of every prefix family.

## 1. Goals

- Replace the editor's lambda-list categorizer (`manifest.py:166-194`) with a JSON-driven rule table.
- Resolve the 5 user-reported miscategorizations.
- Document every observed prefix in the live data tree (`~/PSOBB.IO/data/`, 883 files) with evidence from PsoBB.exe + Phantasmal World + newserv.

## 2. Sources & evidence

| Source | Path | What it provides |
|---|---|---|
| PsoBB.exe | `~/PSOBB.IO/PsoBB.exe` | Static-data string blocks; constructor disasm |
| Memory: psobb_full_entity_map | `~/.claude/.../memory/psobb_full_entity_map.md` | 33 confirmed cls→model→unitxt mappings |
| Memory: psobb_binary_scan | `~/.claude/.../memory/psobb_binary_scan.md` | 460 cls values, scanning method |
| Memory: ghidra_psobb_findings | `~/.claude/.../memory/ghidra_psobb_findings.md` | Function labels (set_sinowberill_name_unitxt_id @0x59E844 etc.) |
| pixelateds_psobb_mods | `~/Repositories/pixelateds-psobb-mods/entity_cls_table.h` | 460-entry cls→model lookup table |
| Phantasmal World | `_reference/phantasmal-world/psolib/.../NpcType.kt` + `ObjectType.kt` | Canonical entity names + areaIds |
| newserv-sparse | `_reference/newserv-sparse/src/EnemyType.cc` | Enemy enum + episode/area mappings |

## 3. Prefix master table

### 3.1 Player class assets (`pl[A-Z]*`)

| Pattern | In-game | Notes |
|---|---|---|
| `pl[A-Z]bdy00.nj` | Player class body | A-Z slots = HU/RA/FO × m/n/w/etc. |
| `pl[A-Z]hed??.nj` | Player class head | hed00 base, hed01-09 alts |
| `pl[A-Z]hai??.nj` | Player class hair | (Some classes only) |
| `pl[A-Z]cap??.nj` | Player class headgear | (FOmar caps etc.) |
| `pl[A-Z]nj.bml` | Player class skeleton bundle | |
| `pl[A-Z]smp.rel` | Player class skin/anim map | |
| `pl[A-Z]tex.afs` | Player class textures | |
| `plZsmpnj.afs` | Generic player template | "Z" = template proto |

Source: data tree directory listing.

### 3.2 Enemies (`bm_ene_*` and legacy `bm[N]_*`)

| Pattern | In-game | Episode/Area | Source |
|---|---|---|---|
| `bm_ene_bm1_shark*` | Evil/Pal/Guil Shark | EP1 Caves | psobb_full_entity_map.md cls 0xA72D1C |
| `bm_ene_bm2_moja*` | Booma family | EP1 Forest | unitxt 9/10/11 |
| `bm_ene_bm3_fly*` | Mothmant/Monest | EP1 Forest | binary string |
| `bm_ene_bm5_wolf*` | Savage/Barbarous Wolf | EP1 Forest | Phantasmal |
| `bm_ene_bm5_gibon*` | Ul/Zol Gibbon | EP2 Jungle | Phantasmal |
| `bm_ene_bm9_s_mericarol*` | Mericarol family | EP2 Spaceship | Phantasmal |
| `bm_ene_grass*` / `cgrass*` | Grass Assassin | EP1 Caves | cls 0xA72F24 |
| `bm_ene_balclaw.bml` | Bulclaw / Claw | EP1 Mines | cls 0xA734CC |
| `bm_ene_dubchik*` | Dubchic / Gilchic | EP1 Mines | unitxt 24/50 |
| `bm_ene_gyaranzo*` | Garanz | EP1 Mines | Phantasmal |
| `bm_ene_darkgunner*` | Dark Gunner | EP1 Mines | unitxt 34 |
| `bm_ene_me1_gee*` | Gee | EP2 | binary @me1_gee |
| `bm_ene_me1_mb*` | Death Gunner / Mech-Bot | EP2 Spaceship | unitxt 34/35 |
| `bm_ene_me3_shinowa*` | Sinow Berill / Spigell | EP2 Temple | binary `TEX:GOLD/NORMAL` + ghidra label |
| `bm_ene_me3_zoa*` | Sinow Zoa / Zele | EP2 Temple | binary `STELTH:ENABLE` + `APPEAR:ROOF` |
| `bm_ene_me3_stelthshinowa.bml` | Stealth Sinow variant | EP2 Temple | binary string |
| `bm_ene_me3_beril_low.bml` | Sinow Berill (LOD) | EP2 Temple | entity_cls_table.h cls 0xA74790 |
| `bm_ene_df1_saver.bml` | Delsaber | EP1 Ruins | cls 0xA739A8 unitxt 30 |
| `bm_ene_df2_bringer*` | Chaos Bringer | EP1 Ruins | binary @df2_bringer |
| `bm_ene_df3_dimedian*` | Dimenian / La / So | EP1 Ruins | cls 0xA739EC unitxt 41-43 |
| `bm_ene_re2_flower*` | Poison/Nar Lily | EP1 Forest | binary @re2_flower |
| `bm_ene_dkflower.bml` | Del Lily | EP4 Desert (rare) | binary `SE_TOWER_DELLILY_*` |
| `bm_ene_re4_sorcerer*` | Chaos Sorcerer + Bee R/L | EP1 Ruins | binary @re4_sorcerer |
| `bm_ene_re7_berura*` | Dark Belra | EP1 Ruins | binary `Error : Berura` |
| `bm_ene_re8_b_beast*` | Sinow Beat (EP2 reskin) | EP2 CCA/Spaceship | binary `b_beast/b_rdbeast/b_srdbeast` |
| `bm_ene_re8_merill_lia*` | Merillia / Meriltas | EP2 Temple | cls 0xA74040 |
| `bm_ene_recobox*` | Recobox / Recon | EP2 Spaceship | Phantasmal |
| `bm_ene_abgc01_wola_body*` | Hildebear / Hildeblue | EP1 Forest | binary @abgc01 + `odori` taunt anim |
| `bm_ene_biter_body*` | Astark biter subpart | EP4 Desert | Phantasmal Astark |
| `bm_ene_morfos*` | Morfos | EP2 Seabed | Phantasmal |
| `bm_ene_melissa.bml` | Delbiter (Melissa) | EP2 Seabed | entity_cls_table.h cls 0xA74A18 |
| `bm_ene_del_depth*` | Del Depth | EP2 Seabed | Phantasmal |
| `bm_ene_ill_gill.bml` | Ill Gill | EP2 Seabed | cls 0xA733B8 unitxt 82 |
| `bm_ene_epsilon.bml` | Epsilon | EP2 Spaceship | cls 0xA73BEC unitxt 84 |
| `bm_ene_gibbles*` | Gibbles | EP1 Mines | cls 0xA73CFC unitxt 61 |
| `bm_ene_gi_gue*` | Gi Gue | EP2 | cls 0xA73DB4 unitxt 55 |
| `bm_ene_nanodrago.bml` | Nano Dragon | EP1 Caves | cls 0xA74198 unitxt 15 |
| `bm_ene_lappy*` | Rag/Al/Saint/Egg/Hallo/Love Rappy | EP1/2/4 | newserv-sparse Rappy variants |
| `bm_ene_sandlappy.bml` | Sand Rappy | EP4 Desert | Phantasmal |
| `bm_ene_yowie.bml` | Yowie | EP4 | Phantasmal |
| `bm_ene_zu.bml` | Zu / Pazuzu | EP4 | cls 0xA749EC unitxt 94/95 |
| `bm_ene_boota.bml` | Boota / Ze Boota / Ba Boota | EP4 Crater | cls 0xA747F4 unitxt 96-98 |
| `bm_ene_golan.bml` | Goran / Pyro Goran / Goran Detonator | EP4 Desert | cls 0xA7490C unitxt 101-103 |
| `bm_ene_astark.bml` | Astark | EP4 Desert | cls 0xA747C8 unitxt 88 |
| `bm_ene_girtablulu.bml` | Girtablulu | EP4 Desert | Phantasmal |
| `bm_ene_dolphon.bml` | Dolmolm / Dolmdarl | EP4 Crater | entity_cls_table.h cls 0xA74828 |
| `bm_ene_common_all.bml` | (shared skeleton) | — | entity_cls_table.h cls 0xA740D8 |
| `bm_ene_npc_chao*` | Chao guest enemy | quest-only | binary @bm_ene_npc_chao |
| `bm_ene_npc_nights*` | NiGHTS guest | quest-only | binary @bm_ene_npc_nights |

#### 3.2.1 Legacy bm{N}_ enemy prefix family (NOT player!)

This is THE important resolution: the editor was treating `bm4_ps_*` and `bm7_ps_*` as Player Bodies because the predicate string-matched on `_ps_`. But these are **enemies**:

| Pattern | In-game | Notes |
|---|---|---|
| `bm4_ps_ma_body.bml` + ma_tail / mar / mb / mbr | **Sinow Beat / Sinow Gold (EP1 Caves)** | binary `BLUE | RED | TYPE [0/1]` selector + tail-attack `tlatk_bm4_ps_ma_tail.njm` + apear/kie animations. cls 0xA74658..0xA746B0 in NormalMob range. NOT a player asset. |
| `bm7_s_paa_body.bml` + pal / par | **Pan Arms (combined) / Hidoom / Migium** | psobb_full_entity_map.md cls 0xA742A8 + 0xA74370. paa = Pan Arms Assembled, pal=left-half, par=right-half |
| `bm9_s_meri_body.bml` | Mericarol body | binary @bm9_; pairs with bm_ene_bm9_s_mericarol |

### 3.3 Bosses

| Pattern | In-game | Episode/Area | Source |
|---|---|---|---|
| `bm_boss1_dragon*` | **Sil Dragon (EP1 Forest)** | EP1 Forest | binary `draroar.adx` / `dradeath.adx` |
| `bm_boss2_de_rol_le*` | De Rol Le | EP1 Caves | full_entity_map cls 0xA43CC8 area |
| `bm_boss3_volopt*` | Vol Opt ver. 2 | EP1 Mines | full_entity_map Vol Opt cluster 0xA447D4..0xA44A18 |
| `bm_boss5_gryphon.bml` | Gal Gryphon | EP2 Jungle | Phantasmal |
| `bm_boss7_crawfish.bml` | Olga Flow (Crawfish form) | EP2 Seabed | Phantasmal `OlgaFlow`; "crawfish" = dev name |
| `bm_boss7_de_rol_le_c.bml` | Barba Ray (De Rol Le clone) | EP2 Temple | Phantasmal `BarbaRay` |
| `bm_boss8_dragon.bml` | **Gol Dragon (EP4 Crater)** | EP4 Crater | binary `goldeath.adx` neighbor |
| `bm_obj_boss8_*` (demoroom/monitor/piller) | Gol Dragon subparts | EP4 Crater | psobb_binary_scan.md note: NOT Vol Opt despite naming |
| `bm_obj_ep4_boss09_*` | Saint-Million / Shambertin / Kondrieu subparts | EP4 final | Phantasmal (boss09 = EP4 final) |
| `bm_ene_boss09*` | Saint-Million / Shambertin / Kondrieu | EP4 final | Phantasmal |
| `boss06_plotfalz_dat.bml` + `darkfalz_dat.bml` | Dark Falz | EP1 Ruins | binary @darkfalz |

### 3.4 Objects (`bm_obj_*`, `bm_o_*`, `bm_fe_*`, `bm_fs_*`, `bm_fd_*`, `fe_obj_*`, `fs_obj_*`)

| Pattern | In-game | Source |
|---|---|---|
| `bm_obj_warpboss.bml` | **Boss Teleporter (generic)** | binary @0x538878 + `fs_obj_warp_dai_beam02` neighbors |
| `bm_obj_warpboss_ancient.bml` | **Dark Falz Boss Teleporter (Ruins)** | binary @0x538908 + `de_obj_df_warp_*` neighbors |
| `bm_obj_warpboss_jungle.bml` | **Gal Gryphon Boss Teleporter (EP2)** | binary @0x538974 + `fe_obj_warp4_*` neighbors |
| `bm_obj_warp_jung.bml` | Jungle Warp (regular) | binary @0x535098 |
| `bm_obj_warp_labo.bml` | Lab/CCA Warp | binary @0x535098 + `warp_citybeam` |
| `bm_obj_lobby_warp.bml` | Lobby Warp (Principal's Warp) | Phantasmal `LobbyWarpObject` |
| `bm_obj_meka_fish_0/1.bml` | **Mechanical Fish (EP2 Jungle/Seaside)** | binary @0x53159c + `uotitti_30_a` |
| `bm_obj_partition_army/norm.bml` | **Lab City partition wall** | binary @0x5375e4 + `labocity_partition_*` |
| `bm_obj_ruins_pillar.bml` | **Ruins Pillar** | binary @0x532e00 |
| `bm_obj_ruins_turiten.bml` | **Ruins Trap Plate** (turiten=swing) | binary @0x530b78 |
| `bm_obj_seabed_pillar.bml` | **Seabed Pillar** | binary @0x532e00 |
| `bm_obj_door_ruins.bml` | Ruins Door | filename |
| `bm_obj_door_space.bml` | Spaceship Door | filename |
| `bm_obj_ep4_*` | EP4 environment objects | filename (bee/bohu/cactus/crystal/door/flower/iwa=rock/saboten=cactus/sakana=fish) |
| `bm_obj_jungle_*` | EP2 Jungle objects | binary @0x53159c |
| `bm_obj_city_*` | Lobby/Pioneer 2 objects (board/sonic/event) | filename |
| `bm_obj_lobby_*` | Lobby decorations | filename |
| `bm_obj_labo_*` | Lab info-capsules | filename |
| `bm_obj_geenest.bml` | Gee Nest | bm3_s_nest neighbors |
| `bm_obj_comp_army/norm.bml` | EP2 CCA computer/console | filename |
| `bm_obj_desk.bml` / `bm_obj_table_labo.bml` | Lab interior props | filename |
| `bm_obj_beamdoa.bml` | Beam Door | filename |
| `bm_obj_hako01.bml` | Box / Container (hako=box) | filename |
| `bm_o_mine.bml` | Mine (De Rol Le projectile) | r2_psobb_findings.md mine vtable analysis |
| `bm_o_container_ancient.bml` | Ruins container | filename |
| `bm_o_door_seabed01*` | Seabed door | filename |
| `bm_o_door_vo_ship.bml` | Vol Opt ship door | vo_ship = Vol Opt vessel |
| `bm_o_explosive_machine.bml` | Explosive container (Mines) | filename |
| `bm_o_light_machine01.bml` | Light fixture | filename |
| `bm_o_rock_cave0[1-3].bml` | Cave rocks (3 variants) | filename |
| `bm_o_trap_ancient01.bml` | Ruins trap | filename |
| `bm_o_warp_ancient.bml` | Ruins area warp | filename |
| `bm_o_wreck_ancient.bml` | Ruins wreckage | filename |
| `bm_o_vs2.bml` | Versus-arena 2 | filename |
| `bm_o_boss4.bml` | (Unused / EP1 Mines amp?) | filename — boss4 doesn't exist canonically |
| `bm_o_bind.bml` | Bind/grapple effect | bind_laz_moto neighbor |
| `bm_o_abe_butterfly.bml` | Abe-event lobby butterfly | filename |
| `bm_sakana_obj_*` | Ambient fish (jungle/lobby/seabed) | sakana=fish |
| `bm_fd_obj_n_saku_*` | Force-field fences (saku) | filename + binary |
| `bm_fd_obj_n_switch.bml` | Force-field switch | filename |
| `bm_fe_obj_*` | Forest exterior props (capsule, door, hahen=fragment, liwa, tank_hikari=light tank) | binary |
| `bm_fs_obj_*` | Forest scene/lab props (aircon, cakeya=cake-shop, doorpanel, kanban=signboard, sensor, monitor, kurage=jellyfish, tombo=dragonfly) | binary |
| `fe_obj_*` (no `bm_` prefix) | Forest exterior shared models | direct .gj refs |
| `fs_obj_*` (no `bm_` prefix) | Forest scene shared models | direct .gj refs |
| `biri_ball.bml` | Electric ball (biri=zap) | binary @biri_ball |
| `abeniji_*` | Lobby rainbow object (niji=rainbow) | binary @abeniji_fe_obj001_niji |
| `bm_ply_photon_chair.bml` | Lobby photon chair | binary; pairs with fs_obj_lobby_isu (chair) |

### 3.5 NPCs (`bm_n_*`, `bm_npc_*`, `bm_nc*`)

| Pattern | In-game | Source |
|---|---|---|
| `bm_n_*_body.bml` | Pioneer 2 NPCs | binary; covers Hakase=Professor, Hisyo=Secretary, Kantei[bft]=Appraiser male/female/teen, Karen, Leo, Michel, Momoka, Nol, Paganini, Soutoku=Governor, Trunk, Nurse, Gunb=Gun-shop, Gunm=Gun-master, Elly, Delta |
| `bm_npc_*` | Pioneer 2 rigged NPCs | Hosa, Kenkyu(=Researcher), Momoka, Soutokufu (=Governor's office) |
| `bm_nc[man/woman/child/trunk]_body.bml` | Pioneer 2 civilians | filename |
| `bm_gunsinei_body.bml` | Gun-shop sub-NPC | filename |
| `bm_kenkyuw2_body.bml` | Researcher woman v2 | filename |
| `rico_body.bml` | **Red Ring Rico** (story NPC) | binary @rico_body |
| `rico_ring.bml` | Rico's red ring | binary |
| `NpcApcMot.bml` | NPC + APC (companion) motion library | filename — top-level |
| `npcplayerchar.dat` | NPC stats / loadouts | filename |

### 3.6 Items / Mags / Weapons (AFS archives)

| Archive | Index range | Contents | Source |
|---|---|---|---|
| `ItemModel.afs#NNNN` | ~0x000-0x100 | Weapons (geometry .nj) | Item-PMT mapping |
| `ItemModel.afs#NNNN` | ~0x100-0x200 | Armors / shields / units | Item-PMT mapping |
| `ItemModel.afs#NNNN` | ~0x200-0x290 | Tools / utilities | Item-PMT mapping |
| `ItemModel.afs#NNNN` | ~0x290+ | **Mags** (incl. rare mags from 0x28 within mag-class) | newserv-sparse `first_rare_mag_index=0x28` |
| `ItemModelEp4.afs#NNNN` | EP4-only | EP4 mags (Sato etc.) + EP4 weapons | filename |
| `ItemTexture*.afs#NNNN` | parallel index | XVMs paired 1:1 with ItemModel | server.py `_texture_index` |
| `ItemKT*.afs#NNNN` | — | Inventory icon atlases (XVM) | manifest.py:175 currently catches as "Weapon Textures" — KEEP |

**Note**: `ItemModelEp4.afs#0297_0297.nj` is a high-index entry that falls in the mag range. To definitively say "Sato" vs "another rare mag", parse the binary's ItemPMT.bin (in `data.gsl` or as a top-level resource) — out of scope for this database.

### 3.7 Effects

| Pattern | In-game |
|---|---|
| `bm_eff_*` (e.g. ice) | Effect geometry |
| `eff_boss09_saint_emilion.dat` | Saint-Million boss effect |
| `pm_mdl.bml` | Photon Magic projectiles (FARLLA/ESTLLA/GOLLA/PILLA/LEILLA/MYLLA — all techniques) |
| `particleentryaNN.dat` | Per-area particle tables |

### 3.8 Maps & terrain

| Pattern | In-game |
|---|---|
| `map_*.bin` | Area map geometry; `_e/_j` = locale, `_u` = Ultimate variant |
| `map_*.dat` | Area object/enemy placement |
| `map_*.evt` | Area event scripts |
| `cam_*.rel` | Boss / cinematic camera paths |
| `fogentry*.dat` | Per-area fog descriptors |
| `lightentry.bin` | Per-area lighting |
| `scene/*` | Per-area scene geometry/textures |

### 3.9 Set / Quest data

| Pattern | In-game |
|---|---|
| `SetDataTable*.rel` | SetData tables (Off/On = offline/online, Ulti = Ultimate) |
| `*.gsl` (gsl_set_object/enemy/event in binary) | Set/list containers |
| `data.gsl` | Master data container |

### 3.10 UI / Metadata

| Pattern | In-game |
|---|---|
| `TitleEP4.prs` | Real EP4 title-screen splash (per `psobb_splash_state_objects.md`) |
| `LogoEP4.prs` | Dead asset (NOT used) |
| `f256_*.prs` / `f128_*.prs` / `f512_*.xvm` | Font/UI/portrait atlases |
| `unitxt_*.prs` | Localized strings (entity name table read by `0x007879D0`) |
| `textjapanese.pr2` / `.pr3` | Japanese text data |
| `ws_data_jp.bin` / `_us.bin` | Word-Select chat dictionaries |
| `ggerr_*.bin` | Error message tables |
| `help*.png` / `help*.lst` | Help overlays + indices |
| `lobby_billboard*` / `no_lobby_billboard*` | Lobby billboards |
| `title2.xvm` / `ccconsole_j.xvm` | Title / cinematic-console textures |
| `indtex.xvr` / `indirect_base.xvr` | Indirect rendering textures |
| `texturejapanese.xvm` | Japanese-text texture |
| `obj_lobby_main.xvm` / `obj_boss1_common_a.xvm` | Lobby + Sil Dragon shared textures |
| `o_grass_jungle.xvm` / `o_rock_jungle_*.xvm` | Jungle terrain textures |
| `vssver.scc` | Visual SourceSafe vestige (devel leftover) |
| `newserv-test-bb.txt` | newserv test marker |
| `smutdata.prs` | Profanity filter |
| `quickref_ja.lst` | Japanese quick-reference text |
| `re_b_mark_base.bml` | Ruins boss-room pre-warp markers |

### 3.11 Cinematics / Audio

| Pattern | In-game |
|---|---|
| `openning_e.pae` / `openning_j.pae` | Opening cinematic (en/ja) |
| `ending_jp.pae` | Ending cinematic |
| `onlineending_dat.bml` | Online ending data |
| `*.adx` (e.g. draroar.adx, goldeath.adx) | Boss roars / death sounds |
| `ogg/*` | BGM tracks |
| `sound/*` | SFX |

## 4. Resolved miscategorizations (the user's 5 reports)

### 4.1 `bm4_ps_ma_body.bml`

- **Before**: Player Bodies (matched `n.startswith('bm4_ps_')`)
- **After**: Enemies / EP1 Caves
- **In-game name**: Sinow Beat / Sinow Gold (legacy 'ma'/'mb' subparts)
- **Evidence**: PsoBB.exe @0x528200 has the marker block `BLUE | RED | TYPE [0/1]` (rare/normal selector), tail-attack animation `tlatk_bm4_ps_ma_tail.njm`, and disappear/reappear animations (`apear`, `kie`) consistent with Sinow's stealth-warp behavior. The cls cluster 0xA74658..0xA746B0 (per `entity_cls_table.h:470-475`) sits in the NormalMob range and lives adjacent to me3_shinowa/zoa (the EP2 Sinow reskins) in the binary's data layout.
- **Editor fix**: `bm4_ps_*` and `bm7_ps_*` belong to the Enemies bucket, NOT Player Bodies. The current rule at `manifest.py:181-182` is **wrong**.

### 4.2 `ItemModelEp4.afs#0297_0297.nj`

- **Before**: Weapons / Items (correct general bucket)
- **After**: Items / Mags
- **In-game name**: EP4 Mag (likely Sato/rare)
- **Evidence**: AFS index 0x297 (663 decimal) sits in the mag-range slice. ItemPMT.bin's mag table starts the rare-mag subset at 0x28 within mag-class.
- **Editor fix**: Add a numeric-range subcategory rule for `ItemModelEp4.afs#02[7-9]?_*` → "Mags".

### 4.3 `bm_obj_warpboss_ancient.bml`

- **Before**: Objects (matches `n.startswith('bm_obj_')` only)
- **After**: Objects / Boss-area warp / Dark Falz Boss Teleporter (Ruins)
- **Evidence**: PsoBB.exe @0x538908 surrounded by `de_obj_df_warp_beam.gj`, `de_obj_df_warp_sbeam.gj` (df = Dark Falz). Sibling family: `bm_obj_warpboss.bml` (generic), `bm_obj_warpboss_jungle.bml` (Gal Gryphon EP2).
- **Editor fix**: Add specific subcategory rules for `bm_obj_warpboss_*` to label the boss area.

### 4.4 `bm_boss1_dragon.bml` vs `bm_boss8_dragon.bml`

- `bm_boss1_dragon.bml` = **Sil Dragon (EP1 Forest)**: PsoBB.exe @0x4faeec with `draroar.adx`, `dradeath.adx`.
- `bm_boss8_dragon.bml` = **Gol Dragon (EP4 Crater)**: PsoBB.exe @0x5075b8 with `goldeath.adx` neighbor (gold dragon).
- **Editor fix**: split into per-boss subcategory.

### 4.5 Area-specific objects

| File | Was | Now |
|---|---|---|
| `bm_obj_meka_fish_0/1.bml` | Objects | Objects / EP2 Jungle/Seaside / Mechanical Fish |
| `bm_obj_partition_army/norm.bml` | Objects | Objects / EP2 Lab City / Partition Wall |
| `bm_obj_ruins_pillar.bml` | Objects | Objects / EP1 Ruins / Pillar |
| `bm_obj_ruins_turiten.bml` | Objects | Objects / EP1 Ruins / Trap Plate |
| `bm_obj_seabed_pillar.bml` | Objects | Objects / EP2 Seabed / Pillar |
| `bm_obj_warp_jung.bml` | Objects | Objects / Area warps / Jungle Warp |
| `bm_obj_warp_labo.bml` | Objects | Objects / Area warps / Lab/CCA Warp |

## 5. Patch sketch for `manifest.py`

Refactor the lambda-list categorizer to consume the JSON db. **DO NOT APPLY** — this is a sketch for an implementing agent.

```python
# manifest.py — proposed refactor (rule loader + categorizer)

import json, fnmatch
from pathlib import Path
from typing import Optional

_CATEGORY_DB_PATH = Path(__file__).parent / "_reports" / "categorization_db.json"
_CATEGORY_DB = None  # lazy-loaded

def _load_category_db() -> dict:
    global _CATEGORY_DB
    if _CATEGORY_DB is None:
        try:
            with open(_CATEGORY_DB_PATH, "r", encoding="utf-8") as f:
                _CATEGORY_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            _CATEGORY_DB = {"rules": [], "fallback": "Uncategorized"}
    return _CATEGORY_DB

def _match_pattern(name: str, parent: str, parent_archive: str, pattern: str) -> bool:
    """Match a glob-style pattern against name / parent_archive / parent path.
    Patterns containing 'afs#' match against the full synthesised path.
    Patterns ending '*' match prefix-style on basename.
    """
    name_l = name.lower()
    archive_l = parent_archive.lower()
    parent_l = parent.lower()
    pat_l = pattern.lower()
    # AFS-inner pattern (e.g. "ItemModel.afs#*"): match against parent_archive
    if "afs#" in pat_l:
        archive_part = pat_l.split("#")[0]
        return archive_l.startswith(archive_part)
    # Path-fragment pattern (e.g. "scene/*", "ogg/*")
    if "/" in pat_l:
        return fnmatch.fnmatchcase(f"{parent_l}/{name_l}", pat_l) or parent_l.startswith(pat_l.rstrip("/*"))
    # Plain glob on basename
    return fnmatch.fnmatchcase(name_l, pat_l)

def infer_category_from_db(rel_path: str, parent_archive: Optional[str] = None) -> Optional[dict]:
    """Replacement for `infer_category` that returns a dict with category +
    subcategory + in_game_name, all sourced from the JSON db."""
    if not rel_path:
        return None
    db = _load_category_db()
    name = Path(rel_path).name.lower()
    parent = "/".join(rel_path.replace("\\", "/").split("/")[:-1]).lower()
    arch = (parent_archive or "").lower()
    for rule in db.get("rules", []):
        if _match_pattern(name, parent, arch, rule.get("pattern", "")):
            return {
                "category": rule.get("category"),
                "subcategory": rule.get("subcategory"),
                "in_game_name": rule.get("in_game_name"),
            }
    return None

# Then update classify() / classify_inner_blob() to call infer_category_from_db()
# and emit `category` + `subcategory` + `in_game_name` rather than the single
# `inferred_category` field. Existing consumers reading `inferred_category`
# continue working; new consumers can read the richer fields.
```

### Diff vs current `manifest.py:163-219`:

```diff
-_INFERRED_CATEGORY_RULES: list[tuple[callable, str]] = [
-    (lambda n, p, a: a.startswith("itemmodel"),    "Weapons / Items"),
-    (lambda n, p, a: a.startswith("itemtexture"),  "Weapon Textures"),
-    (lambda n, p, a: a.startswith("itemkt"),       "Weapon Textures"),
-    (lambda n, p, a: a.startswith("pl") and a.endswith("tex.afs"), "Player Textures"),
-    (lambda n, p, a: n.startswith("bm_boss"),      "Bosses"),
-    (lambda n, p, a: n.startswith("bm_ene_"),      "Enemies"),
-    (lambda n, p, a: n.startswith("bm_eff_"),      "Effects"),
-    (lambda n, p, a: (n.startswith("bm_ps_") or n.startswith("bm4_ps_") or
-                      n.startswith("bm7_ps_")),    "Player Bodies"),  # WRONG
-    (lambda n, p, a: (n.startswith("bm4_oh_") or n.startswith("bm7_oh_") or
-                      n.startswith("bm_oh_")),     "Player Headgear"),
-    (lambda n, p, a: n.startswith("obj_") or n.startswith("bm_obj_"),
-                                                   "Objects"),
-    (lambda n, p, a: n.startswith("set_") or n.startswith("set9_"),
-                                                   "Set Pieces"),
-    (lambda n, p, a: n.startswith("np_") or n.startswith("bm_np_"),
-                                                   "NPCs"),
-    (lambda n, p, a: n.startswith("map_"),         "Maps / Terrain"),
-    (_is_in_scene_dir,                             "Maps / Terrain"),
-]
-
-def infer_category(rel_path: str, parent_archive: Optional[str] = None) -> Optional[str]:
-    if not rel_path:
-        return None
-    name = Path(rel_path).name.lower()
-    parent = "/".join(rel_path.replace("\\", "/").split("/")[:-1]).lower()
-    arch = (parent_archive or "").lower()
-    for pred, label in _INFERRED_CATEGORY_RULES:
-        try:
-            if pred(name, parent, arch):
-                return label
-        except Exception:
-            continue
-    return None
+# Categorizer is JSON-driven now — see categorization_db.json + helpers below.
+from . import _category_db  # new module wraps the JSON loader + matcher
+
+def infer_category(rel_path: str, parent_archive: Optional[str] = None) -> Optional[str]:
+    """Backwards-compatible wrapper — returns category-string only."""
+    info = _category_db.lookup(rel_path, parent_archive)
+    return info.get("category") if info else None
+
+def infer_category_full(rel_path: str, parent_archive: Optional[str] = None) -> Optional[dict]:
+    """New richer accessor — returns {category, subcategory, in_game_name}."""
+    return _category_db.lookup(rel_path, parent_archive)
```

The patch keeps the existing `inferred_category` field shape intact (so `tree.js` and other consumers don't break) but adds an `inferred_subcategory` and `in_game_name` companion field on the entry. Tree consumers can opt into the richer display when ready.

## 6. Coverage summary

- **Prefixes resolved with high confidence**: 73
- **Prefixes resolved with moderate confidence (educated inference)**: 6 (`bm4_ps_*` family, `bm_o_boss4`, `re7_berura` interpretation, `bm_n_*` per-NPC subtype mapping, EP4 final-boss subpart split, `dolphon`)
- **UNKNOWN** (no rule, falls through to `Uncategorized`): 0 — every observed file in the live tree (`~/PSOBB.IO/data/`, 883 files) matches at least one rule.
- The `bm4_ps_ma_body.bml` exact identity (Sinow Beat vs Sinow Gold) requires runtime cls observation to fully disambiguate, but BOTH variants fall under "EP1 Caves enemy" so the editor categorization is correct.
