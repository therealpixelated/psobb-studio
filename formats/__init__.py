# Package marker for the PSOBB Modding Suite asset-format parsers.
#
# Submodules:
#   iff           - little-endian PSO IFF chunk reader (NJCM/NJTL/NMDM/POF0)
#   afs           - Sega AFS archive reader
#   bml           - Binary Model Library extractor (Phase B precondition)
#   match         - texture<->model multi-rule matcher (R1..R6)
#   prs           - PRS (Sega LZSS) encoder + decoder, pure Python
#   battle_param  - BattleParamEntry*.dat parser + serializer
#   audio_pac     - byte-exact .pac PCM SFX-bank codec (pure Python)
#   audio_codec   - optional ffmpeg-backed .ogg/.sfd decode + .ogg encode
#   audio         - thin facade re-exporting audio_pac + audio_codec
#
# All submodules are pure Python and have no third-party dependencies
# (audio_codec shells out to a system ffmpeg only when present; absence is
# degraded to a 501-mappable signal, never a hard failure).
