"""Pure-Python PRS (Sega LZSS) encoder and decoder.

PRS is the LZSS-with-tag-bytes compression Sega used across the Saturn,
Dreamcast and PSO line. PSOBB uses it for nearly every container payload
(BMLs hold PRS-compressed model bodies, the splash screens are PRS,
ItemPMT.prs is PRS, BattleParamEntry's stream is *not* PRS but everything
else around it is). The format has three back-reference encodings:

* SHORT_COPY  -- 4 control bits + 1 data byte, offset in [-0x100, -1],
                 size in 2..5.
* LONG_COPY   -- 2 control bits + 2 data bytes, offset in [-0x1FFF, -1],
                 size in 2..9 (low 3 bits of word).
* EXTENDED_COPY -- 2 control bits + 3 data bytes, offset in [-0x1FFF, -1],
                 size in 1..0x100.

End of stream is signaled by a LONG_COPY whose embedded offset is zero.

This module ports the algorithm from newserv's MIT-licensed
``src/Compression.cc`` (see LICENSES.md). The encoder uses a graph-based
shortest-path approach (newserv ``prs_compress_optimal``) plus a fast
greedy mode. The decoder is straightforward.

Pure Python; no third-party dependencies. Performance is acceptable
(small-MB files in a fraction of a second). For the intended editor
workflow -- compress one or two server-side blobs per export -- it is
fast enough; the AFS / BML pipeline can call this directly without
shelling out to PuyoToolsCli.

CLI:
    python -m formats.prs encode <input> <output>
    python -m formats.prs decode <input> <output>
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (from newserv Compression.cc)
# ---------------------------------------------------------------------------
SHORT_COPY_MIN_SIZE = 2
SHORT_COPY_MAX_SIZE = 5
SHORT_COPY_MIN_OFFSET = -0x100  # offset is negative (back-reference)
LONG_COPY_MIN_SIZE = 2
LONG_COPY_MAX_SIZE = 9
EXTENDED_COPY_MIN_SIZE = 1
EXTENDED_COPY_MAX_SIZE = 0x100

SHORT_WINDOW = 0x100
LONG_WINDOW = 0x1FFF


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def decompress(data: bytes, max_output_size: int = 0, *, tolerant: bool = False) -> bytes:
    """Decompress PRS-encoded ``data`` to bytes.

    The control-byte stream is read one bit at a time; data bytes are
    interleaved between control bytes (see newserv ``prs_decompress``).

    Parameters
    ----------
    tolerant : bool
        When True, end-of-stream truncation (missing terminator marker)
        and partial back-reference reads return whatever output bytes
        have been produced so far instead of raising. Used by
        ``formats/bml.py::extract_bml_texture`` to recover Sega's
        "stub XVM" payloads — 10 BMLs in PSOBB.IO ship a 52,570-byte
        compressed XVM blob whose decompressed output is a valid
        XVMH archive (262,283 bytes) but whose PRS stream lacks a
        proper end marker. The runtime tolerates this because its
        texture allocator never reaches end-of-stream; we mirror that
        behavior. Default (False) preserves strict-mode behavior:
        truncated streams raise ``ValueError``.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("decompress: input must be bytes-like")
    src = bytes(data)
    n = len(src)
    out = bytearray()

    # Bit reader using a 16-bit register: high byte == "more bits left"
    # marker (matches newserv's ControlStreamReader trick).
    bits = 0
    pos = 0

    def read_u8() -> int:
        nonlocal pos
        if pos >= n:
            raise ValueError("decompress: unexpected EOF")
        v = src[pos]
        pos += 1
        return v

    def read_bit() -> int:
        nonlocal bits, pos
        if not (bits & 0x0100):
            bits = 0xFF00 | read_u8()
        v = bits & 1
        bits >>= 1
        return v

    try:
        while pos < n or (bits & 0x0100):
            try:
                ctrl = read_bit()
            except ValueError:
                break  # ran out of bits: the stream is unterminated
            if ctrl:
                # Literal byte
                if max_output_size and len(out) >= max_output_size:
                    raise ValueError("decompress: max_output_size exceeded")
                out.append(read_u8())
                continue

            # Back-reference. Next control bit decides long-vs-short.
            kind = read_bit()
            if kind:
                # LONG_COPY or EXTENDED_COPY
                a = read_u8()
                a |= read_u8() << 8
                offset = (a >> 3) | (~0x1FFF)
                if offset == ~0x1FFF:
                    # offset bits all zero -> end of stream marker.
                    break
                count_low = a & 7
                if count_low:
                    count = count_low + 2
                else:
                    count = read_u8() + 1
            else:
                # SHORT_COPY
                count = (read_bit() << 1)
                count = (count | read_bit()) + 2
                offset = read_u8() | (~0xFF)

            read_offset = len(out) + offset
            if read_offset < 0:
                raise ValueError("decompress: backreference before start of output")
            for _ in range(count):
                if max_output_size and len(out) >= max_output_size:
                    raise ValueError("decompress: max_output_size exceeded")
                if read_offset >= len(out):
                    if tolerant:
                        return bytes(out)
                    raise ValueError(
                        f"decompress: backreference past end of output at "
                        f"read_offset {read_offset}"
                    )
                out.append(out[read_offset])
                read_offset += 1
    except ValueError:
        if tolerant:
            # Stream ran out mid-instruction; return whatever we
            # successfully produced so the caller can salvage the
            # XVMH header that almost always precedes the corruption.
            return bytes(out)
        raise

    return bytes(out)


# ---------------------------------------------------------------------------
# Bit-interleaved writer (matches newserv ``LZSSInterleavedWriter``)
# ---------------------------------------------------------------------------
class _LZSSWriter:
    """Bit-interleaved PRS writer.

    Layout: one control byte holds 8 bits, MSB-of-bit-emission ordering.
    Data bytes accumulate in a buffer that is emitted after the 8th
    control bit lands. ``flush_if_ready`` is called by the encoder
    after each emitted command and is a no-op until 8 bits have been
    written, at which point the control byte + buffered data bytes are
    flushed to the output and a new control byte is opened.
    """

    __slots__ = ("out", "buf", "buf_offset", "next_control_bit")

    def __init__(self) -> None:
        self.out = bytearray()
        # buf[0] is the in-progress control byte; data bytes accumulate
        # at buf[1..].
        self.buf = bytearray(0x19)
        self.buf_offset = 1
        self.next_control_bit = 1  # 1, 2, 4, 8, ... 0x80, then 0 -> flush

    def flush_if_ready(self) -> None:
        if self.next_control_bit == 0:
            self.out.extend(self.buf[: self.buf_offset])
            self.buf[0] = 0
            self.buf_offset = 1
            self.next_control_bit = 1

    def close(self) -> bytes:
        if self.buf_offset > 1 or self.next_control_bit != 1:
            self.out.extend(self.buf[: self.buf_offset])
        return bytes(self.out)

    def write_control(self, v: bool) -> None:
        if self.next_control_bit == 0:
            raise RuntimeError("write_control with no space")
        if v:
            self.buf[0] |= self.next_control_bit & 0xFF
        self.next_control_bit = (self.next_control_bit << 1) & 0x1FF
        if self.next_control_bit == 0x100:
            # Past the 8th bit -> overflow flag
            self.next_control_bit = 0

    def write_data(self, v: int) -> None:
        self.buf[self.buf_offset] = v & 0xFF
        self.buf_offset += 1

    def size(self) -> int:
        return len(self.out) + self.buf_offset


# ---------------------------------------------------------------------------
# Greedy encoder (~quality, fast path)
# ---------------------------------------------------------------------------
def compress(data: bytes) -> bytes:
    """PRS-compress ``data`` with a fast greedy match search.

    Yields valid PRS for any input. Output is typically slightly larger
    than ``compress_optimal`` (a few percent) but encodes ~10x faster.

    Algorithm: for each output position, scan the back-window
    ``[-0x1FFF, -1]`` for the longest match >= 1; emit the best
    fit (SHORT/LONG/EXTENDED) and advance. Falls back to a literal byte
    for matches < 2.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("compress: input must be bytes-like")
    src = bytes(data)
    n = len(src)
    if n == 0:
        return _emit_eof()

    w = _LZSSWriter()

    # Hash-chain LZ77: classic Snappy/zlib trick. Hash 3-byte sequences
    # to a small set of "head" positions; from there, follow a "prev"
    # chain backwards through earlier hash collisions. Each chain walk
    # is bounded by MAX_CHAIN to prevent quadratic blowup.
    HASH_BITS = 16
    HASH_SIZE = 1 << HASH_BITS
    HASH_MASK = HASH_SIZE - 1
    NIL = -1
    head = [NIL] * HASH_SIZE
    prev = [NIL] * n  # prev[i] = previous position with same 3-byte hash

    EXT_MAX = EXTENDED_COPY_MAX_SIZE
    MAX_CHAIN = 32  # bounded chain depth -- cap for speed

    def _hash3(a: int, b: int, c: int) -> int:
        # FNV-style 24->16 bit fold
        return ((a * 2654435761) ^ (b << 8) ^ c) & HASH_MASK

    pos = 0
    while pos < n:
        avail = n - pos
        limit = EXT_MAX if avail >= EXT_MAX else avail

        best_off = 0
        best_size = 0

        if limit >= 3:
            h = _hash3(src[pos], src[pos + 1], src[pos + 2])
            cand = head[h]
            steps = 0
            window_start = pos - LONG_WINDOW
            while cand != NIL and cand >= window_start and steps < MAX_CHAIN:
                # cand is older than pos. Compare from cand and pos.
                if src[cand] == src[pos] and src[cand + 1] == src[pos + 1] and src[cand + 2] == src[pos + 2]:
                    # Extend match length using bulk-equality probes.
                    mlen = 3
                    # Bulk extend: try 16-byte, 8-byte, 4-byte chunks then byte
                    while mlen + 16 <= limit and src[cand + mlen:cand + mlen + 16] == src[pos + mlen:pos + mlen + 16]:
                        mlen += 16
                    while mlen + 4 <= limit and src[cand + mlen:cand + mlen + 4] == src[pos + mlen:pos + mlen + 4]:
                        mlen += 4
                    while mlen < limit and src[cand + mlen] == src[pos + mlen]:
                        mlen += 1
                    if mlen > best_size:
                        best_size = mlen
                        best_off = cand - pos
                        if mlen >= limit:
                            break
                cand = prev[cand]
                steps += 1
        elif limit == 2:
            # Try a single hash lookup but with a 2-byte prefix using
            # a separate scheme: just rfind the 2 bytes in window.
            window_start = pos - LONG_WINDOW
            if window_start < 0:
                window_start = 0
            probe = src[pos:pos + 2]
            p = src.rfind(probe, window_start, pos + 1)
            if p >= 0 and p < pos:
                best_size = 2
                best_off = p - pos

        # Decide how to emit. Note: LONG_COPY's low 3 bits encode (size-2),
        # so size==2 -> bits all zero -> the decoder treats those bits as
        # the EXTENDED_COPY escape. LONG_COPY therefore can only carry
        # sizes 3..9. Size==2 with an offset outside SHORT_COPY's reach
        # falls back to a literal byte (matches newserv greedy).
        emit_long_min = 3
        if best_size < 2:
            # Literal byte
            w.write_control(True)
            w.write_data(src[pos])
            w.flush_if_ready()
            advance = 1
        elif (
            best_off >= SHORT_COPY_MIN_OFFSET
            and SHORT_COPY_MIN_SIZE <= best_size <= SHORT_COPY_MAX_SIZE
        ):
            _emit_short_copy(w, best_off, best_size)
            advance = best_size
        elif best_size < emit_long_min:
            # Size==2 but offset doesn't fit SHORT_COPY -> literal
            w.write_control(True)
            w.write_data(src[pos])
            w.flush_if_ready()
            advance = 1
        elif emit_long_min <= best_size <= LONG_COPY_MAX_SIZE:
            _emit_long_copy(w, best_off, best_size)
            advance = best_size
        else:
            _emit_extended_copy(w, best_off, best_size)
            advance = best_size

        # Insert hashes for every byte we just consumed. Keep this loop
        # tight; hash chaining is the main per-byte cost.
        end_insert = pos + advance
        if end_insert > n - 2:
            end_insert = n - 2
        i = pos
        while i < end_insert:
            h = _hash3(src[i], src[i + 1], src[i + 2])
            prev[i] = head[h]
            head[h] = i
            i += 1
        pos += advance

    # Terminator: long_copy with offset==0
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(True)
    w.write_data(0)
    w.write_data(0)
    return w.close()


def _emit_eof() -> bytes:
    """Emit just the PRS terminator (used for empty inputs)."""
    w = _LZSSWriter()
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(True)
    w.write_data(0)
    w.write_data(0)
    return w.close()


def _emit_short_copy(w: _LZSSWriter, offset: int, size: int) -> None:
    encoded_size = size - 2  # 0..3
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(bool(encoded_size & 2))
    w.flush_if_ready()
    w.write_control(bool(encoded_size & 1))
    w.write_data(offset & 0xFF)
    w.flush_if_ready()


def _emit_long_copy(w: _LZSSWriter, offset: int, size: int) -> None:
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(True)
    a = ((offset & 0x1FFF) << 3) | ((size - 2) & 7)
    w.write_data(a & 0xFF)
    w.write_data((a >> 8) & 0xFF)
    w.flush_if_ready()


def _emit_extended_copy(w: _LZSSWriter, offset: int, size: int) -> None:
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(True)
    a = (offset & 0x1FFF) << 3  # low 3 bits = 0 -> "extended"
    w.write_data(a & 0xFF)
    w.write_data((a >> 8) & 0xFF)
    w.write_data((size - 1) & 0xFF)
    w.flush_if_ready()


# ---------------------------------------------------------------------------
# Optimal (shortest-path) encoder
# ---------------------------------------------------------------------------
# Bit costs per command type (from newserv prs_compress_optimal):
#   LITERAL        :  9 bits (1 control bit + 8 data bits)
#   SHORT_COPY     : 12 bits (4 control bits + 1 byte = 4 + 8)
#   LONG_COPY      : 18 bits (2 control bits + 2 bytes = 2 + 16)
#   EXTENDED_COPY  : 26 bits (2 control bits + 3 bytes = 2 + 24)
_COST_LITERAL = 9
_COST_SHORT = 12
_COST_LONG = 18
_COST_EXTENDED = 26


def _build_window_index(src: bytes, max_chain: int = 64) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Build per-position best-match candidates for SHORT, LONG, EXTENDED windows.

    Returns three parallel lists (one entry per source position):
        short[i]    = (offset, size)  for SHORT window  (0x100, max 5)
        long[i]     = (offset, size)  for LONG window   (0x1FFF, max 9)
        extended[i] = (offset, size)  for LONG window   (0x1FFF, max 0x100)

    Offsets are negative (back-references). A size of 0 means no usable
    match was found for that command type at that position.

    Uses a hash-chain index (3-byte rolling hash over LONG_WINDOW) to
    avoid quadratic behavior on highly-repetitive input. ``max_chain``
    bounds candidate scan depth -- 64 is a good quality/speed balance.
    """
    n = len(src)

    short_arr: List[Tuple[int, int]] = [(0, 0)] * n
    long_arr: List[Tuple[int, int]] = [(0, 0)] * n
    extended_arr: List[Tuple[int, int]] = [(0, 0)] * n

    HASH_BITS = 16
    HASH_SIZE = 1 << HASH_BITS
    HASH_MASK = HASH_SIZE - 1
    NIL = -1
    head = [NIL] * HASH_SIZE
    prev = [NIL] * n

    def _hash3(a: int, b: int, c: int) -> int:
        return ((a * 2654435761) ^ (b << 8) ^ c) & HASH_MASK

    EXT_MAX = EXTENDED_COPY_MAX_SIZE

    for i in range(n):
        avail = n - i
        limit_ext = EXT_MAX if avail >= EXT_MAX else avail
        # Bounds for each window.
        min_short = i - SHORT_WINDOW
        min_long = i - LONG_WINDOW

        best_short_off = 0
        best_short_size = 0
        best_long_size = 0
        best_long_off = 0
        best_ext_size = 0
        best_ext_off = 0

        if limit_ext >= 3 and i + 2 < n:
            h = _hash3(src[i], src[i + 1], src[i + 2])
            cand = head[h]
            steps = 0
            while cand != NIL and cand >= min_long and steps < max_chain:
                # Verify hash collision with explicit compare
                if src[cand] == src[i] and src[cand + 1] == src[i + 1] and src[cand + 2] == src[i + 2]:
                    mlen = 3
                    while mlen + 16 <= limit_ext and src[cand + mlen:cand + mlen + 16] == src[i + mlen:i + mlen + 16]:
                        mlen += 16
                    while mlen + 4 <= limit_ext and src[cand + mlen:cand + mlen + 4] == src[i + mlen:i + mlen + 4]:
                        mlen += 4
                    while mlen < limit_ext and src[cand + mlen] == src[i + mlen]:
                        mlen += 1

                    offset = cand - i  # negative

                    # SHORT window scoring (offset >= -0x100, size 2..5)
                    if cand >= min_short and mlen >= SHORT_COPY_MIN_SIZE:
                        ssize = mlen if mlen <= SHORT_COPY_MAX_SIZE else SHORT_COPY_MAX_SIZE
                        if ssize > best_short_size:
                            best_short_size = ssize
                            best_short_off = offset

                    # LONG window scoring
                    if mlen >= LONG_COPY_MIN_SIZE:
                        lsize = mlen if mlen <= LONG_COPY_MAX_SIZE else LONG_COPY_MAX_SIZE
                        if lsize > best_long_size or (lsize == best_long_size and offset > best_long_off):
                            best_long_size = lsize
                            best_long_off = offset

                    # EXTENDED window scoring
                    if mlen >= EXTENDED_COPY_MIN_SIZE:
                        esize = mlen if mlen <= EXTENDED_COPY_MAX_SIZE else EXTENDED_COPY_MAX_SIZE
                        if esize > best_ext_size or (esize == best_ext_size and offset > best_ext_off):
                            best_ext_size = esize
                            best_ext_off = offset

                cand = prev[cand]
                steps += 1

        # Also catch 2-byte matches in SHORT window (hash needs 3 bytes
        # so we miss size-2 hits; do a single rfind for short window).
        if limit_ext >= 2 and best_short_size < 2:
            ws = i - SHORT_WINDOW
            if ws < 0:
                ws = 0
            probe2 = src[i:i + 2]
            p = src.rfind(probe2, ws, i + 1)
            if p >= 0 and p < i:
                best_short_size = 2
                best_short_off = p - i

        short_arr[i] = (best_short_off, best_short_size)
        long_arr[i] = (best_long_off, best_long_size) if best_long_size else (0, 0)
        extended_arr[i] = (best_ext_off, best_ext_size) if best_ext_size else (0, 0)

        # Insert i into hash chain (only if room for 3 bytes).
        if i + 2 < n:
            h = _hash3(src[i], src[i + 1], src[i + 2])
            prev[i] = head[h]
            head[h] = i

    return short_arr, long_arr, extended_arr


def compress_optimal(data: bytes) -> bytes:
    """PRS-compress ``data`` using a shortest-path search.

    Mirrors newserv ``prs_compress_optimal`` (MIT). For each source
    position we know the longest match in each of the three encoding
    windows; we then run a Bellman-style relaxation over the source
    array (literal / short / long / extended edges) to pick the
    bit-cheapest cover. Backtrack and emit.

    Slower than ``compress`` (roughly 2-3x on small inputs, 5-10x on
    large) but generally produces output 1-3% smaller. Use this for
    artifact-grade output (ItemPMT-bb-v4.prs, ASI-shipped blobs);
    use ``compress`` for one-shot edits where size is uncritical.

    Args:
        data: raw bytes.
    Returns:
        PRS-compressed bytes.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("compress_optimal: input must be bytes-like")
    src = bytes(data)
    n = len(src)
    if n == 0:
        return _emit_eof()

    short_arr, long_arr, ext_arr = _build_window_index(src)

    # Shortest-path: nodes[0..n]. nodes[k] = best total bits to reach
    # position k from start. The terminator at position n is paid by
    # the closing 18 bits (one LONG_COPY w/ offset 0). Use a large
    # sentinel for "unreached" cells; Python ints are arbitrary
    # precision so there's no overflow risk.
    INF = 1 << 60
    bits = [INF] * (n + 1)
    bits[0] = 18  # closing terminator (paid up-front, matches newserv comment)

    # For each node, remember (from_offset, command_type) so we can
    # backtrack. Command types: 0=lit, 1=short, 2=long, 3=ext
    from_off = [0] * (n + 1)
    from_cmd = [0] * (n + 1)

    for z in range(n):
        if bits[z] == INF:
            continue
        base = bits[z]

        # LITERAL: cost +9, advance by 1.
        nb = base + _COST_LITERAL
        if z + 1 <= n and bits[z + 1] > nb:
            bits[z + 1] = nb
            from_off[z + 1] = z
            from_cmd[z + 1] = 0

        # SHORT_COPY: any size in 2..5 within short window.
        sshort = short_arr[z][1]
        if sshort >= SHORT_COPY_MIN_SIZE:
            nb = base + _COST_SHORT
            for x in range(SHORT_COPY_MIN_SIZE, sshort + 1):
                if bits[z + x] > nb:
                    bits[z + x] = nb
                    from_off[z + x] = z
                    from_cmd[z + x] = 1

        # LONG_COPY: any size in 3..9 within long window. Note
        # newserv uses LONG_COPY for 3..9 (it has min size 2 but 2-byte
        # matches go through SHORT_COPY when possible).
        slong = long_arr[z][1]
        if slong >= 3:
            nb = base + _COST_LONG
            # Start at 3: 2-byte matches cost the same as a SHORT_COPY
            # if the offset fits, and SHORT_COPY is preferred. (newserv
            # uses x=3..long for the same reason -- see prs_compress_optimal.)
            for x in range(3, slong + 1):
                if bits[z + x] > nb:
                    bits[z + x] = nb
                    from_off[z + x] = z
                    from_cmd[z + x] = 2

        # EXTENDED_COPY: any size in 1..0x100 within long window.
        sext = ext_arr[z][1]
        if sext >= EXTENDED_COPY_MIN_SIZE:
            nb = base + _COST_EXTENDED
            for x in range(EXTENDED_COPY_MIN_SIZE, sext + 1):
                if bits[z + x] > nb:
                    bits[z + x] = nb
                    from_off[z + x] = z
                    from_cmd[z + x] = 3

    # Backtrack to build "to_offset" array.
    to_offset = [0] * (n + 1)
    z = n
    while z > 0:
        prev = from_off[z]
        to_offset[prev] = z
        z = prev

    # Walk forward and emit.
    w = _LZSSWriter()
    pos = 0
    while pos < n:
        nxt = to_offset[pos]
        size = nxt - pos
        cmd = from_cmd[nxt]
        if cmd == 0:
            # LITERAL
            w.write_control(True)
            w.write_data(src[pos])
            w.flush_if_ready()
        elif cmd == 1:
            # SHORT
            offset = short_arr[pos][0]
            _emit_short_copy(w, offset, size)
        elif cmd == 2:
            # LONG
            offset = long_arr[pos][0]
            _emit_long_copy(w, offset, size)
        elif cmd == 3:
            # EXTENDED
            offset = ext_arr[pos][0]
            _emit_extended_copy(w, offset, size)
        else:
            raise RuntimeError(f"compress_optimal: invalid cmd {cmd}")
        pos = nxt

    # Terminator
    w.write_control(False)
    w.flush_if_ready()
    w.write_control(True)
    w.write_data(0)
    w.write_data(0)
    return w.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="formats.prs", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    enc = sub.add_parser("encode", help="PRS-compress a file")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--optimal", action="store_true", help="use slower shortest-path encoder")
    dec = sub.add_parser("decode", help="PRS-decompress a file")
    dec.add_argument("input")
    dec.add_argument("output")
    args = parser.parse_args(argv)

    inp = open(args.input, "rb").read()
    if args.cmd == "encode":
        out = compress_optimal(inp) if args.optimal else compress(inp)
    elif args.cmd == "decode":
        out = decompress(inp)
    else:
        return 2
    with open(args.output, "wb") as fh:
        fh.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
