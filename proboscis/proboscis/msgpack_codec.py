# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Minimal MessagePack encoder/decoder for AMDGPU metadata notes.

Ported from omniprobe tools/codeobj/msgpack_codec.py.
No external dependencies — handles the subset of MessagePack used
by AMDGPU code object metadata.
"""

from __future__ import annotations

import struct


class Unpacker:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def remaining(self) -> int:
        return len(self.payload) - self.offset

    def read(self, size: int) -> bytes:
        if self.offset + size > len(self.payload):
            raise ValueError("unexpected end of MessagePack payload")
        chunk = self.payload[self.offset: self.offset + size]
        self.offset += size
        return chunk

    def unpack(self) -> object:
        prefix = self.read(1)[0]

        if prefix <= 0x7F:
            return prefix
        if prefix >= 0xE0:
            return prefix - 0x100
        if 0xA0 <= prefix <= 0xBF:
            return self.read(prefix & 0x1F).decode("utf-8")
        if 0x90 <= prefix <= 0x9F:
            return [self.unpack() for _ in range(prefix & 0x0F)]
        if 0x80 <= prefix <= 0x8F:
            return {self.unpack(): self.unpack() for _ in range(prefix & 0x0F)}

        if prefix == 0xC0:
            return None
        if prefix == 0xC2:
            return False
        if prefix == 0xC3:
            return True
        if prefix == 0xC4:
            return self.read(int.from_bytes(self.read(1), "big"))
        if prefix == 0xC5:
            return self.read(int.from_bytes(self.read(2), "big"))
        if prefix == 0xC6:
            return self.read(int.from_bytes(self.read(4), "big"))
        if prefix == 0xCA:
            return struct.unpack(">f", self.read(4))[0]
        if prefix == 0xCB:
            return struct.unpack(">d", self.read(8))[0]
        if prefix == 0xCC:
            return int.from_bytes(self.read(1), "big")
        if prefix == 0xCD:
            return int.from_bytes(self.read(2), "big")
        if prefix == 0xCE:
            return int.from_bytes(self.read(4), "big")
        if prefix == 0xCF:
            return int.from_bytes(self.read(8), "big")
        if prefix == 0xD0:
            return int.from_bytes(self.read(1), "big", signed=True)
        if prefix == 0xD1:
            return int.from_bytes(self.read(2), "big", signed=True)
        if prefix == 0xD2:
            return int.from_bytes(self.read(4), "big", signed=True)
        if prefix == 0xD3:
            return int.from_bytes(self.read(8), "big", signed=True)
        if prefix == 0xD9:
            return self.read(int.from_bytes(self.read(1), "big")).decode("utf-8")
        if prefix == 0xDA:
            return self.read(int.from_bytes(self.read(2), "big")).decode("utf-8")
        if prefix == 0xDB:
            return self.read(int.from_bytes(self.read(4), "big")).decode("utf-8")
        if prefix == 0xDC:
            return [self.unpack() for _ in range(int.from_bytes(self.read(2), "big"))]
        if prefix == 0xDD:
            return [self.unpack() for _ in range(int.from_bytes(self.read(4), "big"))]
        if prefix == 0xDE:
            return {
                self.unpack(): self.unpack()
                for _ in range(int.from_bytes(self.read(2), "big"))
            }
        if prefix == 0xDF:
            return {
                self.unpack(): self.unpack()
                for _ in range(int.from_bytes(self.read(4), "big"))
            }

        raise ValueError(f"unsupported MessagePack prefix 0x{prefix:02x}")


def unpackb(payload: bytes) -> object:
    unpacker = Unpacker(payload)
    value = unpacker.unpack()
    if unpacker.remaining() != 0:
        raise ValueError("trailing data after MessagePack payload")
    return value


def _pack_int(value: int) -> bytes:
    if 0 <= value <= 0x7F:
        return bytes([value])
    if -32 <= value < 0:
        return struct.pack("b", value)
    if value >= 0:
        if value <= 0xFF:
            return b"\xCC" + value.to_bytes(1, "big")
        if value <= 0xFFFF:
            return b"\xCD" + value.to_bytes(2, "big")
        if value <= 0xFFFFFFFF:
            return b"\xCE" + value.to_bytes(4, "big")
        if value <= 0xFFFFFFFFFFFFFFFF:
            return b"\xCF" + value.to_bytes(8, "big")
    else:
        if -0x80 <= value < 0:
            return b"\xD0" + value.to_bytes(1, "big", signed=True)
        if -0x8000 <= value < 0:
            return b"\xD1" + value.to_bytes(2, "big", signed=True)
        if -0x80000000 <= value < 0:
            return b"\xD2" + value.to_bytes(4, "big", signed=True)
        if -0x8000000000000000 <= value < 0:
            return b"\xD3" + value.to_bytes(8, "big", signed=True)
    raise ValueError(f"integer out of MessagePack range: {value}")


def _pack_str(value: str) -> bytes:
    payload = value.encode("utf-8")
    size = len(payload)
    if size <= 31:
        return bytes([0xA0 | size]) + payload
    if size <= 0xFF:
        return b"\xD9" + size.to_bytes(1, "big") + payload
    if size <= 0xFFFF:
        return b"\xDA" + size.to_bytes(2, "big") + payload
    return b"\xDB" + size.to_bytes(4, "big") + payload


def _pack_bin(value: bytes) -> bytes:
    size = len(value)
    if size <= 0xFF:
        return b"\xC4" + size.to_bytes(1, "big") + value
    if size <= 0xFFFF:
        return b"\xC5" + size.to_bytes(2, "big") + value
    return b"\xC6" + size.to_bytes(4, "big") + value


def _pack_array(value: list | tuple) -> bytes:
    payload = b"".join(packb(item) for item in value)
    size = len(value)
    if size <= 15:
        return bytes([0x90 | size]) + payload
    if size <= 0xFFFF:
        return b"\xDC" + size.to_bytes(2, "big") + payload
    return b"\xDD" + size.to_bytes(4, "big") + payload


def _pack_map(value: dict) -> bytes:
    payload = bytearray()
    for key, item in value.items():
        payload.extend(packb(key))
        payload.extend(packb(item))
    size = len(value)
    if size <= 15:
        return bytes([0x80 | size]) + bytes(payload)
    if size <= 0xFFFF:
        return b"\xDE" + size.to_bytes(2, "big") + bytes(payload)
    return b"\xDF" + size.to_bytes(4, "big") + bytes(payload)


def packb(value: object) -> bytes:
    if value is None:
        return b"\xC0"
    if isinstance(value, bool):
        return b"\xC3" if value else b"\xC2"
    if isinstance(value, int):
        return _pack_int(value)
    if isinstance(value, float):
        return b"\xCB" + struct.pack(">d", value)
    if isinstance(value, str):
        return _pack_str(value)
    if isinstance(value, bytes):
        return _pack_bin(value)
    if isinstance(value, (list, tuple)):
        return _pack_array(value)
    if isinstance(value, dict):
        return _pack_map(value)
    raise TypeError(f"unsupported MessagePack value type: {type(value).__name__}")
