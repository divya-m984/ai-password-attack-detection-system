"""A deterministic ``.npz`` container: same arrays in, same bytes out.

``numpy.savez`` is not used for publication.  It writes a ZIP whose members
carry the wall-clock time they were written, in whatever order the mapping
happened to iterate, with whatever compression and platform markers the
interpreter defaulted to.  Two runs producing the same model would produce
different files, and a checksum over those files would say the models differ.

So the archive is written here, by hand, with every varying field pinned:

============================  =========================================
Member order                  sorted by array name
Member timestamp              1980-01-01 00:00:00, the ZIP epoch
Compression                   deflate, fixed level
Platform marker               ``create_system = 0``
Permissions                   ``0o644``
``.npy`` format version       1.0
dtype                         declared per array, little-endian
Memory order                  C-contiguous
============================  =========================================

Nothing about the machine reaches the bytes: no path, no host name, no user
name, no temporary directory, no modification time.

**Deterministic bytes are a property, not an identity.**  A model is identified
by its semantic content fingerprint -- the numbers, the feature order, the
hyperparameters -- and two byte-identical archives are a consequence of that,
not the definition of it.  The distinction matters because a future compression
change would alter every byte while changing no model.

**Reading is hostile-input handling.**  An archive is untrusted data: it may
have been edited, replaced, or authored elsewhere.  ``allow_pickle`` is off, and
member names, sizes, counts, dtypes, and finiteness are all checked *before* any
buffer is interpreted as an array.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from password_attack_detector.exceptions import ModelSerializationError

__all__ = [
    "MAX_ARRAY_BYTES",
    "MAX_MEMBER_COUNT",
    "MAX_TOTAL_BYTES",
    "NPZ_ARRAY_NAME_PATTERN",
    "PERMITTED_DTYPES",
    "ZIP_EPOCH",
    "array_digest",
    "normalize_array",
    "read_npz_bytes",
    "write_npz_bytes",
    "write_npz_file",
]

#: The ZIP format's own epoch.  Not "now", and not the file's mtime.
ZIP_EPOCH: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)

#: Fixed deflate level.  Any level produces a valid archive; a *fixed* one
#: produces the same archive twice.
_COMPRESS_LEVEL: Final[int] = 6

#: ``rw-r--r--``, stated rather than inherited from the process umask.
_EXTERNAL_ATTR: Final[int] = 0o644 << 16

#: MS-DOS/FAT.  Recording the real platform would make a model published on
#: Linux differ byte-for-byte from the same model published elsewhere.
_CREATE_SYSTEM: Final[int] = 0

#: Array names permitted in an archive: lower snake case, nothing path-like.
NPZ_ARRAY_NAME_PATTERN: Final[str] = "lower snake case, 1-64 characters"

#: dtypes an authoritative array may carry.  Object arrays are absent on
#: purpose: an object array is a pickle in disguise, whatever the flag says.
PERMITTED_DTYPES: Final[frozenset[str]] = frozenset(
    {"float64", "float32", "int64", "int32", "int8", "uint8", "uint32", "bool"}
)

#: Safety ceilings applied before any allocation. A declared shape is attacker
#: input until it has been checked against these.
MAX_MEMBER_COUNT: Final[int] = 4096
MAX_ARRAY_BYTES: Final[int] = 512 * 1024 * 1024
MAX_TOTAL_BYTES: Final[int] = 1024 * 1024 * 1024

_NAME_MAX_LENGTH: Final[int] = 64


def _validate_name(name: str) -> str:
    """Return *name* if it is a safe array name, else raise.

    Names become ZIP member names, so anything path-like is refused: a member
    called ``../../etc/passwd`` is the oldest archive attack there is, and the
    cheapest place to stop it is before it is ever written.
    """
    if not name or len(name) > _NAME_MAX_LENGTH:
        raise ModelSerializationError(
            f"array name must be 1-{_NAME_MAX_LENGTH} characters; got {len(name)}"
        )
    if not all(
        character.islower() or character.isdigit() or character == "_"
        for character in name
    ):
        raise ModelSerializationError(
            f"array name {name!r} must be lower snake case; a name carrying a "
            f"path separator, a drive letter, or a traversal component is "
            f"refused rather than sanitised"
        )
    if name.startswith("_") or name.endswith("_"):
        raise ModelSerializationError(
            f"array name {name!r} must not begin or end with an underscore"
        )
    return name


def normalize_array(name: str, array: Any) -> np.ndarray[Any, Any]:
    """Return *array* in the one representation this container publishes.

    Little-endian, C-contiguous, of a declared dtype, finite everywhere.  The
    normalisation is deliberate rather than incidental: the same numbers held
    big-endian, Fortran-ordered, or as a view into a larger buffer would
    otherwise serialise to different bytes and read as a different model.

    Raises:
        ModelSerializationError: on an object dtype, an undeclared dtype, a
            non-finite value, or an array larger than the configured ceiling.
    """
    _validate_name(name)
    if not isinstance(array, np.ndarray):
        raise ModelSerializationError(
            f"array {name!r} must be a numpy array, got {type(array).__name__}"
        )
    if array.dtype.hasobject or array.dtype.kind in {"O", "V", "U", "S"}:
        raise ModelSerializationError(
            f"array {name!r} has dtype {array.dtype!s}; an object, void, or "
            f"string array is refused because deserialising one would mean "
            f"interpreting arbitrary payload as Python values"
        )
    dtype_name = str(array.dtype.newbyteorder("="))
    if dtype_name not in PERMITTED_DTYPES:
        raise ModelSerializationError(
            f"array {name!r} has dtype {dtype_name!r}, which is not one of the "
            f"declared model dtypes {sorted(PERMITTED_DTYPES)}"
        )
    if array.nbytes > MAX_ARRAY_BYTES:
        raise ModelSerializationError(
            f"array {name!r} declares {array.nbytes} bytes, above the "
            f"{MAX_ARRAY_BYTES}-byte ceiling"
        )
    normalized = np.ascontiguousarray(
        array, dtype=np.dtype(dtype_name).newbyteorder("<")
    )
    if normalized.dtype.kind == "f" and not np.isfinite(normalized).all():
        raise ModelSerializationError(
            f"array {name!r} carries a NaN or an infinity; a fitted parameter "
            f"must be a number, and NaN would break every exact comparison the "
            f"round-trip guarantee rests on"
        )
    return normalized


def _npy_bytes(array: np.ndarray[Any, Any]) -> bytes:
    """Return the ``.npy`` v1.0 encoding of *array*, with pickling refused."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def write_npz_bytes(arrays: Mapping[str, Any]) -> bytes:
    """Return the deterministic archive encoding *arrays*.

    Members are emitted in sorted name order, so the mapping's insertion order
    -- which is whatever the caller's dict happened to be built in -- cannot
    reach the bytes.

    Raises:
        ModelSerializationError: on an unsafe name, a rejected dtype, a
            non-finite value, or a payload above the configured ceilings.
    """
    if len(arrays) > MAX_MEMBER_COUNT:
        raise ModelSerializationError(
            f"archive declares {len(arrays)} members, above the "
            f"{MAX_MEMBER_COUNT}-member ceiling"
        )
    normalized = {
        name: normalize_array(name, array) for name, array in sorted(arrays.items())
    }
    total = sum(item.nbytes for item in normalized.values())
    if total > MAX_TOTAL_BYTES:
        raise ModelSerializationError(
            f"archive declares {total} bytes, above the {MAX_TOTAL_BYTES}-byte ceiling"
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=_COMPRESS_LEVEL,
    ) as archive:
        for name in sorted(normalized):
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _EXTERNAL_ATTR
            info.create_system = _CREATE_SYSTEM
            archive.writestr(info, _npy_bytes(normalized[name]))
    return buffer.getvalue()


def write_npz_file(path: Path, arrays: Mapping[str, Any]) -> str:
    """Write the deterministic archive to *path* and return its SHA-256.

    The digest is over the bytes, which is what a manifest records for integrity.
    Model *identity* is a separate, semantic thing -- see
    :mod:`password_attack_detector.ml.serialization`.
    """
    payload = write_npz_bytes(arrays)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_npz_bytes(payload: bytes) -> dict[str, np.ndarray[Any, Any]]:
    """Return the arrays *payload* encodes, treating it as untrusted input.

    Every structural property is checked before a single buffer is handed to
    numpy: the member count, each member's name and declared size, the total
    size, and then -- once the array exists -- its dtype and finiteness.
    ``allow_pickle`` is off throughout, so a member carrying a pickle fails
    rather than executing.

    Raises:
        ModelSerializationError: on a malformed archive, an unexpected member
            name, a pickled or object member, an undeclared dtype, a non-finite
            value, or a declaration above the configured ceilings.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ModelSerializationError(
            f"model array archive is not a readable container ({type(exc).__name__})"
        ) from None

    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBER_COUNT:
            raise ModelSerializationError(
                f"archive declares {len(infos)} members, above the "
                f"{MAX_MEMBER_COUNT}-member ceiling"
            )
        declared = sum(info.file_size for info in infos)
        if declared > MAX_TOTAL_BYTES:
            raise ModelSerializationError(
                f"archive declares {declared} uncompressed bytes, above the "
                f"{MAX_TOTAL_BYTES}-byte ceiling"
            )

        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for info in infos:
            if info.is_dir():
                raise ModelSerializationError("archive carries a directory member")
            if not info.filename.endswith(".npy"):
                raise ModelSerializationError(
                    f"archive carries a member that is not a numpy array "
                    f"({info.filename[-16:]!r})"
                )
            name = info.filename[: -len(".npy")]
            try:
                _validate_name(name)
            except ModelSerializationError:
                raise ModelSerializationError(
                    "archive carries a member whose name is not a safe array "
                    "name; a path-shaped member name is refused outright"
                ) from None
            if name in arrays:
                raise ModelSerializationError(
                    f"archive declares array {name!r} more than once"
                )
            if info.file_size > MAX_ARRAY_BYTES:
                raise ModelSerializationError(
                    f"array {name!r} declares {info.file_size} bytes, above the "
                    f"{MAX_ARRAY_BYTES}-byte ceiling"
                )
            with archive.open(info) as member:
                raw = member.read()
            try:
                array = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            except Exception as exc:
                raise ModelSerializationError(
                    f"array {name!r} is not a readable numpy array "
                    f"({type(exc).__name__}); a member requiring pickle support "
                    f"is refused rather than loaded"
                ) from None
            arrays[name] = normalize_array(name, array)
    return arrays


def array_digest(arrays: Mapping[str, Any]) -> str:
    """Return a semantic digest over *arrays*: names, dtypes, shapes, and values.

    Independent of the archive encoding, so a change to the compression level
    would leave this unchanged -- which is the point. Identity is the numbers.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = normalize_array(name, arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
