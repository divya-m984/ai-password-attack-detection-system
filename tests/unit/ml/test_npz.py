"""The deterministic archive: same arrays in, same bytes out, hostile input refused.

Two properties, and every test is one of them. **Bytes are a function of the
arrays alone** -- not of the clock, the directory, or the order a dict happened
to be built in. And **reading is hostile-input handling** -- a member name, a
dtype, a size, and a pickle flag are all attacker-controlled until checked.
"""

from __future__ import annotations

import hashlib
import io
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from password_attack_detector.exceptions import ModelSerializationError
from password_attack_detector.ml.npz import (
    MAX_MEMBER_COUNT,
    PERMITTED_DTYPES,
    ZIP_EPOCH,
    array_digest,
    normalize_array,
    read_npz_bytes,
    write_npz_bytes,
    write_npz_file,
)


def sample() -> dict[str, np.ndarray]:
    """Return a small mapping spanning several permitted dtypes."""
    return {
        "coefficients": np.array([[1.5, -2.25, 0.0]], dtype=np.float64),
        "intercept": np.array([0.125], dtype=np.float64),
        "children_left": np.array([1, -1, -1], dtype=np.int64),
        "is_leaf": np.array([0, 1, 1], dtype=np.uint8),
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_insertion_order_does_not_reach_the_bytes() -> None:
    """Members are written sorted, so a caller's dict order is not recorded."""
    forward = sample()
    reversed_order = dict(reversed(list(forward.items())))
    assert list(forward) != list(reversed_order)
    assert write_npz_bytes(forward) == write_npz_bytes(reversed_order)


def test_the_bytes_do_not_change_with_the_clock() -> None:
    """Two writes a measurable interval apart produce identical archives.

    The ZIP format records a modification time per member. Left to itself that
    is the wall clock, which would make every republication of an unchanged
    model produce a different checksum.
    """
    first = write_npz_bytes(sample())
    time.sleep(1.1)
    assert write_npz_bytes(sample()) == first


def test_the_bytes_do_not_change_with_the_directory(tmp_path: Path) -> None:
    """The archive records nothing about where it was written."""
    left, right = tmp_path / "a" / "m", tmp_path / "b" / "m"
    for directory in (left, right):
        directory.mkdir(parents=True)
    first = write_npz_file(left / "arrays.npz", sample())
    second = write_npz_file(right / "arrays.npz", sample())
    assert first == second
    assert (left / "arrays.npz").read_bytes() == (right / "arrays.npz").read_bytes()


def test_the_digest_is_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    """What ``write_npz_file`` returns is what a manifest should record."""
    path = tmp_path / "arrays.npz"
    digest = write_npz_file(path, sample())
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_member_carries_the_zip_epoch() -> None:
    """Asserted on the archive itself, not inferred from two writes agreeing."""
    archive = zipfile.ZipFile(io.BytesIO(write_npz_bytes(sample())))
    for info in archive.infolist():
        assert info.date_time == ZIP_EPOCH


def test_every_member_carries_fixed_platform_metadata() -> None:
    """Permissions and platform marker are stated rather than inherited."""
    archive = zipfile.ZipFile(io.BytesIO(write_npz_bytes(sample())))
    for info in archive.infolist():
        assert info.create_system == 0
        assert info.external_attr == 0o644 << 16
        assert info.compress_type == zipfile.ZIP_DEFLATED


def test_members_appear_in_sorted_name_order() -> None:
    """The order is a property of the names, not of the caller."""
    archive = zipfile.ZipFile(io.BytesIO(write_npz_bytes(sample())))
    names = [info.filename for info in archive.infolist()]
    assert names == sorted(names)


def test_the_semantic_digest_ignores_the_encoding() -> None:
    """Identity is the numbers; the archive is one way of writing them down."""
    assert array_digest(sample()) == array_digest(
        dict(reversed(list(sample().items())))
    )


def test_the_semantic_digest_changes_with_the_numbers() -> None:
    """And it is not merely a constant that always agrees."""
    changed = sample()
    changed["intercept"] = np.array([0.126], dtype=np.float64)
    assert array_digest(changed) != array_digest(sample())


def test_the_semantic_digest_changes_with_the_shape() -> None:
    """The same values in a different shape are a different array."""
    flat = {"coefficients": np.array([1.0, 2.0, 3.0, 4.0])}
    square = {"coefficients": np.array([[1.0, 2.0], [3.0, 4.0]])}
    assert array_digest(flat) != array_digest(square)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_byte_order_is_normalised_to_little_endian() -> None:
    """The same numbers stored big-endian encode to the same archive."""
    little = {"values": np.array([1.5, 2.5], dtype="<f8")}
    big = {"values": np.array([1.5, 2.5], dtype=">f8")}
    assert write_npz_bytes(little) == write_npz_bytes(big)


def test_memory_order_is_normalised_to_c_contiguous() -> None:
    """A Fortran-ordered array is the same model, so it is the same bytes."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert write_npz_bytes({"values": values}) == write_npz_bytes(
        {"values": np.asfortranarray(values)}
    )


def test_a_non_contiguous_view_is_normalised() -> None:
    """A slice of a larger buffer must not encode differently from a copy."""
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]
    assert write_npz_bytes({"values": view}) == write_npz_bytes(
        {"values": np.ascontiguousarray(view)}
    )


@pytest.mark.parametrize("dtype", sorted(PERMITTED_DTYPES))
def test_every_permitted_dtype_round_trips(dtype: str) -> None:
    """Each declared dtype survives a write and a read unchanged."""
    array = np.ones((2, 3), dtype=np.dtype(dtype))
    restored = read_npz_bytes(write_npz_bytes({"values": array}))
    assert str(restored["values"].dtype) == dtype
    assert np.array_equal(restored["values"], array)


def test_the_round_trip_preserves_values_exactly() -> None:
    """Not approximately: a fitted coefficient is not a rounded coefficient."""
    restored = read_npz_bytes(write_npz_bytes(sample()))
    for name, array in sample().items():
        assert np.array_equal(restored[name], array)


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def test_an_object_array_is_refused() -> None:
    """An object array is a pickle in disguise, whatever the flag says."""
    with pytest.raises(ModelSerializationError, match="object"):
        write_npz_bytes({"values": np.array([1, "two"], dtype=object)})


@pytest.mark.parametrize("dtype", ["U8", "S8", "complex128", "float16"])
def test_an_undeclared_dtype_is_refused(dtype: str) -> None:
    """The permitted set is a closed list, not a starting point."""
    with pytest.raises(ModelSerializationError):
        write_npz_bytes({"values": np.zeros(2, dtype=np.dtype(dtype))})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_refused(value: float) -> None:
    """NaN breaks every exact comparison the round-trip guarantee rests on."""
    with pytest.raises(ModelSerializationError, match="NaN or an infinity"):
        write_npz_bytes({"values": np.array([1.0, value])})


@pytest.mark.parametrize(
    "name",
    [
        "../evil",
        "/etc/passwd",
        "a/b",
        "a\\b",
        "C:name",
        "Upper",
        "with space",
        "",
        "_x",
        "x_",
    ],
)
def test_an_unsafe_array_name_is_refused(name: str) -> None:
    """Names become archive member names, so anything path-shaped is refused."""
    with pytest.raises(ModelSerializationError):
        write_npz_bytes({name: np.array([1.0])})


def test_a_name_above_the_length_ceiling_is_refused() -> None:
    """A ceiling exists so a name cannot itself be a payload."""
    with pytest.raises(ModelSerializationError, match="characters"):
        write_npz_bytes({"a" * 65: np.array([1.0])})


def test_too_many_members_are_refused() -> None:
    """A declared member count is attacker input until it has been bounded."""
    arrays = {f"a{index}": np.array([1.0]) for index in range(MAX_MEMBER_COUNT + 1)}
    with pytest.raises(ModelSerializationError, match="ceiling"):
        write_npz_bytes(arrays)


def test_a_non_array_value_is_refused() -> None:
    """A list is not an array, and coercing one would guess at a dtype."""
    with pytest.raises(ModelSerializationError, match="numpy array"):
        write_npz_bytes({"values": [1.0, 2.0]})


# ---------------------------------------------------------------------------
# Reading untrusted archives
# ---------------------------------------------------------------------------


def test_a_non_archive_payload_is_refused() -> None:
    """Bytes that are not a container fail before anything is interpreted."""
    with pytest.raises(ModelSerializationError, match="readable container"):
        read_npz_bytes(b"this is not a zip file")


def test_a_truncated_archive_is_refused() -> None:
    """A partial file is a failure rather than a partial model."""
    payload = write_npz_bytes(sample())
    with pytest.raises(ModelSerializationError):
        read_npz_bytes(payload[: len(payload) // 2])


def test_a_member_that_is_not_a_numpy_array_is_refused() -> None:
    """A model archive holds arrays; anything else is smuggled cargo."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"hello")
    with pytest.raises(ModelSerializationError, match="not a numpy array"):
        read_npz_bytes(buffer.getvalue())


def test_a_path_shaped_member_name_is_refused() -> None:
    """The classic archive attack, refused at read as well as at write."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../escape.npy", b"\x93NUMPY")
    with pytest.raises(ModelSerializationError, match="safe array name"):
        read_npz_bytes(buffer.getvalue())


def test_a_pickled_member_is_refused() -> None:
    """``allow_pickle`` is off, so a payload asking to be executed simply fails."""
    buffer = io.BytesIO()
    payload = io.BytesIO()
    np.lib.format.write_array(
        payload, np.array([{"a": 1}], dtype=object), allow_pickle=True
    )
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("evil.npy", payload.getvalue())
    with pytest.raises(ModelSerializationError, match="pickle"):
        read_npz_bytes(buffer.getvalue())


def test_a_directory_member_is_refused() -> None:
    """A model archive is flat; a nested entry has nowhere to be."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(zipfile.ZipInfo("subdir/"), b"")
    with pytest.raises(ModelSerializationError, match="directory member"):
        read_npz_bytes(buffer.getvalue())


def test_a_member_declaring_an_enormous_size_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declaration is checked before anything is allocated for it.

    The ceiling is lowered rather than a half-gigabyte member being built, which
    would test the machine's memory rather than the check.
    """
    from password_attack_detector.ml import npz as module

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("values.npy", b"x" * 4096)
    monkeypatch.setattr(module, "MAX_ARRAY_BYTES", 16)
    with pytest.raises(ModelSerializationError, match="ceiling"):
        read_npz_bytes(buffer.getvalue())


def test_a_repeated_member_name_is_refused() -> None:
    """Two members with one name leave it ambiguous which array is the model."""
    buffer = io.BytesIO()
    payload = io.BytesIO()
    np.lib.format.write_array(payload, np.array([1.0]), allow_pickle=False)
    with warnings.catch_warnings():
        # zipfile warns about the duplicate; producing one is the whole point.
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("values.npy", payload.getvalue())
            archive.writestr("values.npy", payload.getvalue())
    with pytest.raises(ModelSerializationError, match="more than once"):
        read_npz_bytes(buffer.getvalue())


def test_normalize_array_is_the_single_gate() -> None:
    """Both directions apply the same rule, so a read cannot be laxer than a write."""
    with pytest.raises(ModelSerializationError):
        normalize_array("values", np.array([np.inf]))
    with pytest.raises(ModelSerializationError):
        normalize_array("bad name", np.array([1.0]))
