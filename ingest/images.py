"""
Safe image ingestion for the unified table path: an image column holds a
filename (resolved against an uploaded ZIP) or, in JSON tables, a base64 /
data-URI value.

Security (docs/unified_table_plan.md, section M):
- never extracts to disk; entries are read in memory, one at a time;
- rejects absolute paths, drive letters, '..' segments, NUL bytes,
  symlinks and encrypted entries (each counted in `rejected`);
- caps entry count, per-entry and total uncompressed size, and the
  compression ratio (archive bombs) -- the per-entry cap is enforced while
  reading, not by trusting the header's declared size;
- decodes with a pixel cap and treats PIL's decompression-bomb warning as
  an error.
Every row gets a status -- ok | missing | ambiguous | corrupt | too_large |
unsupported -- so no invalid image is ever dropped silently.
"""

import base64
import binascii
import io
import os
import posixpath
import re
import warnings
import zipfile
import zlib
from typing import Dict, List, Optional, Tuple

from PIL import Image

MAX_ZIP_ENTRIES = int(os.getenv("DRIFT_MAX_ZIP_ENTRIES", "20000"))
MAX_ENTRY_BYTES = int(os.getenv("DRIFT_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
MAX_TOTAL_UNCOMPRESSED = int(os.getenv("DRIFT_MAX_ZIP_UNCOMPRESSED", str(2 * 1024 ** 3)))
MAX_COMPRESSION_RATIO = 100
MAX_IMAGE_PIXELS = int(os.getenv("DRIFT_MAX_IMAGE_PIXELS", "50000000"))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff")
ALLOWED_FORMATS = {"JPEG", "PNG", "GIF", "BMP", "WEBP", "TIFF", "MPO"}  # MPO = multi-picture JPEG
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")


class ArchiveError(ValueError):
    """The ZIP as a whole is unusable (not a zip, too many entries, too large)."""


def normalize_name(name: str) -> Optional[str]:
    """Normalized relative path, or None if the name is unsafe."""
    if "\x00" in name:
        return None
    name = name.replace("\\", "/")
    if name.startswith("/") or _DRIVE_RE.match(name):
        return None
    if any(part == ".." for part in name.split("/")):
        return None
    norm = posixpath.normpath(name)
    return None if norm in (".", "") else norm


class ImageArchive:
    def __init__(self, zip_path: str):
        try:
            self._zip = zipfile.ZipFile(zip_path)
        except (zipfile.BadZipFile, OSError) as e:
            raise ArchiveError(f"Image ZIP could not be opened: {e}")
        infos = self._zip.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise ArchiveError(f"Image ZIP has {len(infos)} entries, over the {MAX_ZIP_ENTRIES} limit.")
        self.by_path: Dict[str, zipfile.ZipInfo] = {}
        self.by_basename: Dict[str, List[str]] = {}
        self.rejected: Dict[str, int] = {}
        total = 0
        for info in infos:
            if info.is_dir():
                continue
            reason = self._reject_reason(info)
            if reason:
                self.rejected[reason] = self.rejected.get(reason, 0) + 1
                continue
            norm = normalize_name(info.filename)
            total += info.file_size
            if total > MAX_TOTAL_UNCOMPRESSED:
                raise ArchiveError("Image ZIP's total uncompressed size exceeds the limit.")
            self.by_path[norm] = info
            self.by_basename.setdefault(posixpath.basename(norm), []).append(norm)

    @staticmethod
    def _reject_reason(info: zipfile.ZipInfo) -> Optional[str]:
        norm = normalize_name(info.filename)
        if norm is None:
            return "unsafe_path"
        base = posixpath.basename(norm)
        if norm.startswith("__MACOSX/") or base.startswith("."):
            return "os_metadata"
        if info.flag_bits & 0x1:
            return "encrypted"
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            return "symlink"
        if not base.lower().endswith(IMAGE_EXTENSIONS):
            return "not_an_image_file"
        if info.file_size > MAX_ENTRY_BYTES:
            return "too_large"
        if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            return "suspicious_compression_ratio"
        return None

    def resolve(self, value: str) -> Tuple[str, Optional[str]]:
        """(status, entry path): exact normalized path first, then a unique basename."""
        norm = normalize_name(str(value).strip())
        if norm is None:
            return "missing", None
        if norm in self.by_path:
            return "ok", norm
        candidates = self.by_basename.get(posixpath.basename(norm), [])
        if len(candidates) == 1:
            return "ok", candidates[0]
        return ("ambiguous", None) if len(candidates) > 1 else ("missing", None)

    def read(self, entry: str) -> bytes:
        """Reads one entry, enforcing the size cap while reading."""
        out = io.BytesIO()
        with self._zip.open(self.by_path[entry]) as f:
            while True:
                chunk = f.read(256 * 1024)
                if not chunk:
                    break
                if out.tell() + len(chunk) > MAX_ENTRY_BYTES:
                    raise ValueError("too_large")
                out.write(chunk)
        return out.getvalue()

    def close(self):
        self._zip.close()


def decode_inline(value: str) -> Optional[bytes]:
    """Bytes for a data-URI or bare base64 image value, else None."""
    s = str(value).strip()
    if s.startswith("data:image/"):
        _, _, s = s.partition(",")
    elif len(s) < 64:
        return None
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if raw.startswith(_MAGIC) and len(raw) <= MAX_ENTRY_BYTES else None


def validate_image_bytes(raw: bytes) -> str:
    """'ok', 'corrupt', 'too_large' or 'unsupported' -- decodes fully so a
    truncated file is caught here, not mid-embedding."""
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as img:
                if img.format not in ALLOWED_FORMATS:
                    return "unsupported"
                img.load()
        return "ok"
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return "too_large"
    except Exception:
        return "corrupt"
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def load_image(value, archive: Optional[ImageArchive]) -> Tuple[str, Optional[bytes]]:
    """(status, bytes) for one table cell of an image column."""
    if value is None or (isinstance(value, float) and value != value) or str(value).strip() == "":
        return "missing", None
    raw = decode_inline(value)
    if raw is None:
        if archive is None:
            return "missing", None
        status, entry = archive.resolve(value)
        if status != "ok":
            return status, None
        try:
            raw = archive.read(entry)
        except ValueError:
            return "too_large", None
        except (zipfile.BadZipFile, zlib.error, EOFError, OSError):
            return "corrupt", None  # damaged compressed data / CRC mismatch inside the ZIP
    status = validate_image_bytes(raw)
    return status, (raw if status == "ok" else None)
