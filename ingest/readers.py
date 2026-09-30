"""
File parsing for tabular data uploads -- moved out of dashboard.py (Step
3b, 2026-09-30) so both the Streamlit dashboard and the new multipart
upload endpoints (main.py) share ONE implementation instead of two
copies drifting apart. Behavior is unchanged from dashboard.py's
original `read_uploaded_file`/`_read_csv_with_encoding_fallback` -- only
the interface changed, from a Streamlit UploadedFile object (which has
.name/.getvalue()/.seek()) to plain (filename, raw_bytes), so this module
has no Streamlit dependency and works equally from an HTTP multipart
upload. Error reporting changed from `st.error(...)` + raise to just
raise (ValueError / ImportError with the same messages) -- the caller
(dashboard.py or a FastAPI handler) decides how to surface it.
"""

import csv
import gzip
import importlib.util
import io
import zipfile

import pandas as pd

ENCODINGS = ["utf-8-sig", "cp1252", "latin1", "iso-8859-1"]


def is_module_available(module_name: str) -> bool:
    """Checks if a module can be imported without actually importing it --
    used to give a clear message instead of a raw ImportError when an
    optional dependency (openpyxl, xlrd, pyarrow) is missing."""
    return importlib.util.find_spec(module_name) is not None


def read_csv_with_encoding_fallback(raw_bytes: bytes, **kwargs) -> pd.DataFrame:
    """Tries multiple encodings until one works."""
    last_error = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    raise ValueError(f"Could not decode file with any common encoding. Last error: {last_error}")


def read_uploaded_file(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    """
    Parses a tabular file's raw bytes into a DataFrame, dispatching on
    filename extension. Supports: CSV, TSV, Excel (.xlsx, .xls), JSON,
    JSON Lines, Parquet, ARFF, libsvm .dat, and gzip/zip compressed CSVs.
    Auto-detects encoding for text files.
    """
    filename = filename.lower()

    # --------------------------------------------------------
    # 1. ARFF (Weka) -- handles % comments and @ATTRIBUTE headers
    # --------------------------------------------------------
    if filename.endswith(".arff"):
        content = None
        for enc in ["utf-8", "cp1252", "latin1", "iso-8859-1"]:
            try:
                content = raw_bytes.decode(enc).splitlines()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            raise ValueError("Could not decode .arff file with any common encoding.")

        column_names = []
        data_rows = []
        in_data = False

        for line in content:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper().startswith("@RELATION"):
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                parts = line.split()
                if len(parts) >= 3:
                    attr = parts[1].strip("'\"")
                    column_names.append(attr)
                continue
            if line.upper().startswith("@DATA"):
                in_data = True
                continue
            if in_data and line:
                reader = csv.reader([line], skipinitialspace=True)
                row = next(reader)
                data_rows.append([x.strip() for x in row])

        if not column_names:
            return read_csv_with_encoding_fallback(raw_bytes, sep=None, engine="python")

        df = pd.DataFrame(data_rows, columns=column_names)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # leave as text -- intended "ignore" behaviour
        return df

    # --------------------------------------------------------
    # 2. DAT (libsvm style, with proper fallback to generic text)
    # --------------------------------------------------------
    if filename.endswith(".dat"):
        content = None
        for enc in ["utf-8", "cp1252", "latin1", "iso-8859-1"]:
            try:
                content = raw_bytes.decode(enc).splitlines()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            raise ValueError("Could not decode .dat file with any common encoding.")

        parsed_rows = []
        for line in content:
            parts = line.strip().split()
            if not parts:
                continue
            row = {}
            is_libsvm = False
            for token in parts[1:]:
                if ":" in token:
                    is_libsvm = True
                    idx, val = token.split(":")
                    row[f"Sensor_{idx}"] = float(val)
            if is_libsvm:
                parsed_rows.append(row)

        if parsed_rows:
            return pd.DataFrame(parsed_rows)

        try:
            return read_csv_with_encoding_fallback(raw_bytes, sep=r"\s+", engine="python")
        except Exception:
            return read_csv_with_encoding_fallback(raw_bytes, sep=None, engine="python")

    # --------------------------------------------------------
    # 3. Excel (.xlsx, .xls) -- reads all sheets, concatenates them
    # --------------------------------------------------------
    if filename.endswith((".xlsx", ".xls")):
        engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
        required_module = engine

        if not is_module_available(required_module):
            raise ImportError(f"Please install {required_module} for Excel support: pip install {required_module}")

        all_sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, engine=engine)
        if len(all_sheets) == 1:
            return next(iter(all_sheets.values()))
        return pd.concat(all_sheets.values(), ignore_index=True)

    # --------------------------------------------------------
    # 4. JSON and JSON Lines (.json, .jsonl, .ndjson)
    # --------------------------------------------------------
    if filename.endswith((".json", ".jsonl", ".ndjson")):
        try:
            return pd.read_json(io.BytesIO(raw_bytes))
        except ValueError:
            return pd.read_json(io.BytesIO(raw_bytes), lines=True)

    # --------------------------------------------------------
    # 5. Parquet
    # --------------------------------------------------------
    if filename.endswith(".parquet"):
        if not is_module_available("pyarrow"):
            raise ImportError("Please install pyarrow for Parquet support: pip install pyarrow")
        return pd.read_parquet(io.BytesIO(raw_bytes))

    # --------------------------------------------------------
    # 6. TSV (tab-separated)
    # --------------------------------------------------------
    if filename.endswith(".tsv"):
        return read_csv_with_encoding_fallback(raw_bytes, sep="\t", engine="python")

    # --------------------------------------------------------
    # 7. Compressed CSV (.csv.gz, .gz)
    # --------------------------------------------------------
    if filename.endswith(".gz"):
        decompressed = gzip.decompress(raw_bytes)
        return read_csv_with_encoding_fallback(decompressed, sep=None, engine="python")

    # --------------------------------------------------------
    # 8. Zipped CSV (.zip containing a single CSV/TXT file)
    # --------------------------------------------------------
    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            inner_names = [n for n in z.namelist() if not n.endswith("/")]
            if not inner_names:
                raise ValueError("Zip file contains no readable files.")
            with z.open(inner_names[0]) as f:
                inner_bytes = f.read()
            return read_csv_with_encoding_fallback(inner_bytes, sep=None, engine="python")

    # --------------------------------------------------------
    # 9. CSV / TXT / other text -- auto-detect delimiter and encoding
    # --------------------------------------------------------
    return read_csv_with_encoding_fallback(raw_bytes, sep=None, engine="python")
