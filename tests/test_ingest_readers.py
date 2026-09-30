"""
Step 3b: fixture tests for ingest/readers.py, one per supported format,
asserting the parsed DataFrame matches expected values exactly. This
module is dashboard.py's file-parsing logic moved out verbatim (see
ingest/readers.py's docstring) so both the dashboard and the new
multipart upload endpoints share one implementation -- these tests pin
that shared behavior.

Run:
    python -m pytest tests/test_ingest_readers.py -v
"""

import gzip
import io
import json
import zipfile

import pandas as pd
import pytest

from ingest.readers import read_uploaded_file, is_module_available

EXPECTED_ROWS = [
    {"a": 1, "b": "x"},
    {"a": 2, "b": "y"},
    {"a": 3, "b": "z"},
]


def _assert_matches_expected(df: pd.DataFrame):
    assert list(df.columns) == ["a", "b"]
    assert df["a"].astype(int).tolist() == [1, 2, 3]
    assert df["b"].astype(str).tolist() == ["x", "y", "z"]


class TestCSV:
    def test_plain_csv(self):
        raw = b"a,b\n1,x\n2,y\n3,z\n"
        _assert_matches_expected(read_uploaded_file("data.csv", raw))

    def test_csv_with_bom(self):
        raw = "a,b\n1,x\n2,y\n3,z\n".encode("utf-8-sig")
        _assert_matches_expected(read_uploaded_file("data.csv", raw))

    def test_csv_latin1_encoded(self):
        raw = "a,b\n1,x\n2,y\n3,z\n".encode("latin1")
        _assert_matches_expected(read_uploaded_file("data.csv", raw))


class TestTSV:
    def test_tsv(self):
        raw = b"a\tb\n1\tx\n2\ty\n3\tz\n"
        _assert_matches_expected(read_uploaded_file("data.tsv", raw))


class TestJSON:
    def test_json_records(self):
        raw = json.dumps(EXPECTED_ROWS).encode()
        _assert_matches_expected(read_uploaded_file("data.json", raw))

    def test_jsonl(self):
        raw = "\n".join(json.dumps(r) for r in EXPECTED_ROWS).encode()
        _assert_matches_expected(read_uploaded_file("data.jsonl", raw))

    def test_ndjson(self):
        raw = "\n".join(json.dumps(r) for r in EXPECTED_ROWS).encode()
        _assert_matches_expected(read_uploaded_file("data.ndjson", raw))


class TestCompressed:
    def test_gzip_csv(self):
        raw = gzip.compress(b"a,b\n1,x\n2,y\n3,z\n")
        _assert_matches_expected(read_uploaded_file("data.csv.gz", raw))

    def test_zip_csv(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("inner.csv", "a,b\n1,x\n2,y\n3,z\n")
        _assert_matches_expected(read_uploaded_file("data.zip", buf.getvalue()))


class TestARFF:
    def test_arff(self):
        raw = (
            "% comment\n"
            "@RELATION test\n"
            "@ATTRIBUTE a NUMERIC\n"
            "@ATTRIBUTE b STRING\n"
            "@DATA\n"
            "1,x\n2,y\n3,z\n"
        ).encode()
        _assert_matches_expected(read_uploaded_file("data.arff", raw))


class TestDAT:
    def test_libsvm_dat(self):
        raw = b"1 1:0.5 2:1.5\n0 1:0.2 2:2.2\n"
        df = read_uploaded_file("data.dat", raw)
        assert list(df.columns) == ["Sensor_1", "Sensor_2"]
        assert df["Sensor_1"].tolist() == [0.5, 0.2]

    def test_whitespace_dat_falls_back_to_csv(self):
        raw = b"a b\n1 x\n2 y\n3 z\n"
        df = read_uploaded_file("data.dat", raw)
        assert list(df.columns) == ["a", "b"]


class TestParquet:
    @pytest.mark.skipif(not is_module_available("pyarrow"), reason="pyarrow not installed")
    def test_parquet(self):
        df_in = pd.DataFrame(EXPECTED_ROWS)
        buf = io.BytesIO()
        df_in.to_parquet(buf)
        _assert_matches_expected(read_uploaded_file("data.parquet", buf.getvalue()))

    def test_parquet_missing_pyarrow_raises_clean_error(self, monkeypatch):
        import ingest.readers as readers
        monkeypatch.setattr(readers, "is_module_available", lambda name: False)
        with pytest.raises(ImportError, match="pyarrow"):
            read_uploaded_file("data.parquet", b"irrelevant")


class TestUnsupportedFallback:
    def test_unknown_extension_falls_back_to_csv_parsing(self):
        raw = b"a,b\n1,x\n2,y\n3,z\n"
        _assert_matches_expected(read_uploaded_file("data.txt", raw))
