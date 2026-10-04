// Mirrors ingest/readers.py's read_uploaded_file -- keep this list in
// sync with what the backend actually parses (verified end-to-end via
// tests/test_upload_endpoints.py, not just assumed from the file list).
export const TABULAR_ACCEPT = ".csv,.tsv,.xlsx,.xls,.json,.jsonl,.ndjson,.parquet,.arff,.dat,.gz,.zip";
export const TABULAR_FORMATS_HINT =
  "Whatever you've got works here: CSV, TSV, Excel, JSON or JSON Lines, Parquet, ARFF, a libsvm .dat " +
  "file, or even a gzip/zip-compressed CSV.";
