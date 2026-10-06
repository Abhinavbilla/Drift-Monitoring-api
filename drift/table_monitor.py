"""
Unified table path (docs/unified_table_plan.md, milestone M1): the profile,
fit and analyze job handlers.

One table = one project. Numeric/categorical columns go through the
EXISTING tabular code (main._resolve_and_persist_fit / main's tabular
analysis core, passed in as hooks) -- that is what makes a numeric/
categorical-only table behave exactly like the tabular endpoints. Text and
image columns get per-column embedding baselines and the existing Domain
Classifier Test, decided by the legacy AUC rule and kept OUTSIDE the Holm
family until their nulls are calibrated (M3). Relationship tests between
numeric/categorical columns (M2) join the numeric/categorical Holm family.
"""

import io
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import jobs
from adapters.image import ImageAdapter
from adapters.text import TextAdapter
from db import blob_store, crud
from drift import relationship_detector as rel
from drift.calibration import CalibrationConfig
from drift.embedding_detector import EmbeddingDriftDetector, HARD_MIN_SAMPLES
from ingest import readers
from ingest.images import ArchiveError, ImageArchive, load_image
from utils.profiler import profile_table

N_MAX_REFERENCE = 5000      # seeded sample cap for embeddings and the row-aligned store
REFERENCE_SAMPLE_SEED = 42
EMBED_CHUNK = 64
DCT_AUC_THRESHOLD = 0.65
MONITORED_TYPES = ("numeric", "categorical", "text", "image")
_TEST_BY_TYPE = {"numeric": "KS", "categorical": "PSI", "text": "DCT", "image": "DCT"}
_REL_LABEL = {"num_num": "Spearman split-null", "cat_cat": "log-linear G2 split-null",
              "num_cat": "conditional PIT split-null"}


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------
def _open_stage(stage: Optional[dict], allowed_status: Tuple[str, ...]) -> Tuple[pd.DataFrame, Optional[ImageArchive]]:
    if stage is None or stage["status"] not in allowed_status or not stage["table_blob"]:
        raise jobs.JobError("The staged upload is no longer available (expired, deleted or already used). "
                            "Please upload the file again.")
    try:
        df = readers.read_uploaded_file(stage["table_filename"], blob_store.get_bytes(stage["table_blob"]))
    except (ValueError, ImportError) as e:
        raise jobs.JobError(f"Could not parse the uploaded table ({type(e).__name__}).")
    if df.empty:
        raise jobs.JobError("The uploaded table has zero rows.")
    df.columns = [str(c) for c in df.columns]
    archive = None
    if stage["zip_blob"]:
        try:
            archive = ImageArchive(blob_store.path(stage["zip_blob"]))
        except ArchiveError as e:
            raise jobs.JobError(str(e))
    return df, archive


def _embed_text(values: pd.Series) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """(embeddings of valid rows, row indices of valid rows, invalid counts)."""
    is_valid = values.map(lambda v: v is not None and not (isinstance(v, float) and v != v) and str(v).strip() != "")
    idx = np.flatnonzero(is_valid.to_numpy())
    texts = [str(v) for v in values.iloc[idx]]
    adapter = TextAdapter()
    chunks = [adapter.transform(texts[i:i + EMBED_CHUNK]) for i in range(0, len(texts), EMBED_CHUNK)]
    emb = np.vstack(chunks) if chunks else np.empty((0, 384), dtype=np.float32)
    invalid = int((~is_valid).sum())
    return emb, idx, ({"empty_or_null": invalid} if invalid else {})


def _embed_images(values: pd.Series, archive: Optional[ImageArchive]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[int]]:
    """(embeddings of valid rows, their row indices, invalid counts by status,
    first 20 invalid row positions). Images are validated and embedded in
    chunks, never all decoded in memory at once."""
    adapter = ImageAdapter()
    embs, idx, counts, bad_rows = [], [], {}, []
    pending_raw, pending_idx = [], []

    def flush():
        if pending_raw:
            embs.append(adapter.transform(list(pending_raw)))
            idx.extend(pending_idx)
            pending_raw.clear()
            pending_idx.clear()

    for pos, value in enumerate(values.tolist()):
        status, raw = load_image(value, archive)
        if status != "ok":
            counts[status] = counts.get(status, 0) + 1
            if len(bad_rows) < 20:
                bad_rows.append(pos)
            continue
        pending_raw.append(raw)
        pending_idx.append(pos)
        if len(pending_raw) >= EMBED_CHUNK:
            flush()
    flush()
    emb = np.vstack(embs) if embs else np.empty((0, 512), dtype=np.float32)
    return emb, np.asarray(idx, dtype=int), counts, bad_rows


def _embed_column(col_type: str, values: pd.Series, archive) -> Tuple[np.ndarray, Dict[str, Any]]:
    if col_type == "text":
        emb, idx, counts = _embed_text(values)
        bad_rows: List[int] = []
    else:
        emb, idx, counts, bad_rows = _embed_images(values, archive)
    return emb, {"valid": int(len(idx)), "invalid": counts, "invalid_rows_sample": bad_rows}


def _cap_rows(n: int) -> np.ndarray:
    if n <= N_MAX_REFERENCE:
        return np.arange(n)
    rng = np.random.default_rng(REFERENCE_SAMPLE_SEED)
    return np.sort(rng.choice(n, size=N_MAX_REFERENCE, replace=False))


def _client_from_job(job: dict) -> dict:
    return {"email": job["owner_email"], "name": job["owner_name"] or job["owner_email"],
            "internal_project_id": job["project_id"]}


# ---------------------------------------------------------
# Profile job
# ---------------------------------------------------------
def run_profile(job: dict) -> dict:
    stage = crud.get_stage(job["stage_id"])
    df, archive = _open_stage(stage, ("uploaded",))
    try:
        jobs.progress(job["id"], 30, "Profiling columns")
        columns = profile_table(df, archive)
        jobs.progress(job["id"], 70, "Proposing relationships")
        relationships = rel.propose_relationships({
            p["name"]: (p["proposed_type"], rel.clean_column(df[p["name"]].tolist(), p["proposed_type"]))
            for p in columns if p["proposed_type"] in ("numeric", "categorical")})
        zip_info = None if archive is None else {"image_entries": len(archive.by_path), "rejected": archive.rejected}
    finally:
        if archive:
            archive.close()
    result = {"stage_id": stage["id"], "n_rows": len(df), "columns": columns, "relationships": relationships,
              "image_zip": zip_info,
              "expires_at": stage["expires_at"]}
    crud.update_stage(stage["id"], status="profiled", profile=result)
    return result


# ---------------------------------------------------------
# Fit job
# ---------------------------------------------------------
def run_fit(job: dict, persist_fit: Callable[..., Any], split_high_cardinality: Callable[..., Dict[str, str]]) -> dict:
    payload = job["payload"]
    stage = crud.get_stage(job["stage_id"])
    df, archive = _open_stage(stage, ("profiled",))
    try:
        choices = {c["name"]: c for c in payload["columns"]}
        monitored = {n: c["type"] for n, c in choices.items() if c["monitor"] and c["type"] in MONITORED_TYPES}
        by_type = {t: [n for n, ct in monitored.items() if ct == t] for t in MONITORED_TYPES}

        # Only rows identical in EVERY column are duplicates (same rule as the
        # tabular /fit). Deduplicating on the monitored columns alone collapsed
        # distinct records that merely share low-cardinality values.
        rows_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        duplicate_rows_dropped = rows_before - len(df)

        # Embed text/image columns FIRST: a failure here must not leave a
        # half-written baseline version behind.
        embedded: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        for i, col in enumerate(by_type["text"] + by_type["image"]):
            jobs.progress(job["id"], 10 + int(60 * i / max(1, len(by_type["text"]) + len(by_type["image"]))),
                          f"Embedding column '{col}'")
            emb, quality = _embed_column(monitored[col], df[col], archive)
            if quality["valid"] < HARD_MIN_SAMPLES:
                raise jobs.JobError(
                    f"Column '{col}' has only {quality['valid']} usable {monitored[col]} value(s) "
                    f"(need at least {HARD_MIN_SAMPLES}); invalid: {quality['invalid']}. "
                    f"Fix the data or turn monitoring off for this column.")
            embedded[col] = (emb, quality)
    finally:
        if archive:
            archive.close()

    jobs.progress(job["id"], 75, "Fitting numeric/categorical baselines")
    continuous = {c: df[c].tolist() for c in by_type["numeric"]}
    categorical = {c: [str(v) for v in df[c].tolist()] for c in by_type["categorical"]}
    inferred = {**{c: "continuous" for c in continuous}, **{c: "categorical" for c in categorical}}
    excluded = split_high_cardinality(categorical, inferred)
    fit_resp = persist_fit(
        job["public_project_id"], inferred, continuous, categorical, df, payload.get("calibration_config"),
        _client_from_job(job), payload.get("schema_policy"), payload.get("model_version_label"),
        payload.get("alert_policy"), profiler_excluded_columns=excluded,
        duplicate_rows_dropped=duplicate_rows_dropped,
    )
    project, version = job["project_id"], fit_resp.version
    crud.set_baseline_modality(project, version, "table")

    jobs.progress(job["id"], 90, "Storing baseline artifacts")
    bytes_stored = 0
    column_baselines = []
    for col in by_type["numeric"] + by_type["categorical"]:
        if col in inferred:  # high-cardinality categoricals were excluded above
            column_baselines.append({"column_name": col, "col_type": monitored[col],
                                     "state": {"source": "baselines"}})
    for col, (emb, quality) in embedded.items():
        keep = _cap_rows(len(emb))
        key = blob_store.new_key(project, "embeddings", "npy")
        bytes_stored += blob_store.put_array(key, emb[keep].astype(np.float16))
        model = TextAdapter.model_name if monitored[col] == "text" else ImageAdapter.model_name
        column_baselines.append({"column_name": col, "col_type": monitored[col], "embeddings_blob": key,
                                 "state": {"model_name": model, "embedding_dim": int(emb.shape[1]),
                                           "n_ref": int(len(keep)), "quality": quality}})

    reference_rows, rows = None, None
    numcat = [c for c in by_type["numeric"] + by_type["categorical"] if c in inferred]
    if numcat:
        rows = df[numcat].iloc[_cap_rows(len(df))].copy()
        for c in numcat:
            rows[c] = pd.to_numeric(rows[c], errors="coerce") if inferred[c] == "continuous" \
                else rows[c].map(lambda v: None if pd.isna(v) else str(v).strip())
        rows.insert(0, "row_id", rows.index.astype(int))
        buf = io.BytesIO()
        rows.reset_index(drop=True).to_parquet(buf, index=False)
        key = blob_store.new_key(project, "reference_rows", "parquet")
        bytes_stored += blob_store.put_bytes(key, buf.getvalue())
        reference_rows = {"rows_blob": key, "n_rows": len(rows), "sample_seed": REFERENCE_SAMPLE_SEED,
                          "dedup_dropped": duplicate_rows_dropped}

    relationship_rows, relationships_dropped = _relationship_rows(
        payload.get("relationships", []), monitored, inferred, rows, stage, job["owner_email"])

    proposals = {p["name"]: p for p in (stage["profile"] or {}).get("columns", [])}
    schema_rows = []
    for ordinal, col in enumerate(df.columns):
        p, choice = proposals.get(col, {}), choices.get(col)
        final_type = choice["type"] if choice else "ignore"
        final_monitor = bool(choice and choice["monitor"] and final_type != "ignore" and
                             (col in inferred or col in embedded))
        schema_rows.append({
            "column_name": col, "ordinal": ordinal, "proposed_type": p.get("proposed_type"),
            "proposed_monitor": p.get("proposed_monitor"), "confidence": p.get("confidence"),
            "evidence": p.get("evidence"), "reason": p.get("reason"), "alternative_type": p.get("alternative_type"),
            "final_type": final_type, "final_monitor": final_monitor, "decided_by": job["owner_email"],
        })
    crud.insert_table_version_artifacts(project, version, schema_rows, column_baselines, reference_rows,
                                        relationship_rows)
    jobs.discard_stage(stage, "consumed")

    return {
        "version": version,
        "message": fit_resp.message,
        "monitored": {t: [c for c in cols if c in inferred or c in embedded] for t, cols in by_type.items()},
        "not_monitored": [r["column_name"] for r in schema_rows if not r["final_monitor"]],
        "excluded_columns": fit_resp.excluded_columns,
        "relationships": [rel.pair_name(r["col_a"], r["col_b"]) for r in relationship_rows if r["final_monitor"]],
        "relationships_dropped": relationships_dropped,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "cleaning_summary": fit_resp.cleaning_summary,
        "data_quality": {c: q for c, (_, q) in embedded.items()},
        "bytes_stored": bytes_stored,
    }


def _relationship_rows(choices: List[dict], monitored: Dict[str, str], inferred: Dict[str, str],
                       rows: Optional[pd.DataFrame], stage: dict, owner: str) -> Tuple[List[dict], List[str]]:
    """Confirmed relationships with their reference state, plus proposals the
    user rejected (kept for the audit trail, final_monitor=False)."""
    proposals = {(r["col_a"], r["col_b"]): r for r in (stage["profile"] or {}).get("relationships", [])}
    chosen = {}
    for c in choices:
        if c["monitor"]:
            ta, tb = monitored[c["col_a"]], monitored[c["col_b"]]
            chosen[rel.ordered_pair(c["col_a"], ta, c["col_b"], tb)] = rel.kind_for(ta, tb)
    out, dropped = [], []
    for (a, b), kind in chosen.items():
        if rows is None or a not in inferred or b not in inferred:  # e.g. a high-cardinality column was excluded
            dropped.append(rel.pair_name(a, b))
            continue
        p = proposals.get((a, b), {})
        out.append({"col_a": a, "col_b": b, "kind": kind, "proposed": bool(p.get("proposed")),
                    "proposal_reason": p.get("reason"), "proposal_strength": p.get("strength"),
                    "final_monitor": True, "decided_by": owner, "materiality_floor": rel.DEFAULT_FLOORS[kind],
                    "reference_state": rel.reference_state(kind, rows[a].to_numpy(), rows[b].to_numpy())})
    for (a, b), p in proposals.items():
        if p["proposed"] and (a, b) not in chosen:
            out.append({"col_a": a, "col_b": b, "kind": p["kind"], "proposed": True, "proposal_reason": p["reason"],
                        "proposal_strength": p["strength"], "final_monitor": False, "decided_by": owner})
    return out, dropped


def _relationship_tests(job: dict, table: dict, monitored: Dict[str, str], df: pd.DataFrame,
                        ref_rows: Optional[pd.DataFrame],
                        n_column_tests: int) -> Tuple[List[dict], Dict[str, dict], Optional[int]]:
    """(tests for the correction family, per-pair details for the report, null draws used)."""
    confirmed = [r for r in table["relationships"] if r["final_monitor"]]
    if not confirmed or ref_rows is None:
        return [], {}, None
    project, version = job["project_id"], job["payload"]["version"]
    details, prepared = {}, {}
    for r in confirmed:
        a, b, name = r["col_a"], r["col_b"], rel.pair_name(r["col_a"], r["col_b"])
        if a not in df.columns or b not in df.columns:
            details[name] = {"kind": r["kind"], "testable": False, "reason": "Column missing from this batch."}
            continue
        bat = (rel.clean_column(df[a].tolist(), monitored[a]), rel.clean_column(df[b].tolist(), monitored[b]))
        ref = (ref_rows[a].to_numpy(), ref_rows[b].to_numpy())
        prep = rel.prepare_test(r["kind"], r["reference_state"], ref, bat)
        if prep["testable"]:
            prepared[name] = (prep, r)
        else:
            details[name] = prep

    state = crud.get_baseline_version(project, version) or {}
    alpha = CalibrationConfig.from_dict(state.get("calibration_config")).alpha
    draws = rel.null_draws_for_family(n_column_tests + len(prepared), alpha)
    tests = []
    for i, (name, (prep, r)) in enumerate(prepared.items()):
        jobs.progress(job["id"], 50 + int(25 * i / len(prepared)), f"Testing relationship {name}")
        seed = zlib.crc32(f"{project}|{version}|{name}|{prep['n_batch']}".encode())
        done = rel.finish_test(prep, draws, seed)
        details[name] = done
        tests.append({"name": name, "p_value": done["p_value"], "effect_size": done["effect"],
                      "effect_floor": r["materiality_floor"], "label": _REL_LABEL[r["kind"]]})
    return tests, details, draws


# ---------------------------------------------------------
# Analyze job
# ---------------------------------------------------------
def run_analyze(job: dict, run_analysis: Callable[..., Any]) -> dict:
    payload = job["payload"]
    project, version = job["project_id"], payload["version"]
    table = crud.get_table_version(project, version)
    if table is None:
        raise jobs.JobError(f"Baseline version {version} is not a table baseline.")
    schema = {r["column_name"]: r for r in table["schema"]}
    monitored = {n: r["final_type"] for n, r in schema.items() if r["final_monitor"]}
    col_baselines = {c["column_name"]: c for c in table["columns"]}

    stage = crud.get_stage(job["stage_id"])
    df, archive = _open_stage(stage, ("uploaded",))
    extra_metrics: Dict[str, dict] = {}
    extra_issues: List[dict] = []
    data_quality: Dict[str, dict] = {}
    not_tested: Dict[str, str] = {}
    try:
        for col in df.columns:
            if col not in schema:
                extra_issues.append({"column": col, "issue": "unexpected_column", "severity_key": "default"})
        embedding_cols = [c for c, t in monitored.items() if t in ("text", "image")]
        for i, col in enumerate(embedding_cols):
            if col not in df.columns:
                extra_issues.append({"column": col, "issue": "missing_column", "severity_key": "missing_columns"})
                not_tested[col] = "Column missing from this batch."
                continue
            jobs.progress(job["id"], 10 + int(60 * i / max(1, len(embedding_cols))), f"Embedding column '{col}'")
            cur, quality = _embed_column(monitored[col], df[col], archive)
            if quality["invalid"]:
                data_quality[col] = quality
                extra_issues.append({"column": col, "severity_key": "default",
                                     "issue": "invalid_images" if monitored[col] == "image" else "invalid_text",
                                     "counts": quality["invalid"], "rows_sample": quality["invalid_rows_sample"]})
            if quality["valid"] < HARD_MIN_SAMPLES:
                not_tested[col] = f"Only {quality['valid']} usable value(s); need at least {HARD_MIN_SAMPLES}."
                continue
            ref = blob_store.get_array(col_baselines[col]["embeddings_blob"]).astype(np.float32)
            res = EmbeddingDriftDetector(auc_threshold=DCT_AUC_THRESHOLD).analyze(ref, cur)
            extra_metrics[col] = {
                "statistic": res["statistic"], "p_value": None, "drift_detected": res["drift_detected"],
                "effect_size": res["statistic"], "effect_floor": DCT_AUC_THRESHOLD, "decision_mode": "legacy",
                "threshold_used": f"AUC>{DCT_AUC_THRESHOLD} (DCT; outside the Holm family until calibrated)",
            }
    finally:
        if archive:
            archive.close()

    jobs.progress(job["id"], 80, "Running column drift tests")
    numcat = [c for c, t in monitored.items() if t in ("numeric", "categorical")]
    production_data = {c: df[c].tolist() for c in numcat if c in df.columns}
    ref_rows = (pd.read_parquet(io.BytesIO(blob_store.get_bytes(table["reference_rows"]["rows_blob"])))
                if table["reference_rows"] else None)
    rel_tests, rel_details, null_draws = _relationship_tests(job, table, monitored, df, ref_rows,
                                                             len(production_data))
    screening = []
    if ref_rows is not None:
        present = [c for c in numcat if c in df.columns and c in ref_rows.columns]
        screening = rel.screen_emerged(
            {c: ref_rows[c].to_numpy() for c in present},
            {c: rel.clean_column(df[c].tolist(), monitored[c]) for c in present},
            monitored, {frozenset((r["col_a"], r["col_b"])) for r in table["relationships"] if r["final_monitor"]})
    resp = run_analysis(
        project_id=job["public_project_id"], production_data=production_data, client=_client_from_job(job),
        # Idempotency is enforced at the job level (POST /tables/.../analyze), keyed on
        # the uploaded files' hash; re-checking here against analysis_runs would compare
        # a different hash and could reject a legitimate job.
        background_tasks=None, idempotency_key=None, baseline_version=version,
        extra_metrics=extra_metrics, extra_issues=extra_issues, payload_hash=job["payload_hash"],
        batch_size=len(df), report_kind="table", job_id=job["id"],
        extra_family_tests=rel_tests, family_null_draws=null_draws,
    )
    jobs.discard_stage(stage, "consumed")
    report = build_report(resp, monitored, version, len(df), data_quality, not_tested, rel_details)
    report["screening"] = {"emerged_dependencies": screening,
                           "note": "Informational only: never alerts and is not part of the Holm family."}
    return report


def build_report(resp, monitored: Dict[str, str], version: int, n_rows: int,
                 data_quality: Dict[str, dict], not_tested: Dict[str, str],
                 rel_details: Optional[Dict[str, dict]] = None) -> dict:
    metrics = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v)) for k, v in resp.feature_metrics.items()}
    calibrated = any(m.get("decision_mode") == "calibrated" for m in metrics.values())
    column_drift, members, excluded = {}, [], []
    for col, col_type in monitored.items():
        m = metrics.get(col)
        in_family = calibrated and col_type in ("numeric", "categorical") and m is not None
        if m is None:
            reason = not_tested.get(col) or next(
                (i["issue"] for i in resp.schema_report.get(col, [])), "Not tested in this batch.")
            column_drift[col] = {"type": col_type, "test": _TEST_BY_TYPE[col_type], "status": "NOT_TESTED",
                                 "in_family": False, "reason": reason}
            excluded.append({"test": col, "why": reason})
            continue
        column_drift[col] = {"type": col_type, "test": _TEST_BY_TYPE[col_type], **m, "in_family": in_family,
                             "status": "DRIFT" if m["drift_detected"] else "STABLE"}
        if in_family:
            members.append(col)
        elif col_type in ("text", "image"):
            excluded.append({"test": col, "why": "Embedding test decided by the legacy AUC rule until its null "
                                                 "distribution is calibrated (planned for M3)."})
    relationship_drift = {}
    rel_metrics = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v))
                   for k, v in resp.relationship_metrics.items()}
    for name, d in (rel_details or {}).items():
        base = {k: d.get(k) for k in ("kind", "statistic_name", "reference_value", "current_value", "explanation",
                                      "excluded_unseen", "approximate_null", "null_draws")}
        m = rel_metrics.get(name)
        if m is None:
            relationship_drift[name] = {**base, "status": "NOT_TESTED", "in_family": False, "reason": d.get("reason")}
            excluded.append({"test": name, "why": d.get("reason")})
            continue
        relationship_drift[name] = {**base, **{k: m[k] for k in ("p_value", "p_value_adjusted", "effect_size",
                                                                 "effect_floor", "significant", "material",
                                                                 "threshold_used")},
                                    "status": "DRIFT" if m["drift_detected"] else "STABLE", "in_family": calibrated}
        if calibrated:
            members.append(name)
    has_alert_issue = any(i.get("severity") == "alert" for issues in resp.schema_report.values() for i in issues)
    status = "DRIFT" if resp.system_alert_triggered else ("DATA_ISSUES" if has_alert_issue else "STABLE")
    return {
        "baseline_version": version,
        "n_rows": n_rows,
        "overall": {"status": status, "alert": resp.system_alert_triggered, "sustained_alert": resp.sustained_alert,
                    "alert_state": resp.alert_state, "transition": resp.transition,
                    "triggered_by": [c for c, r in {**column_drift, **relationship_drift}.items()
                                     if r["status"] == "DRIFT"]},
        "column_drift": column_drift,
        "relationship_drift": relationship_drift,
        "schema_report": resp.schema_report,
        "data_quality": data_quality,
        "family": {"method": "holm" if calibrated else "none (legacy decision mode)", "members": members,
                   "excluded": excluded},
    }
