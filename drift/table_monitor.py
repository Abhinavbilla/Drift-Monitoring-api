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
M3: text/image column tests get a p-value from a real-embedding null, and
probes (text/image -> numeric/categorical) and text<->image matching watch
relationships involving embeddings. Both stay outside the Holm family and do
not alert ("report-only") until their validation gate is approved.
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
from drift import embedding_tests as et
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
_EMBEDDING = ("text", "image")
_NUMCAT = ("numeric", "categorical")
MAX_EMBEDDING_PROPOSALS = 15


def pair_kind(type_a: str, type_b: str) -> Optional[str]:
    """Relationship kind for two column types, or None if unsupported (e.g. text<->text)."""
    if type_a in _NUMCAT and type_b in _NUMCAT:
        return rel.kind_for(type_a, type_b)
    if {type_a, type_b} == {"text", "image"}:
        return "text_image"
    if (type_a in _EMBEDDING) != (type_b in _EMBEDDING):
        return "probe"
    return None


def ordered(col_a: str, type_a: str, col_b: str, type_b: str) -> Tuple[str, str]:
    """One canonical order per pair: embedding column first for probes, text first for text<->image."""
    kind = pair_kind(type_a, type_b)
    if kind == "probe":
        return (col_a, col_b) if type_a in _EMBEDDING else (col_b, col_a)
    if kind == "text_image":
        return (col_a, col_b) if type_a == "text" else (col_b, col_a)
    return rel.ordered_pair(col_a, type_a, col_b, type_b)


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


def _embed_text(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """(embeddings of the valid rows in order, a status per row)."""
    statuses = np.array(["ok" if v is not None and not (isinstance(v, float) and v != v) and str(v).strip()
                         else "empty_or_null" for v in values.tolist()])
    texts = [str(v) for v, st in zip(values.tolist(), statuses) if st == "ok"]
    adapter = TextAdapter()
    chunks = [adapter.transform(texts[i:i + EMBED_CHUNK]) for i in range(0, len(texts), EMBED_CHUNK)]
    return (np.vstack(chunks) if chunks else np.empty((0, 384), dtype=np.float32)), statuses


def _embed_images(values: pd.Series, archive: Optional[ImageArchive]) -> Tuple[np.ndarray, np.ndarray]:
    """(embeddings of the valid rows in order, a status per row). Images are
    validated and embedded in chunks, never all decoded in memory at once."""
    adapter = ImageAdapter()
    embs, pending, statuses = [], [], []
    for value in values.tolist():
        status, raw = load_image(value, archive)
        statuses.append(status)
        if status == "ok":
            pending.append(raw)
            if len(pending) >= EMBED_CHUNK:
                embs.append(adapter.transform(pending))
                pending = []
    if pending:
        embs.append(adapter.transform(pending))
    return (np.vstack(embs) if embs else np.empty((0, 512), dtype=np.float32)), np.array(statuses)


def _embed_column(col_type: str, values: pd.Series, archive) -> Tuple[np.ndarray, np.ndarray]:
    return _embed_text(values) if col_type == "text" else _embed_images(values, archive)


def _quality(statuses: np.ndarray) -> Dict[str, Any]:
    bad = np.flatnonzero(statuses != "ok")
    counts = dict(zip(*np.unique(statuses[bad], return_counts=True))) if len(bad) else {}
    return {"valid": int(len(statuses) - len(bad)), "invalid": {str(k): int(v) for k, v in counts.items()},
            "invalid_rows_sample": bad[:20].tolist()}


def _target_values(values: pd.Series, target_type: str, levels: Optional[dict]) -> np.ndarray:
    """Cleaned target values for a probe; None/NaN where unusable (incl. unseen categories)."""
    clean = rel.clean_column(values.tolist(), target_type)
    if target_type == "categorical" and levels is not None:
        mask = rel.complete_rows(clean)
        out = np.full(len(clean), None, dtype=object)
        out[mask] = rel.map_levels(clean[mask], levels)
        return out
    return clean


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
        cache = _cache_embeddings(job, stage, df, columns, archive)
        relationships += _propose_embedding_relationships(df, columns, cache)
        zip_info = None if archive is None else {"image_entries": len(archive.by_path), "rejected": archive.rejected}
    finally:
        if archive:
            archive.close()
    result = {"stage_id": stage["id"], "n_rows": len(df), "columns": columns, "relationships": relationships,
              "image_zip": zip_info, "expires_at": stage["expires_at"],
              "embedding_cache": {c: {"key": v["key"], "type": v["type"]} for c, v in cache.items()}}
    crud.update_stage(stage["id"], status="profiled", profile=result)
    return result


def _cache_embeddings(job: dict, stage: dict, df: pd.DataFrame, columns: List[dict], archive) -> Dict[str, dict]:
    """Embeds proposed text/image columns once, at profile time: the vectors
    feed relationship proposals and are reused by the fit (deleted with the stage)."""
    cache = {}
    targets = [p for p in columns if p["proposed_type"] in _EMBEDDING and p["proposed_monitor"]]
    for i, p in enumerate(targets):
        jobs.progress(job["id"], 40 + int(40 * i / len(targets)), f"Embedding column '{p['name']}'")
        emb, statuses = _embed_column(p["proposed_type"], df[p["name"]], archive)
        key = blob_store.new_key(stage["project_id"], "staged", "npz")
        blob_store.put_arrays(key, emb=emb.astype(np.float16), status=statuses)
        cache[p["name"]] = {"key": key, "type": p["proposed_type"], "emb": emb, "status": statuses}
    return cache


def _propose_embedding_relationships(df: pd.DataFrame, columns: List[dict], cache: Dict[str, dict]) -> List[dict]:
    types = {p["name"]: p["proposed_type"] for p in columns if p["proposed_monitor"]}
    projected = {}
    for col, c in cache.items():
        if len(c["emb"]) > et.MIN_ROWS:
            projected[col] = (et.project(c["emb"], et.fit_pca(c["emb"])), np.flatnonzero(c["status"] == "ok"))
    candidates = []
    for source, (x, rows) in projected.items():
        for target, t_type in types.items():
            if t_type not in _NUMCAT:
                continue
            y_all = rel.clean_column(df[target].iloc[rows].tolist(), t_type)
            if t_type == "categorical":
                ok = rel.complete_rows(y_all)
                y_all = np.where(ok, y_all, None)
                if ok.sum():
                    y_all[ok] = rel.map_levels(y_all[ok], rel.build_levels(y_all[ok]))
            mask = rel.complete_rows(y_all)
            strength = et.probe_strength(x[mask], y_all[mask].astype(str if t_type == "categorical" else float), t_type)
            if strength is not None:
                threshold = et.PROPOSE_MIN[f"probe_{t_type}"]
                what = "balanced accuracy above chance" if t_type == "categorical" else "Spearman of predictions"
                candidates.append({"col_a": source, "col_b": target, "kind": "probe", "strength": round(strength, 3),
                                   "proposed": strength >= threshold,
                                   "reason": f"{source} predicts {target} ({what} = {strength:.2f} in the reference)"})
    texts = [c for c in projected if cache[c]["type"] == "text"]
    images = [c for c in projected if cache[c]["type"] == "image"]
    for t in texts:
        for im in images:
            both = np.intersect1d(projected[t][1], projected[im][1])
            xt = projected[t][0][np.searchsorted(projected[t][1], both)]
            xi = projected[im][0][np.searchsorted(projected[im][1], both)]
            auc = et.cross_fitted_matching(xi, xt)
            if auc is not None:
                strength = auc - 0.5
                candidates.append({"col_a": t, "col_b": im, "kind": "text_image", "strength": round(strength, 3),
                                   "proposed": strength >= et.PROPOSE_MIN["text_image"],
                                   "reason": f"images and texts match each other (AUC {auc:.2f} in the reference)"})
    candidates.sort(key=lambda c: -c["strength"])
    for i, c in enumerate(candidates):
        c["proposed"] = c["proposed"] and i < MAX_EMBEDDING_PROPOSALS
    return candidates


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
        df = df.drop_duplicates()
        kept = df.index.to_numpy()  # original row positions, to reuse profile-time embeddings
        df = df.reset_index(drop=True)
        duplicate_rows_dropped = rows_before - len(df)
        cache = (stage["profile"] or {}).get("embedding_cache") or {}

        # Embed text/image columns FIRST: a failure here must not leave a
        # half-written baseline version behind.
        embedded: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}
        for i, col in enumerate(by_type["text"] + by_type["image"]):
            jobs.progress(job["id"], 10 + int(60 * i / max(1, len(by_type["text"]) + len(by_type["image"]))),
                          f"Embedding column '{col}'")
            if col in cache and cache[col]["type"] == monitored[col]:
                arrays = blob_store.get_arrays(cache[col]["key"])
                rank = np.cumsum(arrays["status"] == "ok") - 1
                statuses = arrays["status"][kept]
                emb = arrays["emb"][rank[kept][statuses == "ok"]].astype(np.float32)
            else:
                emb, statuses = _embed_column(monitored[col], df[col], archive)
            quality = _quality(statuses)
            if quality["valid"] < HARD_MIN_SAMPLES:
                raise jobs.JobError(
                    f"Column '{col}' has only {quality['valid']} usable {monitored[col]} value(s) "
                    f"(need at least {HARD_MIN_SAMPLES}); invalid: {quality['invalid']}. "
                    f"Fix the data or turn monitoring off for this column.")
            embedded[col] = (emb, np.flatnonzero(statuses == "ok"), quality)
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
    stored: Dict[str, dict] = {}
    for col, (emb, valid_rows, quality) in embedded.items():
        keep = _cap_rows(len(emb))
        ref = emb[keep]
        pca = et.fit_pca(ref)
        key, pca_key = blob_store.new_key(project, "embeddings", "npz"), blob_store.new_key(project, "pca", "npz")
        bytes_stored += blob_store.put_arrays(key, emb=ref.astype(np.float16), row_ids=valid_rows[keep])
        bytes_stored += blob_store.put_arrays(pca_key, **pca)
        stored[col] = {"x": et.project(ref, pca), "row_ids": valid_rows[keep]}
        model = TextAdapter.model_name if monitored[col] == "text" else ImageAdapter.model_name
        column_baselines.append({"column_name": col, "col_type": monitored[col], "embeddings_blob": key,
                                 "state": {"model_name": model, "embedding_dim": int(emb.shape[1]),
                                           "n_ref": int(len(keep)), "quality": quality, "pca_blob": pca_key,
                                           "pca_dim": int(pca["components"].shape[0])}})

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
        payload.get("relationships", []), monitored, inferred, rows, stage, job["owner_email"], df, stored, project)
    bytes_stored += sum(r.pop("bytes", 0) for r in relationship_rows)

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
        "data_quality": {c: q for c, (_, _, q) in embedded.items()},
        "bytes_stored": bytes_stored,
    }


def _relationship_rows(choices: List[dict], monitored: Dict[str, str], inferred: Dict[str, str],
                       rows: Optional[pd.DataFrame], stage: dict, owner: str, df: pd.DataFrame,
                       stored: Dict[str, dict], project: str) -> Tuple[List[dict], List[str]]:
    """Confirmed relationships with their reference state, plus proposals the
    user rejected (kept for the audit trail, final_monitor=False)."""
    proposals = {(r["col_a"], r["col_b"]): r for r in (stage["profile"] or {}).get("relationships", [])}
    chosen = {}
    for c in choices:
        if c["monitor"]:
            ta, tb = monitored[c["col_a"]], monitored[c["col_b"]]
            chosen[ordered(c["col_a"], ta, c["col_b"], tb)] = pair_kind(ta, tb)
    out, dropped = [], []
    for (a, b), kind in chosen.items():
        p = proposals.get((a, b), {})
        base = {"col_a": a, "col_b": b, "kind": kind, "proposed": bool(p.get("proposed")),
                "proposal_reason": p.get("reason"), "proposal_strength": p.get("strength"),
                "final_monitor": True, "decided_by": owner}
        if kind in ("probe", "text_image"):
            fitted = _fit_embedding_relationship(kind, a, b, monitored, df, stored, project)
            if fitted is None:
                dropped.append(rel.pair_name(a, b))
            else:
                out.append({**base, **fitted, "materiality_floor": et.FLOORS[kind]})
            continue
        if rows is None or a not in inferred or b not in inferred:  # e.g. a high-cardinality column was excluded
            dropped.append(rel.pair_name(a, b))
            continue
        out.append({**base, "materiality_floor": rel.DEFAULT_FLOORS[kind],
                    "reference_state": rel.reference_state(kind, rows[a].to_numpy(), rows[b].to_numpy())})
    for (a, b), p in proposals.items():
        if p["proposed"] and (a, b) not in chosen:
            out.append({"col_a": a, "col_b": b, "kind": p["kind"], "proposed": True, "proposal_reason": p["reason"],
                        "proposal_strength": p["strength"], "final_monitor": False, "decided_by": owner})
    return out, dropped


def _fit_embedding_relationship(kind: str, a: str, b: str, monitored: Dict[str, str], df: pd.DataFrame,
                                stored: Dict[str, dict], project: str) -> Optional[dict]:
    """Fits a probe (a = text/image column, b = numeric/categorical target) or a
    text<->image matching map on the reference; None if there's too little data."""
    key = blob_store.new_key(project, "probe", "npz")
    if kind == "probe":
        if a not in stored:
            return None
        t_type = monitored[b]
        x, row_ids = stored[a]["x"], stored[a]["row_ids"]
        raw = rel.clean_column(df[b].iloc[row_ids].tolist(), t_type)
        levels = rel.build_levels(raw[rel.complete_rows(raw)]) if t_type == "categorical" else None
        y = _target_values(df[b].iloc[row_ids], t_type, levels)
        mask = rel.complete_rows(y)
        y = y[mask].astype(str if t_type == "categorical" else float)
        score = et.cross_fitted_score(x[mask], y, t_type)
        if score is None:
            return None
        nbytes = blob_store.put_arrays(key, emb_rows=np.flatnonzero(mask), y_ref=y,
                                       **et.fit_probe(x[mask], y, t_type))
        return {"probe_blob": key, "bytes": nbytes,
                "reference_state": {"target_type": t_type, "levels": levels, "ref_score": score, "n_ref": int(len(y))}}
    if a not in stored or b not in stored:
        return None
    both = np.intersect1d(stored[a]["row_ids"], stored[b]["row_ids"])
    it, ii = np.searchsorted(stored[a]["row_ids"], both), np.searchsorted(stored[b]["row_ids"], both)
    xt, xi = stored[a]["x"][it], stored[b]["x"][ii]
    score = et.cross_fitted_matching(xi, xt)
    if score is None:
        return None
    nbytes = blob_store.put_arrays(key, rows_text=it, rows_image=ii, **et.fit_matching(xi, xt))
    return {"probe_blob": key, "bytes": nbytes, "reference_state": {"ref_score": score, "n_ref": int(len(both))}}


def _relationship_tests(job: dict, table: dict, monitored: Dict[str, str], df: pd.DataFrame,
                        ref_rows: Optional[pd.DataFrame],
                        n_column_tests: int) -> Tuple[List[dict], Dict[str, dict], Optional[int]]:
    """(tests for the correction family, per-pair details for the report, null draws used)."""
    confirmed = [r for r in table["relationships"] if r["final_monitor"] and r["kind"] in rel.MIN_ROWS]
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
    embedding_state: Dict[str, dict] = {}  # projected reference/batch vectors, reused by probes
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
            cur, statuses = _embed_column(monitored[col], df[col], archive)
            quality = _quality(statuses)
            if quality["invalid"]:
                data_quality[col] = quality
                extra_issues.append({"column": col, "severity_key": "default",
                                     "issue": "invalid_images" if monitored[col] == "image" else "invalid_text",
                                     "counts": quality["invalid"], "rows_sample": quality["invalid_rows_sample"]})
            if quality["valid"] < HARD_MIN_SAMPLES:
                not_tested[col] = f"Only {quality['valid']} usable value(s); need at least {HARD_MIN_SAMPLES}."
                continue
            ref_x, pca, ref_row_ids = _load_reference_embeddings(col_baselines[col])
            bat_x = et.project(cur, pca)
            res = et.dct_test(EmbeddingDriftDetector()._compute_auc, ref_x, bat_x, et.NULL_DRAWS,
                              zlib.crc32(f"{project}|{version}|{col}|{len(bat_x)}".encode()))
            embedding_state[col] = {"ref_x": ref_x, "bat_x": bat_x, "bat_rows": np.flatnonzero(statuses == "ok"),
                                    "p_value": res["p_value"]}
            extra_metrics[col] = {
                "statistic": res["auc"], "p_value": res["p_value"], "drift_detected": res["auc"] > DCT_AUC_THRESHOLD,
                "effect_size": res["auc"], "effect_floor": DCT_AUC_THRESHOLD, "decision_mode": "legacy",
                "threshold_used": f"AUC>{DCT_AUC_THRESHOLD} on PCA-{ref_x.shape[1]} embeddings (decided outside the "
                                  f"Holm family; p_value is from this project's own reference null"
                                  + (", Gaussian tail" if res["tail_extrapolated"] else "") + ")",
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
    alpha = CalibrationConfig.from_dict((crud.get_baseline_version(project, version) or {}).get("calibration_config")).alpha
    rel_details.update(_embedding_relationship_tests(table, monitored, df, embedding_state, project, version, alpha))
    jobs.discard_stage(stage, "consumed")
    report = build_report(resp, monitored, version, len(df), data_quality, not_tested, rel_details)
    report["screening"] = {"emerged_dependencies": screening,
                           "note": "Informational only: never alerts and is not part of the Holm family."}
    return report


def _load_reference_embeddings(col_baseline: dict) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """(projected reference, PCA, row ids). M1 baselines stored a bare .npy and
    no PCA; for those the PCA is fitted on the fly (deterministic)."""
    key = col_baseline["embeddings_blob"]
    if key.endswith(".npy"):
        emb, row_ids = blob_store.get_array(key).astype(np.float32), None
    else:
        arrays = blob_store.get_arrays(key)
        emb, row_ids = arrays["emb"].astype(np.float32), arrays["row_ids"]
    pca_key = col_baseline["state"].get("pca_blob")
    pca = blob_store.get_arrays(pca_key) if pca_key else et.fit_pca(emb)
    return et.project(emb, pca), pca, row_ids


def _embedding_relationship_tests(table: dict, monitored: Dict[str, str], df: pd.DataFrame,
                                  emb: Dict[str, dict], project: str, version: int, alpha: float) -> Dict[str, dict]:
    """Report-only results for probes and text<->image matching: shown in the
    report with a status, but outside the Holm family and never alerting."""
    out = {}
    for r in table["relationships"]:
        if not r["final_monitor"] or r["kind"] not in ("probe", "text_image"):
            continue
        a, b, name, state = r["col_a"], r["col_b"], rel.pair_name(r["col_a"], r["col_b"]), r["reference_state"]
        seed = zlib.crc32(f"{project}|{version}|{name}".encode())
        params = blob_store.get_arrays(r["probe_blob"])
        if a not in emb or (r["kind"] == "text_image" and b not in emb) or (r["kind"] == "probe" and b not in df.columns):
            out[name] = {"kind": r["kind"], "testable": False, "reason": "A column is missing or untested in this batch."}
            continue
        if r["kind"] == "probe":
            y = _target_values(df[b].iloc[emb[a]["bat_rows"]], state["target_type"], state.get("levels"))
            mask = rel.complete_rows(y)
            res = et.probe_test(emb[a]["ref_x"][params["emb_rows"]], params["y_ref"], emb[a]["bat_x"][mask],
                                y[mask].astype(str if state["target_type"] == "categorical" else float),
                                state["target_type"], params,
                                state["ref_score"], et.NULL_DRAWS, seed)
            sources = [a]
        else:
            both = np.intersect1d(emb[a]["bat_rows"], emb[b]["bat_rows"])
            xt = emb[a]["bat_x"][np.searchsorted(emb[a]["bat_rows"], both)]
            xi = emb[b]["bat_x"][np.searchsorted(emb[b]["bat_rows"], both)]
            res = et.matching_test(emb[b]["ref_x"][params["rows_image"]], emb[a]["ref_x"][params["rows_text"]],
                                   xi, xt, params, state["ref_score"], et.NULL_DRAWS, seed)
            sources = [a, b]
        res["kind"] = r["kind"]
        if res["testable"]:
            res["report_only"] = True
            res["drift_detected"] = bool(res["p_value"] < alpha and res["effect"] >= r["materiality_floor"])
            res["effect_floor"] = r["materiality_floor"]
            res["confounded_by"] = [c for c in sources if emb[c]["p_value"] < alpha]
        out[name] = res
    return out


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
            excluded.append({"test": col, "why": "Embedding test decided by the AUC rule outside the family until "
                                                 "its real-embedding calibration is approved; its p_value is shown."})
    relationship_drift = {}
    rel_metrics = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v))
                   for k, v in resp.relationship_metrics.items()}
    for name, d in (rel_details or {}).items():
        base = {k: d.get(k) for k in ("kind", "statistic_name", "reference_value", "current_value", "explanation",
                                      "excluded_unseen", "approximate_null", "null_draws")}
        if d.get("report_only"):
            relationship_drift[name] = {**base, "p_value": d["p_value"], "effect_size": d["effect"],
                                        "effect_floor": d["effect_floor"], "tail_extrapolated": d["tail_extrapolated"],
                                        "confounded_by": d["confounded_by"], "in_family": False, "report_only": True,
                                        "status": "DRIFT" if d["drift_detected"] else "STABLE"}
            excluded.append({"test": name, "why": "Report-only: probe/matching tests do not alert until validated."})
            continue
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
                                     if r["status"] == "DRIFT" and not r.get("report_only")]},
        "column_drift": column_drift,
        "relationship_drift": relationship_drift,
        "schema_report": resp.schema_report,
        "data_quality": data_quality,
        "family": {"method": "holm" if calibrated else "none (legacy decision mode)", "members": members,
                   "excluded": excluded},
    }
