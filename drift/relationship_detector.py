"""
Relationship (dependency) drift between pairs of numeric/categorical columns.

Each statistic is built so that a change in either column's OWN distribution
(marginal drift) does not by itself register as relationship drift -- that is
the column tests' job. See docs/unified_table_plan.md, section I.2-I.4.

  num_num  |Spearman_batch - Spearman_ref|. Rank-based, so monotone marginal
           changes (shift, scale, log) leave it unchanged.
  cat_cat  G^2 of the no-three-way-interaction log-linear model [AC][AS][CS]:
           each source keeps its own marginals, only the odds ratios (the
           association pattern) must match. Catches re-pairing at equal
           strength, which comparing Cramer's V would miss.
  num_cat  weighted mean shift in each category's average mid-rank PIT of the
           numeric column (where the category sits in the overall ordering),
           with the batch ECDF re-weighted to the reference's category mix --
           so neither a common monotone numeric shift nor a change in category
           proportions alone moves it. Mid-ranks keep it stable under heavy
           ties (a KS distance on PIT values was not: tie groups land on
           slightly different PIT values per sample). Limitation: it tracks
           relative position, not changes in spread within a category.

Significance: disjoint splits of the reference at the batch's size (the
reference half plays "reference", the other plays "batch"), K draws,
p = (1 + #{T* >= T}) / (K + 1).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kruskal, rankdata

MIN_ROWS = {"num_num": 30, "cat_cat": 50, "num_cat": 40}
MIN_ROWS_PER_CATEGORY = 20
MAX_LEVELS = 20
OTHER = "__other__"
DEFAULT_FLOORS = {"num_num": 0.10, "cat_cat": round(math.log(1.25), 4), "num_cat": 0.05}
STATISTIC_NAMES = {"num_num": "Spearman", "cat_cat": "Cramér's V (bias-corrected)", "num_cat": "median by category"}

# Proposal thresholds (reference data only); conventional "moderate" effect sizes.
PROPOSE_MIN = {"num_num": 0.3, "cat_cat": 0.2, "num_cat": 0.06}
MAX_PROPOSED = 25
MAX_CANDIDATES = 50
MAX_COLUMNS_FOR_PAIRS = 60


def kind_for(type_a: str, type_b: str) -> Optional[str]:
    types = {type_a, type_b}
    if types == {"numeric"}:
        return "num_num"
    if types == {"categorical"}:
        return "cat_cat"
    if types == {"numeric", "categorical"}:
        return "num_cat"
    return None


def ordered_pair(col_a: str, type_a: str, col_b: str, type_b: str) -> Tuple[str, str]:
    """num_cat pairs are stored numeric-first; others alphabetically, so a pair has one name."""
    if {type_a, type_b} == {"numeric", "categorical"}:
        return (col_a, col_b) if type_a == "numeric" else (col_b, col_a)
    return tuple(sorted((col_a, col_b)))


def pair_name(col_a: str, col_b: str) -> str:
    return f"{col_a}<->{col_b}"


# ---------------------------------------------------------
# Category levels (fixed at fit time from the reference)
# ---------------------------------------------------------
def build_levels(values: np.ndarray) -> Dict[str, List[str]]:
    """Frequent reference values keep their own level (at most MAX_LEVELS-1);
    the rest share OTHER. Production values never seen in the reference are
    'unseen' and excluded from relationship tests."""
    uniq, counts = np.unique(values.astype(str), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    min_count = max(5, math.ceil(0.01 * len(values)))
    kept = [str(uniq[i]) for i in order if counts[i] >= min_count][:MAX_LEVELS - 1]
    other = sorted(set(map(str, uniq)) - set(kept))
    return {"kept": kept, "other": other}


def map_levels(values: np.ndarray, levels: Dict[str, List[str]]) -> np.ndarray:
    kept, other = set(levels["kept"]), set(levels["other"])
    return np.array([v if v in kept else (OTHER if v in other else None) for v in values.astype(str)], dtype=object)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def num_num_statistic(ref: Tuple[np.ndarray, np.ndarray], bat: Tuple[np.ndarray, np.ndarray]) -> float:
    return abs(spearman(*bat) - spearman(*ref))


def level_codes(mapped: np.ndarray, levels: List[str]) -> np.ndarray:
    index = {lv: i for i, lv in enumerate(levels)}
    return np.array([index[v] for v in mapped], dtype=int)


def table_from_codes(ca: np.ndarray, cb: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    return np.bincount(ca * n_b + cb, minlength=n_a * n_b).reshape(n_a, n_b).astype(float)


def _contingency(a: np.ndarray, b: np.ndarray, levels_a: List[str], levels_b: List[str]) -> np.ndarray:
    return table_from_codes(level_codes(a, levels_a), level_codes(b, levels_b), len(levels_a), len(levels_b))


def _fit_no_three_way(x: np.ndarray, iters: int = 200, tol: float = 1e-7) -> np.ndarray:
    """Iterative proportional fitting of [AC][AS][CS] on an I x J x 2 table."""
    mu = np.ones_like(x)
    for _ in range(iters):
        prev = mu
        mu = mu * x.sum(axis=2, keepdims=True) / mu.sum(axis=2, keepdims=True)
        mu = mu * x.sum(axis=1, keepdims=True) / mu.sum(axis=1, keepdims=True)
        mu = mu * x.sum(axis=0, keepdims=True) / mu.sum(axis=0, keepdims=True)
        if np.abs(mu - prev).max() < tol:
            break
    return mu


def _interaction(table: np.ndarray) -> np.ndarray:
    """Double-centred log counts: the association terms, free of both marginals."""
    lg = np.log(table + 0.5)
    return lg - lg.mean(axis=1, keepdims=True) - lg.mean(axis=0, keepdims=True) + lg.mean()


def cat_cat_statistic(ref_table: np.ndarray, bat_table: np.ndarray) -> float:
    x = np.stack([ref_table, bat_table], axis=2) + 0.5  # Haldane pseudo-count keeps every cell positive
    mu = _fit_no_three_way(x)
    return float(2 * np.sum(x * np.log(x / mu)))


def cat_cat_effect(ref_table: np.ndarray, bat_table: np.ndarray) -> float:
    return float(np.sqrt(np.mean((_interaction(bat_table) - _interaction(ref_table)) ** 2)))


def cramers_v(table: np.ndarray) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    n = table.sum()
    r, k = table.shape
    if n < 2 or r < 2 or k < 2:
        return 0.0
    expected = np.outer(table.sum(1), table.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    return float(math.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _mid_pit(x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Mid-rank PIT under the (optionally weighted) ECDF of x itself:
    (F(x-) + F(x)) / 2, so a tie group maps to the middle of its mass."""
    w = np.ones(len(x)) if weights is None else weights
    order = np.argsort(x, kind="stable")
    sx, sw = x[order], w[order]
    cum = np.cumsum(sw)
    first = np.searchsorted(sx, sx, side="left")
    last = np.searchsorted(sx, sx, side="right") - 1
    out = np.empty(len(x))
    out[order] = (cum[first] - sw[first] + cum[last]) / (2 * cum[-1])
    return out


def num_cat_statistic(ref: Tuple[np.ndarray, np.ndarray], bat: Tuple[np.ndarray, np.ndarray],
                      categories: List[str]) -> Tuple[float, Dict[str, float]]:
    (rx, rk), (bx, bk) = ref, bat
    p_ref = {c: np.mean(rk == c) for c in categories}
    p_bat = {c: np.mean(bk == c) for c in categories}
    present = [c for c in categories if p_ref[c] > 0 and p_bat[c] > 0]
    if len(present) < 2:
        return 0.0, {}
    weights = np.array([p_ref[c] / p_bat[c] if c in present else 0.0 for c in bk])
    u_ref, u_bat = _mid_pit(rx), _mid_pit(bx, weights)
    per_cat = {c: abs(u_ref[rk == c].mean() - u_bat[bk == c].mean()) for c in present}
    total = sum(p_ref[c] for c in present)
    return float(sum(p_ref[c] / total * d for c, d in per_cat.items())), per_cat


# ---------------------------------------------------------
# Null distribution: disjoint splits of the reference at the batch size
# ---------------------------------------------------------
def split_null_pvalue(statistic, ref_arrays: Tuple[np.ndarray, ...], n_batch: int, observed: float,
                      draws: int, seed: int) -> Tuple[float, bool]:
    """(p_value, approximate). `statistic(ref_part, batch_part)` takes tuples of
    row-aligned arrays. A batch larger than half the reference can't be matched
    with a disjoint split; the largest possible split is used and flagged."""
    n_ref = len(ref_arrays[0])
    size = min(n_batch, n_ref // 2)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        perm = rng.permutation(n_ref)
        b_idx, a_idx = perm[:size], perm[size:]
        null = statistic(tuple(arr[a_idx] for arr in ref_arrays), tuple(arr[b_idx] for arr in ref_arrays))
        exceed += null >= observed
    return (1 + exceed) / (draws + 1), size < n_batch


def null_draws_for_family(family_size: int, alpha: float) -> int:
    """Enough draws that the smallest achievable p (1/(K+1)) clears Holm's
    first threshold alpha/m with room to spare."""
    return max(1000, math.ceil(2 * max(1, family_size) / alpha))


# ---------------------------------------------------------
# Fit-time reference state and analyze-time tests
# ---------------------------------------------------------
def reference_state(kind: str, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """What a relationship needs stored with its baseline version: the level
    maps (fixed at fit so production is mapped the same way) and the
    reference statistic shown in reports."""
    mask = complete_rows(a, b)
    a, b = a[mask], b[mask]
    state: Dict[str, Any] = {"n_ref": int(len(a))}
    if kind == "num_num":
        state["rho_ref"] = spearman(a.astype(float), b.astype(float)) if len(a) > 2 else None
    elif kind == "cat_cat":
        state["levels_a"], state["levels_b"] = build_levels(a), build_levels(b)
    else:
        state["levels"] = build_levels(b)
    return state


def prepare_test(kind: str, state: Dict[str, Any], ref: Tuple[np.ndarray, np.ndarray],
                 bat: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """Observed statistic and everything finish_test needs, or
    {"testable": False, "reason": ...}. Cheap -- no null draws yet, so the
    caller can size the correction family before computing p-values."""
    ref_mask, bat_mask = complete_rows(*ref), complete_rows(*bat)
    (ra, rb), (ba, bb) = (ref[0][ref_mask], ref[1][ref_mask]), (bat[0][bat_mask], bat[1][bat_mask])
    out: Dict[str, Any] = {"kind": kind, "statistic_name": STATISTIC_NAMES[kind]}

    if kind == "num_num":
        ra, rb, ba, bb = (v.astype(float) for v in (ra, rb, ba, bb))
        if min(len(ra), len(ba)) < MIN_ROWS[kind]:
            return {**out, "testable": False, "reason": f"Fewer than {MIN_ROWS[kind]} complete rows."}
        rho_r, rho_b = spearman(ra, rb), spearman(ba, bb)
        return {**out, "testable": True, "observed": abs(rho_b - rho_r), "effect": abs(rho_b - rho_r),
                "stat_fn": num_num_statistic, "ref_arrays": (ra, rb), "n_batch": len(ba),
                "reference_value": round(rho_r, 4), "current_value": round(rho_b, 4),
                "explanation": f"Spearman {rho_r:.2f} -> {rho_b:.2f}"}

    if kind == "cat_cat":
        la, lb = state["levels_a"], state["levels_b"]
        names_a, names_b = la["kept"] + [OTHER], lb["kept"] + [OTHER]
        ma_r, mb_r, ma_b, mb_b = map_levels(ra, la), map_levels(rb, lb), map_levels(ba, la), map_levels(bb, lb)
        seen = np.array([x is not None and y is not None for x, y in zip(ma_b, mb_b)], dtype=bool)
        out["excluded_unseen"] = int((~seen).sum())
        ma_b, mb_b = ma_b[seen], mb_b[seen]
        if min(len(ma_r), len(ma_b)) < MIN_ROWS[kind]:
            return {**out, "testable": False, "reason": f"Fewer than {MIN_ROWS[kind]} complete rows."}
        ca_r, cb_r = level_codes(ma_r, names_a), level_codes(mb_r, names_b)
        ca_b, cb_b = level_codes(ma_b, names_a), level_codes(mb_b, names_b)
        n_a, n_b = len(names_a), len(names_b)
        t_ref, t_bat = table_from_codes(ca_r, cb_r, n_a, n_b), table_from_codes(ca_b, cb_b, n_a, n_b)

        def stat(r, b):
            return cat_cat_statistic(table_from_codes(*r, n_a, n_b), table_from_codes(*b, n_a, n_b))

        delta = _interaction(t_bat) - _interaction(t_ref)
        top = np.dstack(np.unravel_index(np.argsort(-np.abs(delta), axis=None)[:3], delta.shape))[0]
        changed = ", ".join(f"{names_a[i]} x {names_b[j]} ({delta[i, j]:+.2f})" for i, j in top)
        v_r, v_b = cramers_v(t_ref), cramers_v(t_bat)
        return {**out, "testable": True, "observed": cat_cat_statistic(t_ref, t_bat),
                "effect": cat_cat_effect(t_ref, t_bat), "stat_fn": stat, "ref_arrays": (ca_r, cb_r),
                "n_batch": len(ca_b), "reference_value": round(v_r, 4), "current_value": round(v_b, 4),
                "explanation": f"Cramér's V {v_r:.2f} -> {v_b:.2f}; largest pairing changes (log-odds): {changed}"}

    levels = state["levels"]
    mk_r, mk_b = map_levels(rb, levels), map_levels(bb, levels)
    seen = np.array([k is not None for k in mk_b], dtype=bool)
    out["excluded_unseen"] = int((~seen).sum())
    ba, mk_b = ba[seen].astype(float), mk_b[seen]
    ra = ra.astype(float)
    categories = [c for c in levels["kept"] + [OTHER]
                  if np.sum(mk_r == c) >= MIN_ROWS_PER_CATEGORY and np.sum(mk_b == c) >= MIN_ROWS_PER_CATEGORY]
    if len(categories) < 2:
        return {**out, "testable": False,
                "reason": f"Fewer than 2 categories with {MIN_ROWS_PER_CATEGORY}+ rows in both reference and batch."}
    keep_r, keep_b = np.isin(mk_r, categories), np.isin(mk_b, categories)
    ref_arrays = (ra[keep_r], mk_r[keep_r])
    observed, per_cat = num_cat_statistic(ref_arrays, (ba[keep_b], mk_b[keep_b]), categories)
    worst = sorted(per_cat, key=lambda c: -per_cat[c])[:2]
    detail = "; ".join(f"{c}: median {np.median(ref_arrays[0][ref_arrays[1] == c]):.3g} -> "
                       f"{np.median(ba[mk_b == c]):.3g}" for c in worst)
    return {**out, "testable": True, "observed": observed, "effect": observed,
            "stat_fn": lambda r, b: num_cat_statistic(r, b, categories)[0], "ref_arrays": ref_arrays,
            "n_batch": int(keep_b.sum()), "reference_value": None, "current_value": None,
            "explanation": f"Largest shifts in category position: {detail}"}


def finish_test(prepared: Dict[str, Any], draws: int, seed: int) -> Dict[str, Any]:
    p, approximate = split_null_pvalue(prepared["stat_fn"], prepared["ref_arrays"], prepared["n_batch"],
                                       prepared["observed"], draws, seed)
    return {**{k: v for k, v in prepared.items() if k not in ("stat_fn", "ref_arrays")},
            "p_value": p, "approximate_null": approximate, "null_draws": draws}


# ---------------------------------------------------------
# Proposals (reference data only, at profile time)
# ---------------------------------------------------------
def _clean_numeric(values) -> np.ndarray:
    import pandas as pd
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _clean_categorical(values) -> np.ndarray:
    return np.array([None if v is None or (isinstance(v, float) and v != v) else str(v).strip()
                     for v in values], dtype=object)


def clean_column(values, col_type: str) -> np.ndarray:
    return _clean_numeric(values) if col_type == "numeric" else _clean_categorical(values)


def complete_rows(*cols: np.ndarray) -> np.ndarray:
    mask = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        mask &= np.array([v is not None and not (isinstance(v, float) and v != v) for v in c])
    return mask


def reference_strength(kind: str, a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Association strength on the reference, on a scale comparable across kinds
    (|rho|, V, sqrt(epsilon^2)), or None if there's too little data."""
    mask = complete_rows(a, b)
    a, b = a[mask], b[mask]
    if len(a) < MIN_ROWS[kind]:
        return None
    if kind == "num_num":
        return abs(spearman(a.astype(float), b.astype(float)))
    if kind == "cat_cat":
        la, lb = build_levels(a), build_levels(b)
        ma, mb = map_levels(a, la), map_levels(b, lb)
        return cramers_v(_contingency(ma, mb, la["kept"] + [OTHER], lb["kept"] + [OTHER]))
    groups = [a[b == c].astype(float) for c in np.unique(b.astype(str)) if np.sum(b == c) >= 5]
    if len(groups) < 2 or all(np.ptp(g) == 0 for g in groups):
        return None
    h = kruskal(*groups).statistic
    return float(math.sqrt(max(0.0, h / (len(a) - 1))))


EMERGED = {"num_num": (0.1, 0.3), "cat_cat": (0.1, 0.2)}  # (weak in reference, strong in batch)


def screen_emerged(ref: Dict[str, np.ndarray], bat: Dict[str, np.ndarray], types: Dict[str, str],
                   watched: set) -> List[Dict[str, Any]]:
    """Informational only (never alerts, never in the family): unwatched
    numeric or categorical pairs that were weak in the reference but are
    strong in this batch. Watching only pairs that were strong at fit time
    would otherwise miss a dependency that appears in production."""
    names = [c for c in types if c in ref and c in bat and types[c] in ("numeric", "categorical")]
    if len(names) > MAX_COLUMNS_FOR_PAIRS:
        return []
    found = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            kind = kind_for(types[x], types[y])
            if kind not in EMERGED or frozenset((x, y)) in watched:
                continue
            s_ref, s_bat = reference_strength(kind, ref[x], ref[y]), reference_strength(kind, bat[x], bat[y])
            weak, strong = EMERGED[kind]
            if s_ref is not None and s_bat is not None and s_ref < weak and s_bat >= strong:
                found.append({"col_a": x, "col_b": y, "kind": kind, "reference": round(s_ref, 3),
                              "current": round(s_bat, 3)})
    return found


def propose_relationships(columns: Dict[str, Tuple[str, np.ndarray]]) -> List[Dict[str, Any]]:
    """Candidate pairs among numeric/categorical columns, strongest first.
    `columns` maps name -> (type, cleaned values). The top MAX_PROPOSED pairs
    above their kind's threshold are proposed; the rest are listed so the user
    can add them by hand."""
    names = [n for n, (t, _) in columns.items() if t in ("numeric", "categorical")]
    if len(names) > MAX_COLUMNS_FOR_PAIRS:
        return []
    candidates = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            tx, ty = columns[x][0], columns[y][0]
            kind = kind_for(tx, ty)
            a, b = ordered_pair(x, tx, y, ty)
            strength = reference_strength(kind, columns[a][1], columns[b][1])
            if strength is None:
                continue
            threshold = PROPOSE_MIN[kind] if kind != "num_cat" else math.sqrt(PROPOSE_MIN[kind])
            candidates.append({"col_a": a, "col_b": b, "kind": kind, "strength": round(strength, 3),
                               "meets_threshold": strength >= threshold})
    candidates.sort(key=lambda c: -c["strength"])
    proposed = 0
    for c in candidates:
        c["proposed"] = c.pop("meets_threshold") and proposed < MAX_PROPOSED
        proposed += c["proposed"]
        label = {"num_num": "Spearman |rho|", "cat_cat": "Cramér's V", "num_cat": "rank effect size"}[c["kind"]]
        c["reason"] = f"{label} = {c['strength']} in the reference" + ("" if c["proposed"] else " (weak)")
    return candidates[:MAX_CANDIDATES]
