"""A complete synthetic input set for the end-to-end run.

``write_synthetic_inputs(root, seed)`` writes into ``root``, a directory used as
``$LCP_DATA_ROOT``, every restricted input the pipeline reads
(``config.RESTRICTED_INPUTS``, the RoBERTa file of ``config.DERIVED_FILES``), and the
marker file (``config.SYNTHETIC_MARKER_FILE``) without which the runner refuses the
smoke configuration. Only this module writes that marker.

Everything is synthetic and deterministic given ``seed``: IDs are ``SYN000001``,
``SYN000002``, ...; every value is drawn with ``numpy.random.default_rng(seed)`` from
made-up latent traits (ability, social position, behaviour, internalising,
motivation). Nothing is copied from or modelled on a real participant. Taken from
public files in the repository: the NCDS codes and their table (``data/variables.xlsx``)
and the ``aspiration_n2771`` strings of ``data/occupation_aspiration_mapping.xlsx``,
used as the value labels of ``n2771``. The label texts of the other codes are made up
(except the missing-value strings, which are those ``clean_ncds`` blanks). The column
names of the tool outputs (SALAT, readability, embeddings, polygenic scores) are made
up too, because the real ones are not known here: ``nwords`` (the R's text-length
predictor, ``_targets.R:L261``) is placed in the TAALED and TAALES files, a choice made
for the synthetic data, not a fact about the tools.

Planted relations (so that some models have a clearly positive R²): ability drives
the cognitive test scores, the teacher's ability ratings, the essay length and several
essay metrics and embedding dimensions, the highest qualification and one polygenic
score; behaviour and internalising drive the teacher's behaviour items; motivation
drives the motivation items.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from llm_cong_predict import config
from llm_cong_predict.cleaning.clean_ncds import MISSING_LABELS
from llm_cong_predict.io.readers import ESSAY_ID_SEPARATOR, ESSAY_WORDS_SEPARATOR, read_datalist

SPELLING_CATEGORIES = ("grammar", "misspelling", "typographical", "locale-violation", "duplication",
                       "style", "whitespace", "uncategorized", "inconsistency")
GPT35_DIMS = 16  # made up; the real text-embedding-ada-002 has 1536
GPT4_DIMS = 24  # made up; the real text-embedding-3-large has 3072
ROBERTA_DIMS = 768  # fixed by the R: paste0("roberta_dim_", 1:768) (_targets.R:L139)
N_PGS = 4
NCDS_FILES = ("ncds_1_2_3", "ncds_4", "ncds_5", "ncds_6", "ncds_7", "ncds_8", "ncds_9",
              "ncds_occ_2", "ncds_occ_5", "ncds_occ_6", "ncds_occ_7", "ncds_occ_8")
# Share of cohort members present in each file (made up; sweep 0-3 holds everyone).
FILE_COVERAGE = {"ncds_1_2_3": 1.0, "ncds_4": 0.9, "ncds_5": 0.95, "ncds_6": 0.85, "ncds_7": 0.8,
                 "ncds_8": 0.85, "ncds_9": 0.75, "ncds_occ_2": 0.97, "ncds_occ_5": 0.8,
                 "ncds_occ_6": 0.75, "ncds_occ_7": 0.7, "ncds_occ_8": 0.7}
UPPER_CASE_ID_FILES = ("ncds_occ_2", "ncds_8")  # "NCDSID", so read_ncds' lower-casing is used


def synthetic_id(i: int) -> str:
    return f"SYN{i:06d}"


def _file_for(code: str, sweep: float) -> str:
    """Each code of the table goes into exactly one NCDS file, by sweep."""
    if code.lower() == "n2snssec":  # the father's NS-SEC comes from the occupation coding
        return "ncds_occ_2"
    return {0: "ncds_1_2_3", 2: "ncds_1_2_3", 3: "ncds_1_2_3", 5: "ncds_5", 8: "ncds_8"}[int(sweep)]


RATING_5 = {1: "Well above average", 2: "Above average", 3: "Average", 4: "Below average",
            5: "Well below average"}
APPLIES_3 = {1: "Does not apply", 2: "Applies somewhat", 3: "Certainly applies"}
AGREE_5 = {1: "Strongly agree", 2: "Agree", 3: "Uncertain", 4: "Disagree", 5: "Strongly disagree"}
CLASS_6 = {1: "Class I", 2: "Class II", 3: "Class III", 4: "Class IV", 5: "Class V", 6: "Class VI"}
QUAL_6 = {0: "No qualifications", 1: "Level 1", 2: "Level 2", 3: "Level 3", 4: "Level 4", 5: "Level 5"}
AGE_LEFT = {k: f"Age band {k}" for k in range(1, 13)}
NA_1 = {-1: "Not applicable"}

EXTERNALISING = ("s3_te_restlessness", "s3_te_squirmy", "s3_te_destructive", "s3_te_fight_others",
                 "s3_te_irritable", "s3_te_disobedient", "s3_te_cannot_settle", "s3_te_lying",
                 "s3_te_steals", "s3_te_resentful", "s3_te_bully")
INTERNALISING = ("s3_te_worried", "s3_te_solitary", "s3_te_miserable", "s3_te_fearful",
                 "s3_te_cries_in_school")
BSAG_INTERNAL = ("s2_te_unforthcomingness", "s2_te_depression", "s2_te_withdrawal",
                 "s2_te_anxiety_adults", "s2_te_anxiety_children", "s2_te_miscellneous_symptoms")


def _variable(full_name: str, rng, lat: dict, n: int, aspiration_labels: dict) -> tuple[np.ndarray, dict]:
    """Values (float, before missing codes are added) and value labels of one code."""
    z = rng.standard_normal(n)

    def cont(loc, scale, signal, noise, lo, hi, decimals=0):
        return np.clip(np.round(loc + scale * signal + noise * z, decimals), lo, hi)

    def ordinal(signal, cuts, weight=0.8, noise=0.6):
        s = weight * signal + noise * z
        return 1.0 + sum((s > c).astype(float) for c in cuts)

    A, S, B, Int, M = lat["A"], lat["S"], lat["B"], lat["I"], lat["M"]
    if full_name == "s0_mo_birthweight":
        return cont(118, 3, S, 16, 60, 200), {**NA_1, 999: "Not known"}
    if full_name == "s0_mo_class_mother_husband":
        return cont(3.2, -1.1, S, 0.8, 1, 6), {**CLASS_6, 8: "Unclassifiable", 9: "Too vague", **NA_1}
    if full_name == "s0_mo_class_mother_father":
        return cont(3.4, -0.9, S, 0.9, 1, 6), {**CLASS_6, 8: "Imprecise", 9: "Inapplicable", **NA_1}
    if full_name == "s0_mo_persons_per_room":
        return cont(1.3, -0.3, S, 0.35, 0.3, 4.0, decimals=1), {99: "Do not know", **NA_1}
    if full_name in ("s2_te_general_knowledge", "s2_te_number_work", "s2_te_use_of_books",
                     "s2_te_oral_ability"):
        # 1 = well above average: the rating falls as ability rises
        return 6.0 - ordinal(A, (-1.2, -0.4, 0.4, 1.2), weight=1.0, noise=0.5), {**RATING_5, 8: "Dont know", **NA_1}
    if full_name in ("s2_te_poor_hand_control", "s2_te_squirmy", "s2_te_poor_coordination",
                     "s2_te_hardly_ever_still", "s2_te_poor_speech", "s2_te_imperfect_english"):
        return ordinal(B, (0.6, 1.4)), {**APPLIES_3, 8: "Dont know", **NA_1}
    if full_name.startswith("s2_te_"):  # the twelve BSAG totals
        signal = Int if full_name in BSAG_INTERNAL else B
        return np.clip(rng.poisson(np.exp(1.0 + 0.5 * signal)), 0, 30).astype(float), {99: "Not known", -1: "Inapplicable"}
    if full_name in ("s2_co_verbal_ability", "s2_co_nonverbal_ability", "s2_co_mathematics_ability"):
        return cont(20, 6, A, 3, 0, 40), NA_1
    if full_name == "s2_co_reading_ability":
        return cont(17, 5, A, 3, 0, 35), NA_1
    if full_name in EXTERNALISING:
        return ordinal(B, (0.5, 1.3)), {**APPLIES_3, 8: "Dont know", -1: "No information"}
    if full_name in INTERNALISING:
        return ordinal(Int, (0.5, 1.3)), {**APPLIES_3, 8: "Dont know", -1: "No information"}
    if full_name in ("s3_co_school_waste_of_time", "s3_co_homework_a_bore",
                     "s3_co_never_take_work_seriously", "s3_co_not_like_schol"):
        # agreeing (1) goes with low motivation
        return ordinal(M, (-1.2, -0.4, 0.4, 1.2)), {**AGREE_5, 8: "Cant say", 7: " Cant say,inappl", **NA_1}
    if full_name == "s3_co_aspiration_first_job":
        codes = np.array(sorted(aspiration_labels), dtype=float)
        return rng.choice(codes, size=n), {**aspiration_labels, 98: "Dont know", 99: "Not answered", **NA_1}
    if full_name == "s5_co_highest_edu":
        return (np.clip(np.round(2.3 + 1.0 * A + 0.5 * S + 0.4 * M + 0.7 * z), 0, 5),
                {**QUAL_6, 97: "N/a: proxy/block not entered", 98: "Misrouted - incomplete interview", -9: "Refused", **NA_1})
    if full_name == "s0_co_sex":
        return np.where(lat["male"], 1.0, 2.0), {1: "Male", 2: "Female", -1: "Not known"}
    if full_name.startswith("s8_co_"):
        signal = {"s8_co_openness": A, "s8_co_conscientiousness": M}.get(full_name, rng.standard_normal(n))
        return cont(27, 3, signal, 6, 5, 50), {99: "Self completion qnaire not completed", -8: "Dont know",
                                                -9: "Refused", -1: "Item not applicable"}
    if full_name == "s3_co_height":
        return np.round(163 + 8 * lat["male"] + 2 * S + 6 * z, 1), {999: "Item not applicable", **NA_1}
    if full_name == "s2_pa_nssec_father":
        valid = np.array([1.1, 1.2, 2.0, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.0, 6.0, 7.1, 7.2, 7.3,
                          7.4, 8.1, 8.2, 9.1, 9.2, 10.0, 11.1, 11.2, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6,
                          12.7, 13.1, 13.2, 13.3, 13.4, 13.5])
        rank = np.clip(((1 - (np.tanh(0.8 * S + 0.6 * z) + 1) / 2) * len(valid)).astype(int), 0, len(valid) - 1)
        values = valid[rank]
        gap = rng.random(n) < 0.03  # values between the recode's ranges become NA
        values[gap] = rng.choice([3.0, 6.5, 14.0], size=int(gap.sum()))
        return values, {-1: "Inapplicable"}  # decimal codes carry no value labels
    if full_name in ("s3_pa_age_edu_left_father", "s3_pa_age_edu_left_mother"):
        return cont(5.0, 1.8, S, 1.5, 1, 12), {**AGE_LEFT, 98: "Do not know, DNA", 99: "Refused", **NA_1}
    if full_name == "s3_co_mathematics_ability":
        return cont(15, 5, A, 3, 0, 31), {99: "Not applicable", **NA_1}
    if full_name == "s3_co_reading_ability":
        return cont(18, 5, A, 3, 0, 35), {99: "Not applicable", **NA_1}
    raise KeyError(f"no synthetic generator for {full_name}")


def _write_dta(path: Path, frame: pd.DataFrame, labels: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_dta(frame, str(path), variable_value_labels={c: lab for c, lab in labels.items() if lab})


def write_synthetic_inputs(root, seed: int, n: int = 200, gene_data: bool = False,
                           essay_share: float = 0.9, missing_share: float = 0.12) -> dict:
    """Write a complete synthetic input set for ``n`` cohort members into ``root``.

    Returns a manifest: the IDs, the files written, and what was planted. With
    ``gene_data=True`` the optional polygenic score table is written too.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    ids = np.array([synthetic_id(i) for i in range(1, n + 1)])

    # --- latent traits (made up) ---
    A = rng.standard_normal(n)
    S = 0.3 * A + np.sqrt(1 - 0.3**2) * rng.standard_normal(n)
    M = 0.3 * A + np.sqrt(1 - 0.3**2) * rng.standard_normal(n)
    lat = {"A": A, "S": S, "M": M, "B": rng.standard_normal(n), "I": rng.standard_normal(n),
           "male": rng.random(n) < 0.5}

    # --- NCDS files: every code of the table, each in exactly one file ---
    table = read_datalist(str(config.VARIABLES_XLSX))
    table["full_name"] = ("s" + table["sweep"].astype(int).astype(str) + "_"
                          + table["respondent"].str[:2] + "_" + table["new_varname"])
    mapping = pd.read_excel(config.OCCUPATION_ASPIRATION_XLSX)
    aspiration_labels = dict(zip(mapping["aspiration_code"].astype(int), mapping["aspiration_n2771"]))

    columns: dict[str, dict[str, np.ndarray]] = {k: {} for k in NCDS_FILES}
    labels: dict[str, dict[str, dict]] = {k: {} for k in NCDS_FILES}
    missing_pool: dict[tuple[str, str], list[float]] = {}
    for row in table.itertuples():
        values, lab = _variable(row.full_name, rng, lat, n, aspiration_labels)
        key = _file_for(row.variable, row.sweep)
        columns[key][row.variable] = values.astype(float)
        labels[key][row.variable] = lab
        # the codes that become NA: negative ones (read_ncds) and those labelled with a
        # missing-value string (clean_ncds); -99 has no label
        missing_pool[(key, row.variable)] = [float(v) for v, t in lab.items()
                                             if v < 0 or t in MISSING_LABELS] + [-99.0]

    # n885 is in the real sweep-2 file but NOT in variables.xlsx, so read_ncds drops it
    # unless the corrected variant is on (config.INCLUDE_N885; PORTING_NOTES N2).
    values, lab = _variable("s2_te_imperfect_english", rng, lat, n, aspiration_labels)
    columns["ncds_1_2_3"]["N885"] = values.astype(float)
    labels["ncds_1_2_3"]["N885"] = lab
    missing_pool[("ncds_1_2_3", "N885")] = [float(v) for v, t in lab.items()
                                            if v < 0 or t in MISSING_LABELS] + [-99.0]

    # Missing values: a share of cohort members get one to three missing codes (a
    # negative code, or a positive code whose label is a missing-value string).
    hit_rows = rng.choice(n, size=int(round(missing_share * n)), replace=False)
    pool_keys = list(missing_pool)
    for i in hit_rows:
        for j in rng.choice(len(pool_keys), size=int(rng.integers(1, 4)), replace=False):
            key, code = pool_keys[j]
            columns[key][code][i] = rng.choice(missing_pool[(key, code)])

    written = []
    for key in NCDS_FILES:
        keep = np.sort(rng.choice(n, size=int(round(FILE_COVERAGE[key] * n)), replace=False))
        order = keep if key == "ncds_1_2_3" else rng.permutation(keep)
        id_col = "NCDSID" if key in UPPER_CASE_ID_FILES else "ncdsid"
        frame = pd.DataFrame({id_col: ids[order]})
        for code, values in columns[key].items():
            frame[code] = values[order]
        if not columns[key]:  # a file with no code of the table: columns read_ncds drops
            frame["syn_filler_a"] = rng.normal(size=len(order)).round(3)
            frame["syn_filler_b"] = rng.integers(1, 5, size=len(order)).astype(float)
        path = root / config.RESTRICTED_INPUTS[key]
        _write_dta(path, frame, labels[key])
        written.append(config.RESTRICTED_INPUTS[key])

    # --- essays ---
    has_essay = np.sort(rng.choice(n, size=int(round(essay_share * n)), replace=False))
    words = np.clip(np.round(150 + 35 * A[has_essay] + 30 * rng.standard_normal(len(has_essay))), 20, 400).astype(int)
    vocab = np.array(["synthetic", "essay", "word", "text", "sample", "token", "filler", "placeholder"])
    essay_dir = root / config.RESTRICTED_INPUTS["essays"]
    essay_dir.mkdir(parents=True, exist_ok=True)
    filenames = [f"essay_{k:04d}.txt" for k in range(1, len(has_essay) + 1)]
    for fname, i, w in zip(filenames, has_essay, words):
        text = " ".join(rng.choice(vocab, size=w))
        content = f"ID: {ids[i]}{ESSAY_ID_SEPARATOR}{text}{ESSAY_WORDS_SEPARATOR}{w}\n"
        (essay_dir / fname).write_text(content, encoding="utf-8")
    written.append(config.RESTRICTED_INPUTS["essays"])
    a_e = A[has_essay]
    m_e = len(has_essay)

    def noisy(weight):
        return np.round(weight * a_e + rng.standard_normal(m_e), 6)

    # --- SALAT: three tools x three batches; nwords is reported by TAALED and TAALES
    # (identical values), so it becomes a join key of the TAALES step ---
    batches = np.array_split(np.arange(m_e), 3)
    taaled = pd.DataFrame({"filename": filenames, "nwords": words,
                           **{f"taaled_metric_{k}": noisy(w) for k, w in ((1, 0.8), (2, 0.4), (3, 0.0))}})
    taales = pd.DataFrame({"Filename": filenames, "nwords": words,
                           **{f"taales_metric_{k}": noisy(w) for k, w in ((1, 0.7), (2, 0.3), (3, 0.0))}})
    seance = pd.DataFrame({"filename": filenames,
                           **{f"seance_metric_{k}": noisy(w) for k, w in ((1, 0.2), (2, 0.0), (3, 0.0))}})
    for tool, frame in (("taaled", taaled), ("taales", taales), ("seance", seance)):
        for b, idx in enumerate(batches, start=1):
            path = root / config.RESTRICTED_INPUTS[f"{tool}_{b}"]
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.iloc[idx].to_csv(path, index=False)
            written.append(config.RESTRICTED_INPUTS[f"{tool}_{b}"])

    # --- spelling: one row per error; some essays have none; all nine categories occur ---
    rate = np.maximum(words / 60.0, 0.3) * np.exp(-0.6 * a_e)
    n_err = rng.poisson(rate)
    n_err[: max(3, m_e // 20)] = 0  # essays without any error
    probs = np.array([0.25, 0.3, 0.15, 0.03, 0.05, 0.1, 0.05, 0.04, 0.03])
    rows = [(ids[i], rng.choice(SPELLING_CATEGORIES, p=probs)) for i, k in zip(has_essay, n_err) for _ in range(k)]
    with_errors = [ids[i] for i, k in zip(has_essay, n_err) if k > 0]
    for j, cat in enumerate(SPELLING_CATEGORIES):  # make sure every category occurs
        rows.append((with_errors[j % len(with_errors)], cat))
    spelling = pd.DataFrame(rows, columns=["ncdsid", "rule_issue_type"])
    spelling["rule_id"] = [f"SYN_RULE_{k}" for k in rng.integers(1, 40, size=len(spelling))]
    spelling.to_csv(root / config.RESTRICTED_INPUTS["spelling_mistakes"], index=False)
    written.append(config.RESTRICTED_INPUTS["spelling_mistakes"])

    # --- readability, in the ingestion format: filename, ncdsid, one column per index ---
    readability = pd.DataFrame({"filename": filenames, "ncdsid": ids[has_essay],
                                **{f"readability_index_{k}": noisy(w) for k, w in ((1, 0.6), (2, 0.5), (3, 0.2), (4, 0.0))}})
    readability.to_csv(root / config.RESTRICTED_INPUTS["readability_metrics"], index=False)
    written.append(config.RESTRICTED_INPUTS["readability_metrics"])

    # --- embeddings: GPT-3.5 and GPT-4 (ncdsid + embedding_k), RoBERTa (id + roberta_dim_k) ---
    def embedding(dims, signal_dims, weight):
        e = rng.standard_normal((m_e, dims))
        e[:, :signal_dims] += weight * a_e[:, None]
        return np.round(e, 6)

    for key, dims, sig in (("gpt35_embeddings", GPT35_DIMS, 4), ("gpt4_embeddings", GPT4_DIMS, 6)):
        frame = pd.DataFrame(embedding(dims, sig, 0.7), columns=[f"embedding_{k}" for k in range(1, dims + 1)])
        frame.insert(0, "ncdsid", ids[has_essay])
        path = root / config.RESTRICTED_INPUTS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written.append(config.RESTRICTED_INPUTS[key])
    roberta = pd.DataFrame(embedding(ROBERTA_DIMS, 8, 0.6), columns=[f"roberta_dim_{k}" for k in range(1, ROBERTA_DIMS + 1)])
    roberta.insert(0, "id", ids[has_essay])
    path = root / "derived" / config.DERIVED_FILES["roberta_embeddings"]
    path.parent.mkdir(parents=True, exist_ok=True)
    roberta.to_csv(path, index=False)
    written.append(f"derived/{config.DERIVED_FILES['roberta_embeddings']}")

    # --- optional polygenic scores (placeholder format: ncdsid + one column per score) ---
    if gene_data:
        genotyped = np.sort(rng.choice(n, size=int(round(0.8 * n)), replace=False))
        pgs = rng.standard_normal((len(genotyped), N_PGS))
        pgs[:, 0] += 0.6 * A[genotyped]
        frame = pd.DataFrame(np.round(pgs, 6), columns=[f"syn_pgs_{k}" for k in range(1, N_PGS + 1)])
        frame.insert(0, "ncdsid", ids[genotyped])
        path = root / config.RESTRICTED_INPUTS["gene_data"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written.append(config.RESTRICTED_INPUTS["gene_data"])

    manifest = {"synthetic": True, "generator": "tests/fixtures/synthetic_ncds.py", "seed": seed, "n": n,
                "gene_data": gene_data, "n_essays": int(m_e), "files": written}
    (root / config.SYNTHETIC_MARKER_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "ids": list(ids), "essay_ids": list(ids[has_essay]), "root": str(root)}
