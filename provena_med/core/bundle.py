"""Evidence-unit bundle construction for PROVENA-MED provenance.

Turns a case into citable evidence units with stable typed IDs:
  NOTE_SPAN:hpi:<i>        - a sentence span of the history of present illness
  TABLE_ROW:vital:<name>   - an initial vital sign
  TABLE_ROW:lab:<name>     - a parsed laboratory value
  IMAGE_FINDING:<study>:<f>- (multimodal cohort only; not built here)
The model is shown these units and must cite their IDs for each claim.
"""
from __future__ import annotations

import re

VITAL_CANON = {
    "temperature": "temperature", "temp": "temperature", "heartrate": "heart_rate",
    "hr": "heart_rate", "resprate": "resp_rate", "rr": "resp_rate", "o2sat": "o2_sat",
    "spo2": "o2_sat", "sbp": "sbp", "dbp": "dbp",
}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.;])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 12]


def parse_vitals(s: str) -> list[tuple[str, str]]:
    out, seen = [], set()
    for name, val in re.findall(r"([A-Za-z][A-Za-z0-9]+)\s*[:=]\s*([\d.]+)", str(s)):
        key = VITAL_CANON.get(name.lower())
        if key and key not in seen:
            seen.add(key)
            out.append((key, val))
    return out


def parse_labs(s: str) -> list[tuple[str, str]]:
    """MIMIC lab strings look like 'wbc-7.1 hgb-9.3* creat-0.8 na-139'."""
    out, seen = [], set()
    for name, val in re.findall(r"\b([a-zA-Z]{2,}[a-zA-Z]*)-(\d+\.?\d*)\*?", str(s)):
        key = name.lower()
        if key not in seen and not key.isdigit():
            seen.add(key)
            out.append((key, val))
    return out


_RAD_SECTION = re.compile(r"(FINDINGS|IMPRESSION)\s*:\s*(.*?)(?=\n[A-Z][A-Z /]{2,}:|\Z)", re.S)


def extract_image_findings(report: str, study_id: str = "cxr") -> list[dict]:
    """IMAGE_FINDING units = radiologist-authored finding/impression sentences."""
    units: list[dict] = []
    for _, body in _RAD_SECTION.findall(str(report)):
        for sent in split_sentences(body):
            units.append({"id": f"IMAGE_FINDING:{study_id}:{len(units)}",
                          "type": "IMAGE_FINDING", "text": sent})
    return units


IMG_FIELDS = {"X-ray": "xray", "CT": "ct", "Ultrasound": "us", "MRI": "mri",
              "ECG": "ecg", "CATH": "cath"}


def _parse_field(v) -> str:
    """Cardiac-cohort imaging fields are Python-list-like strings; flatten to text."""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    if s[0] in "[(":
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return " ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass
    return s


def build_bundle_mm(case: dict) -> list[dict]:
    """Multimodal bundle: NOTE_SPAN (HPI + exam) + TABLE_ROW (recovered labs) +
    IMAGE_FINDING (imaging fields)."""
    units: list[dict] = []
    for i, s in enumerate(split_sentences(case.get("HPI", ""))):
        units.append({"id": f"NOTE_SPAN:hpi:{i}", "type": "NOTE_SPAN", "text": s})
    for i, s in enumerate(split_sentences(case.get("physical_exam", ""))):
        units.append({"id": f"NOTE_SPAN:exam:{i}", "type": "NOTE_SPAN", "text": s})
    units += _med_rows(case)
    units += _lab_rows(window_labs(case))
    for col, tag in IMG_FIELDS.items():
        for j, s in enumerate(split_sentences(_parse_field(case.get(col, "")))):
            units.append({"id": f"IMAGE_FINDING:{tag}:{j}", "type": "IMAGE_FINDING", "text": s})
    return units


# default decision-time window (hours from the stay anchor); set hi=None for full stay.
WINDOW_LO, WINDOW_HI = -12.0, 24.0


def window_labs(case: dict, lo: float = WINDOW_LO, hi: float | None = WINDOW_HI) -> dict:
    """Lab values visible within [lo, hi] hours, from the timestamped `labs_ts` if present
    (window is a filter, not baked in); falls back to the static `labs` dict."""
    ts = case.get("labs_ts")
    if not ts:
        return case.get("labs") or {}
    out = {}
    for name, pts in ts.items():
        for off, val, flag in pts:  # pts sorted by offset; take first within window
            if off >= lo and (hi is None or off <= hi):
                out[name] = {"value": val, "uom": "", "flag": flag}
                break
    return out


def _med_rows(case: dict) -> list[dict]:
    """Home/pre-admission medications as citable NOTE_SPAN units (leakage-safe context)."""
    return [{"id": f"NOTE_SPAN:med:{i}", "type": "NOTE_SPAN", "text": f"Home medication: {m}"}
            for i, m in enumerate(case.get("home_meds") or [])]


def _lab_rows(labs: dict) -> list[dict]:
    units = []
    for name, rec in (labs or {}).items():
        flag = f" ({rec['flag']})" if isinstance(rec, dict) and rec.get("flag") else ""
        val = rec.get("value") if isinstance(rec, dict) else rec
        uom = rec.get("uom", "") if isinstance(rec, dict) else ""
        units.append({"id": f"TABLE_ROW:lab:{name}", "type": "TABLE_ROW",
                      "text": f"{name} = {val} {uom}{flag}".strip()})
    return units


def build_bundle_icu(case: dict, image_findings: list[str] | None = None) -> list[dict]:
    """Multimodal ICU bundle: NOTE_SPAN (HPI + exam) + TABLE_ROW (first-24h vitals + labs)
    + IMAGE_FINDING units. By default the IMAGE_FINDING units are the RADIOLOGIST report
    findings stored on the case (ground truth); pass image_findings to override (e.g. a
    vision model's reading, for a VLM baseline). DICOM pixels remain at case['dicom_path']."""
    if image_findings is None:
        image_findings = case.get("image_findings") or []
    units: list[dict] = []
    for i, s in enumerate(split_sentences(case.get("HPI", ""))):
        units.append({"id": f"NOTE_SPAN:hpi:{i}", "type": "NOTE_SPAN", "text": s})
    for i, s in enumerate(split_sentences(case.get("physical_exam", ""))):
        units.append({"id": f"NOTE_SPAN:exam:{i}", "type": "NOTE_SPAN", "text": s})
    for name, val in (case.get("vitals") or {}).items():
        units.append({"id": f"TABLE_ROW:vital:{name}", "type": "TABLE_ROW",
                      "text": f"{name.replace('_', ' ')} = {val}"})
    units += _med_rows(case)
    units += _lab_rows(window_labs(case))
    for j, s in enumerate(image_findings):
        units.append({"id": f"IMAGE_FINDING:cxr:{j}", "type": "IMAGE_FINDING",
                      "text": str(s).strip()})
    return units


def build_bundle_eicu(case: dict) -> list[dict]:
    """eICU structured external bundle: NOTE_SPAN (past history + exam findings) +
    TABLE_ROW (vitals + labs). No imaging in eICU."""
    units: list[dict] = []
    for i, s in enumerate(case.get("past_history") or []):
        units.append({"id": f"NOTE_SPAN:hx:{i}", "type": "NOTE_SPAN",
                      "text": f"Past history: {str(s).strip()}"})
    for i, s in enumerate(case.get("physical_exam") or []):
        units.append({"id": f"NOTE_SPAN:exam:{i}", "type": "NOTE_SPAN",
                      "text": f"Exam: {str(s).strip()}"})
    for name, val in (case.get("vitals") or {}).items():
        units.append({"id": f"TABLE_ROW:vital:{name}", "type": "TABLE_ROW",
                      "text": f"{name.replace('_', ' ')} = {val}"})
    units += _med_rows(case)
    units += _lab_rows(window_labs(case))
    return units


def build_bundle_mimic3(case: dict) -> list[dict]:
    """MIMIC-III external bundle: NOTE_SPAN (HPI + exam) + TABLE_ROW (labs) +
    IMAGE_FINDING (radiology-report findings; report text, no pixels)."""
    units: list[dict] = []
    for i, s in enumerate(split_sentences(case.get("HPI", ""))):
        units.append({"id": f"NOTE_SPAN:hpi:{i}", "type": "NOTE_SPAN", "text": s})
    for i, s in enumerate(split_sentences(case.get("physical_exam", ""))):
        units.append({"id": f"NOTE_SPAN:exam:{i}", "type": "NOTE_SPAN", "text": s})
    units += _med_rows(case)
    units += _lab_rows(window_labs(case))
    for j, s in enumerate(case.get("image_findings") or []):
        units.append({"id": f"IMAGE_FINDING:cxr:{j}", "type": "IMAGE_FINDING", "text": str(s).strip()})
    return units


def build_bundle_cxr_pixel(case: dict, image_findings: list[str]) -> list[dict]:
    """True-pixel bundle: NOTE_SPAN (HPI + exam) + IMAGE_FINDING units that a vision
    model read off the actual radiograph pixels (not the radiologist's report text)."""
    units: list[dict] = []
    for i, s in enumerate(split_sentences(case.get("HPI", ""))):
        units.append({"id": f"NOTE_SPAN:hpi:{i}", "type": "NOTE_SPAN", "text": s})
    for i, s in enumerate(split_sentences(case.get("physical_exam", ""))):
        units.append({"id": f"NOTE_SPAN:exam:{i}", "type": "NOTE_SPAN", "text": s})
    for j, s in enumerate(image_findings):
        units.append({"id": f"IMAGE_FINDING:cxr:{j}", "type": "IMAGE_FINDING",
                      "text": str(s).strip()})
    return units


def build_bundle(case: dict, max_labs: int = 25) -> list[dict]:
    units: list[dict] = []
    for i, sent in enumerate(split_sentences(case.get("HPI", ""))):
        units.append({"id": f"NOTE_SPAN:hpi:{i}", "type": "NOTE_SPAN", "text": sent})
    units += _med_rows(case)
    for name, val in parse_vitals(case.get("initial_vitals", "")):
        units.append({"id": f"TABLE_ROW:vital:{name}", "type": "TABLE_ROW",
                      "text": f"{name.replace('_', ' ')} = {val}"})
    if case.get("labs_ts") or case.get("labs"):  # recovered structured labs (timestamped)
        units += _lab_rows(window_labs(case))
    else:  # legacy: parse from the ED 'tests' free text
        for name, val in parse_labs(case.get("tests", ""))[:max_labs]:
            units.append({"id": f"TABLE_ROW:lab:{name}", "type": "TABLE_ROW", "text": f"{name} = {val}"})
    return units


def bundle_to_prompt(units: list[dict]) -> str:
    lines = [f"[{u['id']}] {u['text']}" for u in units]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from run_generate_safety import load_safety_cases

    df = load_safety_cases(3, seed=1)
    for _, row in df.iterrows():
        b = build_bundle(row.to_dict())
        print("=" * 70, "\nstay", row["stay_id"], "->", len(b), "evidence units")
        for u in b[:12]:
            print(f"   [{u['id']}] {u['text'][:70]}")
        print("   gold:", row["primary_diagnosis"][:120])
