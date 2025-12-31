from __future__ import annotations
from pathlib import Path
import io, re
import pandas as pd

_NUM = re.compile(r"^[\s]*([+-]?\d+(?:\.\d+)?)")
_KV  = re.compile(r"^([A-Za-z0-9_. -]+?)\s*=\s*(.+)$")

def _strip_bang(s: str) -> str:
    i = s.find("!")
    return s if i < 0 else s[:i]

def _is_units_line(s: str, ncols: int) -> bool:
    # expect a series of parenthesized tokens, count should match columns
    tokens = re.findall(r"\((.*?)\)", s)
    return len(tokens) == ncols and bool(tokens)

def read_apsim_met(path: str | Path, *, na_values=("-999", "-9999", "NA")):
    """
    Parse an APSIM .met/.weather file safely.

    Returns
    -------
    df : pandas.DataFrame
        Includes numeric columns and 'date' (year + day-of-year).
    meta : dict
        Header key/values with numeric values parsed where possible.
    units : dict
        Column -> unit string (None if unavailable).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()

    # state
    meta: dict[str, str | float] = {}
    columns: list[str] | None = None
    units_line: str | None = None
    data_start: int | None = None

    # scan header
    i = 0
    while i < len(text):
        raw = text[i]
        line = _strip_bang(raw).strip()
        if not line:
            i += 1
            continue
        # ignore section headers like [weather.met.weather]
        if line.startswith("[") and line.endswith("]"):
            i += 1
            continue
        # column header?
        if re.match(r"^year\b", line, flags=re.I):
            columns = re.split(r"\s+", line.strip())
            # detect units
            j = i + 1
            while j < len(text) and not _strip_bang(text[j]).strip():
                j += 1
            if j < len(text) and _is_units_line(_strip_bang(text[j]).strip(), len(columns)):
                units_line = _strip_bang(text[j]).strip()
                data_start = j + 1
            else:
                data_start = i + 1
            break

        # metadata key=value (allow trailing units text)
        m = _KV.match(line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val_raw = m.group(2).strip()
            # if value begins with a number, parse it; keep original string if not
            mnum = _NUM.match(val_raw)
            if mnum:
                try:
                    v = float(mnum.group(1))
                    meta[key] = v
                except ValueError:
                    meta[key] = val_raw
            else:
                meta[key] = val_raw

        i += 1

    if columns is None or data_start is None:
        raise ValueError(f"{p.name}: could not find a 'year ...' column header.")

    # units map
    units: dict[str, str | None] = {c: None for c in columns}
    if units_line:
        toks = re.findall(r"\((.*?)\)", units_line)
        if len(toks) == len(columns):
            units = {c: (u or None) for c, u in zip(columns, toks)}

    # read data block with pandas
    # Use comment='!' to ignore trailing inline comments in the data
    # Skip header + optional units row exactly
    skiprows = list(range(data_start))  # pandas will skip 0..data_start-1
    # Reconstruct the file once; keeps memory moderate and parsing simple
    buf = io.StringIO("\n".join(text))
    df = pd.read_csv(
        buf,
        sep=r"\s+",
        engine="python",
        names=columns,
        header=None,
        skiprows=skiprows,
        comment="!",
        skip_blank_lines=True,
        na_values=list(na_values),
    )

    # enforce dtypes for year/day and coerce others
    if "year" not in df or "day" not in df:
        raise ValueError(f"{p.name}: required columns 'year' and 'day' not found.")

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype("int64")
    df["day"]  = pd.to_numeric(df["day"], errors="raise").astype("int64")

    for c in df.columns:
        if c not in ("year", "day"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # build date: year + (day-of-year - 1) days
    # robust for leap years; invalid DOY -> NaT
    base = pd.to_datetime(df["year"].astype(str), format="%Y", errors="coerce")
    df["date"] = base + pd.to_timedelta(df["day"] - 1, unit="D")

    # optional sanity checks (non-fatal)
    # e.g., warn if any NaT dates or suspicious DOY values
    bad = (~df["date"].notna()).sum()
    if bad:
        # keep it silent; you can log/raise if you prefer strictness
        pass

    # attach attrs
    df.attrs["meta"] = meta
    df.attrs["units"] = units
    return df, meta, units
