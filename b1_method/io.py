"""
b1_method.io — CSV loading and JSON export for the B1 method.

Handles three input formats:
  1. Separate alignment CSV + sources CSV
  2. Combined single-file format (metadata + alignment in one CSV)
  3. JSON export of B1Result objects

All functions use stdlib only (csv, json, os).
"""

import csv
import json
import os


def load_alignment_csv(path):
    """Load an alignment matrix from CSV.

    Expected format:
        candidate,S1_Goldberg_1990,S2_Costa_McCrae_1992,...
        Extraversion,Y,Y,...
        Agreeableness,Y,Y*,...

    First column is the candidate construct name. Remaining columns are
    source names. Cell values are one of: Y, Y*, N, N*.

    Parameters
    ----------
    path : str
        Path to the alignment CSV file.

    Returns
    -------
    tuple[dict, list[str]]
        (alignment_dict, source_names) where alignment_dict maps each
        candidate name to a list of alignment values (Y/Y*/N/N*), and
        source_names is the ordered list of source column headers.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CSV has fewer than two columns or contains invalid values.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Alignment CSV not found: {path}")

    alignment = {}
    source_names = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(
                f"Alignment CSV must have at least 2 columns "
                f"(candidate + 1 source). Got: {header}"
            )

        source_names = [h.strip() for h in header[1:]]

        for line_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue  # skip blank rows

            candidate = row[0].strip()
            if not candidate:
                continue

            values = [v.strip() for v in row[1:]]

            # Pad with "N" if row is shorter than header
            while len(values) < len(source_names):
                values.append("N")

            # Validate values
            valid = {"Y", "Y*", "N", "N*"}
            for i, v in enumerate(values):
                if v not in valid:
                    raise ValueError(
                        f"Invalid alignment value '{v}' at line {line_num}, "
                        f"column '{source_names[i]}'. "
                        f"Expected one of: {', '.join(sorted(valid))}"
                    )

            alignment[candidate] = values

    if not alignment:
        raise ValueError(f"No data rows found in {path}")

    return alignment, source_names


def load_sources_csv(path):
    """Load source metadata from CSV.

    Expected format:
        name,method_type,languages,tradition,year
        S1_Goldberg_1990,lexical FA,English,lexical,1990
        S3_Ashton_Lee_2004,lexical FA,English;French;German,cross-linguistic,2004

    The ``languages`` column is semicolon-separated (e.g. "English;French").

    Parameters
    ----------
    path : str
        Path to the sources CSV file.

    Returns
    -------
    list[dict]
        Each dict has keys: name, method_type, languages (list of str),
        tradition, year (int).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sources CSV not found: {path}")

    required_cols = {"name", "method_type", "languages", "tradition", "year"}
    sources = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Sources CSV is empty: {path}")

        fields = {fn.strip().lower() for fn in reader.fieldnames}
        missing = required_cols - fields
        if missing:
            raise ValueError(
                f"Sources CSV missing required columns: {', '.join(sorted(missing))}. "
                f"Found: {', '.join(sorted(fields))}"
            )

        # Build a mapping from lower-case to original field names
        field_map = {fn.strip().lower(): fn for fn in reader.fieldnames}

        for row in reader:
            name = row[field_map["name"]].strip()
            if not name:
                continue

            langs_raw = row[field_map["languages"]].strip()
            languages = [lang.strip() for lang in langs_raw.split(";")
                         if lang.strip()]

            year_raw = row[field_map["year"]].strip()
            try:
                year = int(year_raw)
            except ValueError:
                raise ValueError(
                    f"Invalid year '{year_raw}' for source '{name}'. "
                    f"Expected integer."
                )

            sources.append({
                "name": name,
                "method_type": row[field_map["method_type"]].strip(),
                "languages": languages,
                "tradition": row[field_map["tradition"]].strip(),
                "year": year,
            })

    return sources


def load_combined_csv(path):
    """Load a combined single-file format containing both metadata and alignment.

    Expected format:
        # B1 Combined Input File
        # Domain: Personality Structure
        <blank line>
        name,method_type,languages,tradition,year
        S1_Goldberg_1990,lexical FA,English,lexical,1990
        S2_Costa_McCrae_1992,questionnaire,English,questionnaire,1992
        <blank line>
        candidate,S1_Goldberg_1990,S2_Costa_McCrae_1992
        Extraversion,Y,Y
        Agreeableness,Y,Y*

    Lines starting with ``#`` are metadata comments (ignored).
    Sections are separated by blank lines.

    Parameters
    ----------
    path : str
        Path to the combined CSV file.

    Returns
    -------
    tuple[dict, list[dict]]
        (alignment_dict, sources_list) — same formats as returned by
        ``load_alignment_csv`` and ``load_sources_csv`` respectively.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed into two sections.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Combined CSV not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    # Strip comment lines and collect content lines grouped by blank separators
    sections = []
    current_section = []

    for line in raw_lines:
        stripped = line.strip()

        # Skip comment lines
        if stripped.startswith("#"):
            continue

        if stripped == "":
            # Blank line: end current section if non-empty
            if current_section:
                sections.append(current_section)
                current_section = []
        else:
            current_section.append(stripped)

    # Don't forget the last section
    if current_section:
        sections.append(current_section)

    if len(sections) < 2:
        raise ValueError(
            f"Combined CSV must have at least 2 sections "
            f"(sources + alignment) separated by blank lines. "
            f"Found {len(sections)} section(s) in {path}"
        )

    # Section 1: source metadata
    sources_lines = sections[0]
    sources = _parse_sources_section(sources_lines)

    # Section 2: alignment matrix
    alignment_lines = sections[1]
    alignment, source_names = _parse_alignment_section(alignment_lines)

    return alignment, sources


def _parse_sources_section(lines):
    """Parse source metadata from a list of CSV-formatted lines."""
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError("Sources section is empty")

    field_map = {fn.strip().lower(): fn for fn in reader.fieldnames}
    sources = []

    for row in reader:
        name = row[field_map.get("name", "name")].strip()
        if not name:
            continue

        langs_raw = row.get(field_map.get("languages", "languages"), "")
        if langs_raw is None:
            langs_raw = ""
        languages = [lang.strip() for lang in langs_raw.strip().split(";")
                     if lang.strip()]

        year_raw = row.get(field_map.get("year", "year"), "0")
        if year_raw is None:
            year_raw = "0"
        try:
            year = int(year_raw.strip())
        except ValueError:
            year = 0

        sources.append({
            "name": name,
            "method_type": row.get(
                field_map.get("method_type", "method_type"), ""
            ).strip(),
            "languages": languages,
            "tradition": row.get(
                field_map.get("tradition", "tradition"), ""
            ).strip(),
            "year": year,
        })

    return sources


def _parse_alignment_section(lines):
    """Parse alignment matrix from a list of CSV-formatted lines."""
    reader = csv.reader(lines)
    header = next(reader, None)
    if header is None or len(header) < 2:
        raise ValueError("Alignment section must have at least 2 columns")

    source_names = [h.strip() for h in header[1:]]
    alignment = {}

    for row in reader:
        if not row or all(cell.strip() == "" for cell in row):
            continue
        candidate = row[0].strip()
        if not candidate:
            continue
        values = [v.strip() for v in row[1:]]
        while len(values) < len(source_names):
            values.append("N")
        alignment[candidate] = values

    return alignment, source_names


def save_report_json(result, path):
    """Save a B1Result (or any dict-like object) as JSON.

    If ``result`` has a ``to_dict`` method, it is called first. Otherwise
    the object is serialised directly. Non-serialisable values are
    converted to their string representation.

    Parameters
    ----------
    result : object
        A B1Result instance (with ``to_dict()``) or a plain dict.
    path : str
        Output file path. Parent directory is created if it does not exist.
    """
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    if hasattr(result, "to_dict"):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = result
    else:
        # Last resort: convert to dict via __dict__ if available
        data = getattr(result, "__dict__", {"result": str(result)})

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
