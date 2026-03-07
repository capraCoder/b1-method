"""
B1 Method — Streamlit Web Interface
====================================

Browser-based B1 convergent derivation analysis.
Upload alignment + sources CSVs, get tier-classified report.

Also available as:
  - pip install b1-method (Python CLI)
  - Google Colab notebook (zero install)
  - GitHub: github.com/capraCoder/AIBOS_B1
"""

import io
import json
import math
import importlib.resources

import streamlit as st

from b1_method.core import B1Analysis, B1Result
from b1_method.io import load_alignment_csv, load_sources_csv

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="B1 Method — Convergent Basis Derivation",
    page_icon="📐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("B1 Method")
st.markdown(
    "**Domain-independent convergent derivation of canonical basis vectors "
    "from K independent sources.**"
)

st.caption(
    "Also available: "
    "[`pip install b1-method`](https://pypi.org/project/b1-method/) · "
    "[Google Colab](https://colab.research.google.com/) · "
    "[GitHub](https://github.com/capraCoder/b1-method)"
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: input mode
# ---------------------------------------------------------------------------

mode = st.sidebar.radio(
    "Input mode",
    ["Upload CSVs", "Domain examples"],
    index=1,
)

alignment = None
sources = None
domain = ""

if mode == "Upload CSVs":
    st.sidebar.markdown("### Upload files")

    align_file = st.sidebar.file_uploader(
        "Alignment CSV (required)",
        type=["csv"],
        help="Rows = candidates, columns = sources. Values: Y, Y*, N, N*",
    )
    src_file = st.sidebar.file_uploader(
        "Sources CSV (optional)",
        type=["csv"],
        help="Columns: name, method_type, languages, tradition, year",
    )
    domain = st.sidebar.text_input("Domain name", value="Analysis")

    if align_file is not None:
        # Save uploaded file temporarily
        align_path = "/tmp/b1_alignment.csv"
        with open(align_path, "wb") as f:
            f.write(align_file.getvalue())

        alignment, source_names = load_alignment_csv(align_path)

        if src_file is not None:
            src_path = "/tmp/b1_sources.csv"
            with open(src_path, "wb") as f:
                f.write(src_file.getvalue())
            sources = load_sources_csv(src_path)

else:
    example = st.sidebar.selectbox(
        "Domain",
        ["personality", "emotion", "brand", "intelligence"],
        format_func=str.title,
    )
    domain = example.title()

    examples_dir = importlib.resources.files("b1_method") / "examples"
    align_path = str(examples_dir / f"{example}_alignment.csv")
    src_path = str(examples_dir / f"{example}_sources.csv")

    alignment, source_names = load_alignment_csv(align_path)
    sources = load_sources_csv(src_path)

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if alignment is not None:
    result = B1Analysis(alignment, sources=sources, domain=domain).run()

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("K (sources)", result.k)
    col2.metric("Lower bound", result.lower_bound)
    col3.metric("Tier 1", len(result.tier1_candidates))
    col4.metric("Tier 2", len(result.tier2_candidates))

    st.divider()

    # --- Tier results ---
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Tier Classification")

        t1_thresh = math.ceil(2 * result.k / 3)
        t2_thresh = math.ceil(result.k / 3)
        st.caption(
            f"Thresholds: Tier 1 ≥ {t1_thresh} · Tier 2 ≥ {t2_thresh} · "
            f"Tier 3 < {t2_thresh}"
        )

        # Build table data
        rows = []
        sorted_candidates = sorted(
            result.tiers.items(),
            key=lambda item: (item[1]["tier"], -item[1]["count"], item[0]),
        )
        for candidate, info in sorted_candidates:
            alignment_str = " ".join(info["alignment"])
            rows.append({
                "Candidate": candidate,
                "Count": info["count"],
                "Tier": info["tier"],
                "Alignment": alignment_str,
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Summary")

        if result.tier1_candidates:
            st.markdown(
                f"**Tier 1** ({len(result.tier1_candidates)}): "
                f"{', '.join(result.tier1_candidates)}"
            )
        if result.tier2_candidates:
            st.markdown(
                f"**Tier 2** ({len(result.tier2_candidates)}): "
                f"{', '.join(result.tier2_candidates)}"
            )
        if result.tier3_candidates:
            st.markdown(
                f"**Tier 3** ({len(result.tier3_candidates)}): "
                f"{', '.join(result.tier3_candidates)}"
            )

        st.divider()

        # Warnings
        if result.warnings:
            st.subheader("Independence Warnings")
            for w in result.warnings:
                st.warning(w, icon="⚠️")
        else:
            st.success("Independence check passed — diverse sources.", icon="✅")

    st.divider()

    # --- Download JSON ---
    json_str = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    st.download_button(
        label="Download JSON report",
        data=json_str,
        file_name=f"b1_report_{domain.lower().replace(' ', '_')}.json",
        mime="application/json",
    )

else:
    st.info("Upload an alignment CSV or select a bundled example to begin.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "B1 Method v0.1.0 · Caprazli (2026) · "
    "[cosmologic.pro](https://cosmologic.pro)"
)
