"""Placeholder Streamlit review surface wired to future pair-label artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="TriAlign review", layout="wide")
st.title("TriAlign pair review")
st.info("Pair labeling activates after the reviewed criterion corpus and candidates are built.")

default_path = Path("data/processed/pair_candidates.parquet")
path = Path(st.text_input("Pair candidate file", str(default_path)))
if path.is_file():
    pairs = pd.read_parquet(path)
    st.metric("Candidate pairs", len(pairs))
    st.dataframe(pairs, use_container_width=True)
else:
    st.warning(f"No pair candidate file found at {path}.")

