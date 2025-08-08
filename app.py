import os
import streamlit as st
import pandas as pd
from pathlib import Path

from engine import (
    store_resume_embedding,
    rank_resumes,
    final_output_with_summaries,
    clear_index,
)

st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")
st.title("Candidate Recommendation Engine")

# --- Inputs ---
st.subheader("Job Description")
jd_text = st.text_area("Paste the job description here...", height=240, placeholder="Paste the job description here...")
uploaded = st.file_uploader("Upload resumes (PDF, DOCX, TXT). Multiple allowed.", type=["pdf","docx","txt"], accept_multiple_files=True)

colA, _ = st.columns([1, 2])
with colA:
    top_n = st.number_input("Top N", min_value=1, max_value=20, value=5, step=1)
run_btn = st.button("Rank & Summarize", type="primary")

# --- Run pipeline ---
if run_btn:
    if not jd_text.strip():
        st.error("Please paste the job description.")
        st.stop()
    if not uploaded:
        st.error("Please upload at least one resume.")
        st.stop()

    try:
        clear_index()
        for f in uploaded:
            store_resume_embedding(f.name, f.getvalue(), filename=f.name)

        rank_df = rank_resumes(jd_text, top_n=top_n)
        final_df = final_output_with_summaries(rank_df, jd_text)

        final_df = final_df[["resume_id", "score", "summary"]]

        st.success("Done.")
        st.dataframe(final_df, use_container_width=True)

        st.download_button(
            "Download Results (CSV)",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_candidates.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
