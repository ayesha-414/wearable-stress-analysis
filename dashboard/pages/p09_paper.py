"""Page 9: Research Paper & Verified Bibliography Viewer.

Presents:
- Full in-app reader for the IEEE-style research paper (docs/paper/NeuroTwin_paper.md)
- Downloadable research paper (Markdown & compiled PDF)
- Complete Verified References library with clickable DOIs, verification level badges, and 'Used For' tags
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.theme import evidence_badge


def render_page_research_paper():
    """Render the Research Paper and Bibliography viewer."""
    st.markdown("## 📄 Research Paper & Verified Bibliography")
    st.markdown(
        """
        Read the complete IEEE-style scientific paper detailing the theoretical foundations,
        mathematical formulas, leave-one-subject-out empirical findings, and ethical governance of NeuroTwin AI.
        """
    )

    paper_path = Path("docs/paper/NeuroTwin_paper.md")
    if not paper_path.is_file():
        paper_path = Path(__file__).resolve().parents[2] / "docs" / "paper" / "NeuroTwin_paper.md"
    refs_path = Path("docs/neurotwin/references_verified.csv")
    if not refs_path.is_file():
        refs_path = Path(__file__).resolve().parents[2] / "docs" / "neurotwin" / "references_verified.csv"

    tab_paper, tab_refs = st.tabs(["📖 Read Research Paper", "📚 Verified References Library"])

    # --- TAB 1: In-App Paper Reader ---
    with tab_paper:
        if paper_path.is_file():
            paper_text = paper_path.read_text(encoding="utf-8")

            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                st.download_button(
                    label="📥 Download Paper as Markdown (.md)",
                    data=paper_text.encode("utf-8"),
                    file_name="NeuroTwin_IEEE_Paper.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_d2:
                # Also allow downloading as formatted text / document
                st.download_button(
                    label="📄 Download Paper Text (.txt)",
                    data=paper_text.encode("utf-8"),
                    file_name="NeuroTwin_Paper.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.markdown("<div class='waveform-divider'></div>", unsafe_allow_html=True)

            # Render full markdown inside an styled container
            st.markdown(paper_text)
        else:
            st.error("Paper file docs/paper/NeuroTwin_paper.md not found.")

    # --- TAB 2: Verified References Library ---
    with tab_refs:
        st.markdown("### 📚 Verified References & DOI Registry")
        st.markdown(
            """
            In accordance with open-science standards, every reference is cataloged with its exact verification level:
            - **Level A:** Primary publication or DOI record verified directly.
            - **Level B:** Secondary verification via peer-reviewed review articles.
            - **Level C:** Pre-print or non-peer-reviewed source (flagged with caution).
            """
        )

        if refs_path.is_file():
            refs_df = pd.read_csv(refs_path)

            search_query = st.text_input("🔍 Search references by author, keyword, or topic:", "")
            if search_query:
                filtered_df = refs_df[
                    refs_df["citation"].str.contains(search_query, case=False, na=False)
                    | refs_df["key"].str.contains(search_query, case=False, na=False)
                    | refs_df["used_for"].str.contains(search_query, case=False, na=False)
                ]
            else:
                filtered_df = refs_df

            st.markdown(f"**Showing {len(filtered_df)} of {len(refs_df)} verified entries**")

            for _, row in filtered_df.iterrows():
                level = str(row.get("verification_level", "A")).strip()
                lvl_class = "nt-badge-standard" if level == "A" else ("nt-badge-heuristic" if level == "B" else "nt-badge-uncertain")
                doi = str(row.get("doi_or_url", ""))
                doi_link = f"https://doi.org/{doi.replace('doi:', '')}" if doi.startswith("doi:") else doi

                st.markdown(
                    f"""
                    <div class="nt-card" style="padding: 1rem 1.2rem; margin-bottom: 0.8rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <span style="font-weight:600; color:#38bdf8; font-size:0.95rem;">[{row.get('key')}]</span>
                            <span class="nt-badge {lvl_class}">Level {level}</span>
                        </div>
                        <div style="font-size:0.87rem; color:#f1f5f9; margin-bottom:0.4rem;">
                            {row.get('citation')}
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.3rem;">
                            <b>Used for:</b> {row.get('used_for', 'General scientific grounding')}
                        </div>
                        <div style="font-size:0.78rem;">
                            🔗 <a href="{doi_link}" target="_blank" style="color:#00f2fe; text-decoration:none;">{doi}</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error("References file docs/neurotwin/references_verified.csv not found.")

