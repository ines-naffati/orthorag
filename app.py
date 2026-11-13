"""Streamlit demo: search cleared orthopedic devices and draft 510(k) summary sections.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from orthorag import config
from orthorag.generate import TARGET_SECTIONS, draft
from orthorag.index import Retriever

st.set_page_config(page_title="orthorag", page_icon="🦴", layout="wide")


@st.cache_resource
def get_retriever(model: str) -> Retriever:
    return Retriever(model)


EXAMPLE = (
    "The XYZ Clavicle Plating System consists of pre-contoured titanium alloy (Ti-6Al-4V, ASTM F136) "
    "superior and anterior clavicle plates in left and right configurations, with 3.5 mm locking and "
    "non-locking cortical screws. The plates are provided sterile and non-sterile. The system is "
    "intended for fixation of fractures, malunions and nonunions of the clavicle."
)

st.title("orthorag")
st.caption("Grounded drafting of FDA 510(k) summaries for orthopedic implants · public FDA data · local models")

with st.sidebar:
    model = st.selectbox("Embedding model", config.EMBED_MODELS,
                         index=config.EMBED_MODELS.index(config.DEFAULT_EMBED_MODEL))
    mode = st.radio("Retrieval", ["hybrid", "dense", "bm25"], horizontal=True)
    st.markdown(f"LLM: `{config.LLM_MODEL}` via Ollama")

description = st.text_area("Describe your new device", EXAMPLE, height=150)
retriever = get_retriever(model)

tab_search, tab_draft = st.tabs(["Similar cleared devices", "Draft summary sections"])

with tab_search:
    if description.strip():
        hits = retriever.search(description, k=30, mode=mode)
        seen = set()
        for h in hits:
            if h.k_number in seen:
                continue
            seen.add(h.k_number)
            url = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={h.k_number}"
            with st.expander(f"{h.k_number} · {h.device_name} · best match in *{h.section}*"):
                st.write(h.raw)
                st.markdown(f"[FDA record]({url})")
            if len(seen) == 8:
                break

with tab_draft:
    sections = st.multiselect("Sections", list(TARGET_SECTIONS), default=list(TARGET_SECTIONS),
                              format_func=TARGET_SECTIONS.get)
    if st.button("Draft", type="primary") and description.strip():
        with st.spinner("Retrieving sources and drafting with the local LLM…"):
            drafts = draft(description, retriever, sections)
        for d in drafts:
            st.subheader(TARGET_SECTIONS[d.section])
            st.write(d.text)
            if d.invalid_citations:
                st.error(f"Citations not found in the retrieved sources: {', '.join(d.invalid_citations)}")
            else:
                st.success(f"All {len(d.cited)} citations point to retrieved sources.")
            with st.expander("Sources"):
                for h in d.sources:
                    st.markdown(f"**{h.k_number}** · {h.device_name} · *{h.section}*")
                    st.caption(h.raw[:600])
