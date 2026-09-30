import os
import streamlit as st

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

import rag_core
from rag_core import CONFIG, Index, answer

CONFIG["DEFAULT_MODE"] = "hybrid_rerank"
CONFIG["RERANK_THRESHOLD"] = 0.004  
st.set_page_config(page_title="Lois numériques du Bénin", page_icon="⚖️")
st.title("Assistant juridique : lois numériques du Bénin")
st.caption("Réponses fondées uniquement sur le Code du numérique et le Code pénal. "
           "Ne saisissez pas de données personnelles sensibles.")


@st.cache_resource(show_spinner="Chargement de l'index et des modèles…")
def load():
    idx = Index.load("index")
    rag_core.get_embedder()
    rag_core.get_cross_encoder()
    return idx


idx = load()

if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for s in m.get("sources", []):
            with st.expander(f"{s['label']} (score {s['score']:.2f})"):
                st.write(s["text"])

q = st.chat_input("Posez votre question…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les textes…"):
            # multi-query désactivé : il dégradait le retrieval à l'évaluation et consomme du quota
            r = answer(idx, q, multiquery=False)
        st.markdown(r["answer"])
        for s in r["sources"]:
            with st.expander(f"{s['label']} (score {s['score']:.2f})"):
                st.write(s["text"])
    st.session_state.history.append(
        {"role": "assistant", "content": r["answer"], "sources": r["sources"]})