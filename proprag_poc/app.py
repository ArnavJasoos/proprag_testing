"""Streamlit GUI: ChatGPT-style sessions, PDF library/indexing, chat, graph view.

Run from the repo root:
    streamlit run proprag_poc/app.py
"""

from __future__ import annotations

import os
import sys

# Allow `streamlit run proprag_poc/app.py` (repo root on path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from proprag_poc.config import POCConfig  # noqa: E402
from proprag_poc.core.index import Engine  # noqa: E402
from proprag_poc.core.session import Message, SessionStore  # noqa: E402
from proprag_poc.logging_setup import setup_logging  # noqa: E402
from proprag_poc.viz.graph_view import ego_graph_html, find_entity_key, list_entities  # noqa: E402

setup_logging()  # progress -> the terminal running `streamlit run`
st.set_page_config(page_title="PropRAG-PDF POC", layout="wide")


@st.cache_resource
def get_engine() -> Engine:
    return Engine(POCConfig())


@st.cache_resource
def get_sessions() -> SessionStore:
    return SessionStore(POCConfig().data_dir)


def corpora_dir() -> str:
    return os.path.join(POCConfig().data_dir, "corpora")


def safe_filename(name: str, index: int) -> str:
    """Sanitize + truncate an uploaded filename to survive Windows MAX_PATH (260).

    Long PDF names (e.g. multi-title z-library dumps) overflow the path limit and
    make ``open()`` raise FileNotFoundError. Keep the extension, strip non-ASCII /
    illegal chars, truncate the stem, and prefix an index to stay unique.
    """
    import re

    stem, ext = os.path.splitext(os.path.basename(name))
    stem = re.sub(r"[^A-Za-z0-9 ._()-]+", "_", stem).strip(" ._") or "doc"
    stem = re.sub(r"\s+", " ", stem)[:60].strip()
    ext = ext if ext.lower() == ".pdf" else ".pdf"
    return f"{index:02d}_{stem}{ext}"


def list_corpora():
    d = corpora_dir()
    if not os.path.isdir(d):
        return []
    return [c for c in os.listdir(d) if os.path.isfile(os.path.join(d, c, "graph.graphml"))]


engine = get_engine()
sessions = get_sessions()

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Chats")
    if st.button("＋ New chat", use_container_width=True):
        s = sessions.create()
        st.session_state.active_session = s.id
        st.rerun()

    all_sessions = sessions.list()
    if all_sessions and "active_session" not in st.session_state:
        st.session_state.active_session = all_sessions[0].id

    for s in all_sessions:
        cols = st.columns([0.8, 0.2])
        if cols[0].button(s.title, key=f"sel_{s.id}", use_container_width=True):
            st.session_state.active_session = s.id
            st.rerun()
        if cols[1].button("🗑", key=f"del_{s.id}"):
            sessions.delete(s.id)
            st.session_state.pop("active_session", None)
            st.rerun()

    st.divider()
    st.subheader("Corpus")
    corpora = list_corpora()
    active_corpus = st.selectbox("Active corpus", ["(none)"] + corpora)
    if active_corpus == "(none)":
        active_corpus = None

active_session_id = st.session_state.get("active_session")
if active_session_id and active_corpus:
    sessions.set_corpus(active_session_id, active_corpus)

tab_lib, tab_chat, tab_compare, tab_graph = st.tabs(
    ["📚 Library", "💬 Chat", "📊 Compare", "🕸 Graph"]
)

# ----------------------------------------------------------------- library
with tab_lib:
    st.subheader("Build a corpus from PDFs")
    corpus_name = st.text_input("Corpus name", value="my_corpus")
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Build index", type="primary", disabled=not uploads):
        cdir = os.path.join(corpora_dir(), corpus_name, "_uploads")
        os.makedirs(cdir, exist_ok=True)
        paths = []
        for i, up in enumerate(uploads):
            p = os.path.join(cdir, safe_filename(up.name, i))
            with open(p, "wb") as f:
                f.write(up.getbuffer())
            paths.append(p)
        with st.spinner("Extracting propositions and building the knowledge graph..."):
            corpus = engine.indexer.build_from_pdfs(corpus_name, paths)
            engine._corpora[corpus_name] = corpus
            engine._retrievers.pop(corpus_name, None)
        g = corpus.graph
        st.success(
            f"Indexed '{corpus_name}': {len(corpus.chunk_store)} chunks, "
            f"{len(corpus.entity_store)} entities, {len(corpus.proposition_store)} propositions, "
            f"{g.vcount()} graph nodes / {g.ecount()} edges."
        )

# ----------------------------------------------------------------- chat
with tab_chat:
    if not active_session_id:
        st.info("Create a chat from the sidebar to begin.")
    elif not active_corpus:
        st.warning("Select an active corpus in the sidebar (or build one in Library).")
    else:
        session = sessions.get(active_session_id)
        for m in session.messages:
            with st.chat_message(m.role):
                st.markdown(m.content)
        prompt = st.chat_input("Ask about your PDFs...")
        if prompt:
            sessions.append(active_session_id, Message("user", prompt))
            if session.title == "New chat":
                sessions.rename(active_session_id, prompt[:40])
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and reasoning..."):
                    history = sessions.history_for_llm(active_session_id)[:-1]
                    result = engine.ask(active_corpus, prompt, history)
                st.markdown(result.answer)
                with st.expander("Retrieved passages"):
                    for p in result.passages:
                        st.markdown(f"**score {p.score:.4f}** — {p.text[:600]}")
            sessions.append(
                active_session_id,
                Message("assistant", result.answer, refs=[p.chunk_id for p in result.passages]),
            )
            st.rerun()

# --------------------------------------------------------------- compare
with tab_compare:
    st.subheader("Compare RAG systems side-by-side")
    st.caption(
        "Runs **BaseRAG** (dense), **GraphRAG** (entity-graph), and **PropRAG** "
        "(proposition beam + PPR) on one query. Same local embedding model for all "
        f"three. Chat (QA) calls are rate-limited to {engine.config.rpm_limit} RPM."
    )
    if not active_corpus:
        st.warning("Select an active corpus in the sidebar first.")
    else:
        cmp_query = st.text_input("Query", key="cmp_query",
                                  placeholder="Ask something answerable from the corpus...")
        ablation = st.checkbox(
            "Ablation: all systems retrieve over chunk vectors only",
            key="cmp_ablation",
            help="Strips the index differences that define GraphRAG/PropRAG so the "
                 "systems converge. This is an algorithm-only view, NOT a fair "
                 "framework comparison — leave OFF for the real benchmark.",
        )
        if st.button("Run comparison", type="primary", disabled=not cmp_query):
            label = "chunk-only ablation" if ablation else "BaseRAG, GraphRAG and PropRAG"
            with st.spinner(f"Running {label}..."):
                comparison = engine.compare(active_corpus, cmp_query, ablation=ablation)
            st.session_state["last_comparison"] = comparison

        comparison = st.session_state.get("last_comparison")
        if comparison is not None and comparison.query:
            # ---- fairness panel: the shared controls, made explicit
            if comparison.ablation:
                st.warning(
                    "**Ablation mode** — all systems retrieve over the same chunk "
                    "vectors. Results converge by design; this is not a framework "
                    "comparison."
                )
            st.markdown("#### Shared controls (held constant for every system)")
            fc = st.columns(4)
            fc[0].metric("Embedding model", comparison.embedding_model.split("/")[-1])
            fc[1].metric("Generation LLM", comparison.llm_model.split("/")[-1])
            fc[2].metric("Passages to generator", comparison.qa_top_k)
            fc[3].metric("Systems", len(comparison.systems))
            st.caption(
                "Only the **retrieval strategy** differs between systems — embedding "
                "model, generation LLM, query, and passages-fed-to-generator are identical."
            )
            if comparison.search_query != comparison.query:
                st.caption(f"Retrieval query (contextualized): _{comparison.search_query}_")

            # ---- metrics summary table + charts
            rows = []
            for sr in comparison.systems:
                u = sr.usage
                rows.append({
                    "System": sr.system,
                    "Passages": len(sr.passages),
                    "Answer chars": len(sr.answer),
                    "Retrieval (ms)": round(sr.retrieval_latency_s * 1000),
                    "QA (ms)": round(sr.qa_latency_s * 1000),
                    "Total (ms)": round(sr.total_latency_s * 1000),
                    "LLM calls": u.chat_calls,
                    "Chat tokens": u.chat_total_tokens,
                    "Embed calls": u.embed_calls,
                    "Embed tokens": u.embed_tokens,
                    "Total tokens": u.total_tokens,
                    "Rate wait (s)": round(u.rate_wait_s, 1),
                })
            metrics_df = pd.DataFrame(rows).set_index("System")
            st.markdown("#### Metrics")
            st.dataframe(metrics_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Total latency (ms)")
                st.bar_chart(metrics_df[["Total (ms)"]])
            with c2:
                st.caption("Token consumption")
                st.bar_chart(metrics_df[["Chat tokens", "Embed tokens"]])

            # ---- side-by-side answers + passages
            st.markdown("#### Answers")
            cols = st.columns(len(comparison.systems))
            for col, sr in zip(cols, comparison.systems):
                with col:
                    st.markdown(f"**{sr.system}**")
                    if sr.error:
                        st.error(sr.error)
                    else:
                        st.markdown(sr.answer or "_(no answer)_")
                    with st.expander(f"Passages ({len(sr.passages)})"):
                        for p in sr.passages:
                            st.markdown(f"**{p.score:.4f}** — {p.text[:400]}")

# ----------------------------------------------------------------- graph
with tab_graph:
    if not active_corpus:
        st.warning("Select an active corpus first.")
    else:
        corpus = engine.corpus(active_corpus)
        ents = list_entities(corpus)
        choice = st.selectbox("Entity", ents) if ents else None
        typed = st.text_input("...or type an entity name")
        hops = st.slider("Neighborhood hops", 1, 2, 1)
        query = typed.strip() or choice
        if query:
            key = find_entity_key(corpus, query)
            if key:
                components.html(ego_graph_html(corpus, key, hops=hops), height=640, scrolling=True)
            else:
                st.error(f"No entity matching '{query}'.")
