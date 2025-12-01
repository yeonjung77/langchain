import os
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from search_timeline import (
    generate_timeline_synthesis,
    search_keyword_timeline,
    summarize_yearly_insights,
)

# ========================================
# 기본 설정
# ========================================
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("❌ GROQ_API_KEY가 없습니다. .env 또는 Streamlit Secrets에 등록해주세요.")
    st.stop()


# ========================================
# 벡터스토어 로딩
# ========================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


# ========================================
# LLM 로딩
# ========================================
@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=groq_key,
    )


@st.cache_resource
def load_bm25_retriever(_vs: FAISS):
    # 벡터스토어 안에 있는 전체 문서를 기반으로 BM25 인덱스 생성
    all_docs = list(_vs.docstore._dict.values())
    # k는 한 번에 반환할 최대 문서 수 (여유 있게 설정)
    return BM25Retriever.from_documents(all_docs, k=50)


vectorstore = load_vectorstore()
bm25_retriever = load_bm25_retriever(vectorstore)
llm = load_llm()

# 기본 retriever는 k를 조금 넉넉하게
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

CHAPTER_LABELS = ["Global Economy", "Consumer Shifts", "Fashion System"]


# ========================================
# 하이브리드 검색 함수 (semantic + BM25)
# ========================================
def hybrid_search(
    query: str,
    semantic_k: int = 30,
    keyword_k: int = 30,
    combined_k: int = 12,
    chapter_filter: str | None = None,
    region_filter: str | None = None,
):
    """
    - semantic: FAISS similarity_search
    - keyword: BM25Retriever
    두 결과의 rank를 점수로 변환해서 가중 평균 후 재정렬.
    """
    semantic_docs = vectorstore.similarity_search(query, k=semantic_k)
    # 최신 BM25Retriever는 get_relevant_documents 대신 invoke 사용
    keyword_docs = bm25_retriever.invoke(query)[:keyword_k]

    def make_key(doc):
        return (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content,
        )

    scores = {}
    n_sem = len(semantic_docs) or 1
    n_kw = len(keyword_docs) or 1

    # semantic rank 기반 점수 (높을수록 좋게)
    for rank, doc in enumerate(semantic_docs):
        key = make_key(doc)
        sem_score = (n_sem - rank) / n_sem
        prev_sem, prev_kw, prev_doc = scores.get(key, (0.0, 0.0, doc))
        scores[key] = (max(prev_sem, sem_score), prev_kw, doc)

    # BM25 rank 기반 점수
    for rank, doc in enumerate(keyword_docs):
        key = make_key(doc)
        kw_score = (n_kw - rank) / n_kw
        prev_sem, prev_kw, prev_doc = scores.get(key, (0.0, 0.0, doc))
        scores[key] = (prev_sem, max(prev_kw, kw_score), doc)

    # 가중 평균으로 최종 점수 생성
    alpha = 0.6  # semantic 비중
    scored_docs = []
    for sem_score, kw_score, doc in scores.values():
        final_score = alpha * sem_score + (1 - alpha) * kw_score

        # 메타데이터 기반 필터링
        if chapter_filter and doc.metadata.get("chapter") != chapter_filter:
            continue
        if region_filter and doc.metadata.get("region") != region_filter:
            continue

        scored_docs.append((final_score, doc))

    # 필터링 후 결과가 너무 적으면 필터 없이 fallback
    if not scored_docs:
        scored_docs = [
            (
                alpha * ((n_sem - i) / n_sem),
                d,
            )
            for i, d in enumerate(semantic_docs)
        ]

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored_docs[:combined_k]]


# ========================================
# 문서 그룹 로딩
# ========================================
@st.cache_resource
def load_grouped_docs():
    all_docs = list(vectorstore.docstore._dict.values())
    by_year_chapter = defaultdict(list)
    by_chapter = defaultdict(list)

    for d in all_docs:
        year = d.metadata.get("year")
        chapter = d.metadata.get("chapter")
        by_year_chapter[(year, chapter)].append(d)
        by_chapter[chapter].append(d)

    return by_year_chapter, by_chapter


by_year_chapter, by_chapter = load_grouped_docs()


# ========================================
# 헬퍼: 문서 포맷팅
# ========================================
def format_docs(docs):
    processed = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", "?")
        year = d.metadata.get("year", "")
        chapter = d.metadata.get("chapter", "")
        region = d.metadata.get("region", "")
        if region:
            header = f"[{year} / {chapter} / {region} / {src} p.{page}]"
        else:
            header = f"[{year} / {chapter} / {src} p.{page}]"
        processed.append(header + "\n" + d.page_content)
    return "\n\n".join(processed)


# ========================================
# 공통 RAG 프롬프트
# ========================================
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Fashion MD Research Assistant.\n"
            "Use ONLY the content from McKinsey & BoF 'State of Fashion' (2021–2025).\n"
            "답변은 한국어로, 핵심 용어는 영어 병기해줘.",
        ),
        (
            "human",
            "질문: {question}\n\n"
            "참고 문서:\n{context}",
        ),
    ]
)

qa_chain = qa_prompt | llm | StrOutputParser()


# ========================================
# 대화 로그 기반 리포트 생성용 프롬프트
# ========================================
report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior fashion strategy consultant.\n"
            "Below is a conversation between a Fashion MD and an AI research assistant\n"
            "about insights from McKinsey & BoF 'State of Fashion' (2021–2025).\n"
            "Use ONLY information that can be reasonably grounded in this conversation.\n"
            "답변은 한국어로 작성하고, 핵심 개념은 필요할 때만 영어 병기해줘.",
        ),
        (
            "human",
            "다음은 사용자(패션 MD)와 AI 리서치 어시스턴트의 대화 로그입니다.\n"
            "이 대화를 바탕으로 간결한 인사이트 리포트를 작성해주세요.\n\n"
            "대화 로그:\n{conversation}\n\n"
            "📌 리포트 구성은 다음 섹션을 포함해 주세요.\n"
            "1. Executive Summary\n"
            "2. Key Insights (bullet 형태)\n"
            "3. Implications & Action Ideas (현업 활용 아이디어 중심)\n\n"
            "⚠️ 주의사항\n"
            "- 반드시 대화 내용에서 파생될 수 있는 인사이트만 정리할 것\n"
            "- McKinsey/BoF 리포트에 일반적으로 등장할 법한 문장이라도, 대화에 전혀 나오지 않았다면 생성하지 말 것\n"
            "- 한국어 문장을 사용하되, 필요한 핵심 용어만 영어 병기\n"
            "- 문장은 짧고 명료하게, 실제 보고서에 바로 붙여 넣을 수 있는 톤으로 작성",
        ),
    ]
)

report_chain = report_prompt | llm | StrOutputParser()


# ========================================
# Streamlit UI 시작
# ========================================
st.set_page_config(page_title="State of Fashion — AI Insight Engine")

st.title("The State of Fashion")
st.title("- AI Insight Engine")
st.caption("AI-powered Insight from SoF 2021–2025 Reports")

st.markdown("---")

# ========================================
# 메인 탭 구성
# ========================================
tab_main, tab_keyword, tab_chapter, tab_country, tab_chat = st.tabs([
    "1️⃣ AI Report Search",
    "2️⃣ Keyword Analytics",
    "3️⃣ Chapter Insights",
    "4️⃣ Regional Insights",
    "5️⃣ Chat & Report",
])


# ============================================================================
# 📌 TAB 1 — 전체 검색 & 질문하기
# ============================================================================
with tab_main:
    st.subheader("Ask Anything — AI Analyzes the Report to Answer Your Questions")

    question = st.text_area("질문 입력", key="qa_question")
    chapter_filter = st.selectbox(
        "검색할 챕터 (옵션)", ["전체"] + CHAPTER_LABELS, index=0
    )

    if st.button("AI에게 질문하기", key="qa_button"):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("보고서를 분석하고 있습니다..."):
                ch = None if chapter_filter == "전체" else chapter_filter

                # 하이브리드 검색으로 문서 검색
                docs = hybrid_search(
                    question,
                    semantic_k=30,
                    keyword_k=30,
                    combined_k=12,
                    chapter_filter=ch,
                )

                # LLM 컨텍스트는 상위 8개 정도만 사용
                context = format_docs(docs[:8])
                answer = qa_chain.invoke({"question": question, "context": context})

            st.markdown("### 📌 답변")
            st.write(answer)

            # -----------------------
            # RAG Validation Snippets
            # -----------------------
            st.markdown("### 🔍 참고 문장 (Top 3)")
            if not docs:
                st.info("참고할 문서를 찾지 못했습니다.")
            else:
                for i, d in enumerate(docs[:3], start=1):
                    src = os.path.basename(d.metadata.get("source", ""))
                    page = d.metadata.get("page", "?")
                    year = d.metadata.get("year", "")
                    chapter = d.metadata.get("chapter", "")
                    region = d.metadata.get("region", "")

                    meta_line = f"{year} / {chapter}"
                    if region:
                        meta_line += f" / {region}"
                    meta_line += f" / {src} p.{page}"

                    st.markdown(f"**[{i}] {meta_line}**")
                    st.write(d.page_content)
                    st.markdown("---")


# ============================================================================
# 📌 TAB 2 — Chapter Insight (서브탭 4개)
# ============================================================================
with tab_chapter:

    sub1, sub2, sub3 = st.tabs(
        [
            "Annual Keyword Insights",
            "Chapter Keyword Timeline",
            "Keyword Mapping"
        ]
    )

    # ---------------------------------------------------
    # 📌 서브탭 1 — 연도별 핵심 키워드
    # ---------------------------------------------------
    with sub1:
        st.subheader("Key Keywords by Year")

        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("연도 선택", [2021, 2022, 2023, 2024, 2025])
        with col2:
            chapter = st.selectbox("챕터 선택", CHAPTER_LABELS)

        if st.button("키워드 생성", key="year_chapter_summary_keywords"):
            key = (year, chapter)
            docs = by_year_chapter.get(key, [])

            if not docs:
                st.warning("해당 연도/챕터에 대한 문서를 찾을 수 없습니다.")
            else:
                text = "\n\n".join(d.page_content for d in docs[:20])

                summary_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a senior fashion strategy analyst. "
                            "아래 텍스트를 기반으로 해당 연도/챕터의 핵심 트렌드 키워드를 5개 뽑아 "
                            "각 키워드당 1~2문장 설명을 만들어줘.\n"
                            "설명은 한국어로, 중요한 용어는 영어 병기해줘."
                        ),
                        (
                            "human",
                            "연도: {year}\n챕터: {chapter}\n\n"
                            "분석 텍스트:\n{text}\n\n"
                            "➡ 출력 형식:\n"
                            "Key Insights\n"
                            "- 키워드 1: 설명(1~2줄)\n"
                            "- 키워드 2: 설명\n"
                            "- 키워드 3: 설명\n"
                            "- 키워드 4: 설명\n"
                            "- 키워드 5: 설명"
                        ),
                    ]
                )

                chain = summary_prompt | llm | StrOutputParser()

                with st.spinner("핵심 키워드를 추출하는 중..."):
                    summary = chain.invoke(
                        {"year": year, "chapter": chapter, "text": text}
                    )

                st.write(summary)


    # ---------------------------------------------------
    # 📌 서브탭 2 — 챕터별 키워드 타임라인
    # ---------------------------------------------------
    with sub2:
        st.subheader("Chapter-Based Keyword Timeline Analysis")

        keyword = st.text_input(
            "분석할 키워드 (예: AI, resale, sustainability, Gen Z, silver spenders...)", key="timeline_keyword"
        )
        chapter_sel = st.selectbox(
            "챕터 선택", ["전체"] + CHAPTER_LABELS, index=0, key="timeline_chapter"
        )

        if st.button("타임라인 생성", key="timeline_button"):
            if not keyword.strip():
                st.warning("키워드를 입력해주세요.")
            else:
                ch = None if chapter_sel == "전체" else chapter_sel

                with st.spinner("타임라인 분석 중..."):
                    grouped = search_keyword_timeline(keyword, retriever, chapter=ch)

                    timeline_full = {yr: grouped.get(yr, []) for yr in [2021, 2022, 2023, 2024, 2025]}

                    yearly_summary = {}
                    for yr, docs in timeline_full.items():

                        if not docs:
                            yearly_summary[yr] = "⚠️ 해당 연도에서는 키워드 언급이 거의 없었습니다."
                        else:
                            text = "\n\n".join(docs[:3])
                            prompt = ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        "You are a fashion trend analyst. "
                                        "아래 텍스트에 기반하여 해당 연도의 관점을 2~3문장으로 요약해줘.\n"
                                        "❗ 절대 금지:\n"
                                        "- '2023년의 키워드는 ~입니다' 같은 문장 생성\n"
                                        "- 텍스트에 없는 대표 키워드 생성\n"
                                        "- 패션 트렌드 키워드 선언\n"
                                        "- 해석 지어내기\n"
                                        "❗ 반드시 지킬 것:\n"
                                        "- 텍스트 기반 요약만 생성\n"
                                        "- 한국어로 설명하되 핵심 용어만 영어 병기"
                                    ),
                                    (
                                        "human",
                                        "키워드: {keyword}\n연도: {year}\n\n텍스트:\n{text}"
                                    ),
                                ]
                            )
                            chain = prompt | llm | StrOutputParser()
                            summary = chain.invoke({"keyword": keyword, "year": yr, "text": text})
                            yearly_summary[yr] = summary

                    synthesis_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a senior fashion strategist."
                                "연도별 분석 내용을 기반으로 전체 흐름을 딱 3문장으로 요약.\n"
                                "❗ 절대 금지:\n"
                                "- '전체 키워드는 ~입니다' 문장 생성\n"
                                "- 대표 키워드 선언\n"
                                "- 텍스트에 없는 개념 추가\n"
                                "❗ 반드시 지킬 것:\n"
                                "- 자연스러운 3문장 요약만 생성"
                            ),
                            (
                                "human",
                                "키워드: {keyword}\n\n연도별 내용:\n{summary}"
                            ),
                        ]
                    )

                    combined = "\n".join(f"[{yr}] {txt}" for yr, txt in yearly_summary.items())
                    chain = synthesis_prompt | llm | StrOutputParser()
                    synthesis = chain.invoke({"keyword": keyword, "summary": combined})

                st.subheader(f"키워드 타임라인 : **{keyword}**")

                for yr in [2021, 2022, 2023, 2024, 2025]:
                    st.write(f"### 📌 {yr}년")
                    st.write(yearly_summary[yr])
                    st.markdown("---")

                st.write("### 전체 흐름 요약")
                st.write(synthesis)


    # ---------------------------------------------------
    # 📌 서브탭 3 — 키워드 × 챕터 매핑
    # ---------------------------------------------------
    with sub3:
        st.subheader("Keyword Mapping Table")

        keyword_map = st.text_input(
            "키워드 입력 (예: AI, resale, sustainability, Gen Z, silver spenders...)", key="mapping_keyword"
        )

        if st.button("매핑 생성", key="mapping_button"):
            if not keyword_map.strip():
                st.warning("키워드를 입력해주세요.")
            else:
                import pandas as pd

                rows = []

                with st.spinner("매핑 테이블 생성 중..."):
                    for ch in CHAPTER_LABELS:
                        grouped = search_keyword_timeline(keyword_map, retriever, chapter=ch)

                        # 📌 챕터 내 검색결과 없을 경우
                        if not grouped:
                            rows.append({"Chapter": ch, "Perspective": "관련된 내용이 부족합니다."})
                            continue

                        # 연도별 요약
                        yearly = summarize_yearly_insights(grouped, keyword_map, chapter=ch)

                        # 연도별 텍스트 조합
                        combined = "\n\n".join(
                            f"[{y}]\n{txt}" for y, txt in sorted(yearly.items())
                        )

                        # 📌 핵심 문장 3문장만 생성하도록 제한하는 프롬프트
                        map_prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    "You are a fashion strategy analyst."
                                    "아래 요약 텍스트를 기반으로 해당 챕터가 이 키워드를 어떻게 다루는지 핵심 3문장으로만 정리해줘\n"
                                    "⚠️ 절대 금지:\n"
                                    "- '키워드: ~' 형식 문장 생성 금지\n"
                                    "- '202X년 ~ 흐름은 다음과 같습니다' 금지\n"
                                    "- 텍스트에 없는 숫자/사실/키워드 생성 금지\n"
                                    "⚠️ 반드시 지킬 것:\n"
                                    "- 텍스트 기반 핵심 내용을 자연스러운 3문장으로만 요약\n"
                                    "- 한국어로 서술, 필요한 경우 핵심 용어만 영어 병기"
                                ),
                                (
                                    "human",
                                    "키워드: {keyword}\n챕터: {chapter}\n\n"
                                    "요약 텍스트:\n{summary}"
                                ),
                            ]
                        )

                        chain = map_prompt | llm | StrOutputParser()

                        perspective = chain.invoke(
                            {
                                "keyword": keyword_map,
                                "chapter": ch,
                                "summary": combined,
                            }
                        )

                        rows.append({"Chapter": ch, "Perspective": perspective})

                df = pd.DataFrame(rows)
                st.table(df)

# =====================================================================
# 📌 TAB 2 — 국가별 인사이트
# =====================================================================
with tab_country:

    st.subheader("🌍 Regional Market Insights (2024 & 2025)")

    country = st.selectbox(
        "국가 선택",
        ["🇯🇵 Japan", "🇮🇳 India", "🇺🇸 US", "🇨🇳 China", "🇪🇺 EU"],
        index=0,
    )

    # 국가명을 AI가 이해할 수 있는 텍스트로 변환
    country_map = {
        "🇯🇵 Japan": "Japan",
        "🇮🇳 India": "India",
        "🇺🇸 US": "United States",
        "🇨🇳 China": "China",
        "🇪🇺 EU": "European Union",
    }
    country_text = country_map[country]

    if st.button("국가별 인사이트 생성", key="country_insight"):
        with st.spinner("국가별 시장 인사이트 분석 중..."):

            # 1) RAG 검색: 국가 관련 문서 필터링
            query = f"{country_text} market consumer trend economy fashion"

            # region 메타데이터를 활용한 하이브리드 검색
            docs = hybrid_search(
                query,
                semantic_k=30,
                keyword_k=30,
                combined_k=25,
                region_filter=country_text,
            )

            # 연도별 분리
            docs_2025 = [d.page_content for d in docs if d.metadata.get("year") == 2025]
            docs_2024 = [d.page_content for d in docs if d.metadata.get("year") == 2024]

            def get_summary(texts, year):
                """LLM을 이용한 연도별 요약 함수"""
                if not texts:
                    return f"⚠️ {year}년에는 해당 국가 관련 정보가 거의 없습니다."

                combined = "\n\n".join(texts[:8])  # 너무 긴 경우 압축

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a senior global fashion strategist.\n"
                            "아래 텍스트를 기반으로 해당 국가의 시장 특성을 정확하게 3문장으로만 요약하라.\n\n"
                            "⚠️ 절대 금지:\n"
                            "- '해당 국가의 시장 특성은 다음과 같다' 같은 서론 문장 생성 금지\n"
                            "- 키워드 선언(예: '2025년의 키워드는 ~이다') 금지\n\n"
                            "- '키워드: ~' 형식 금지\n"
                            "- '202X년의 트렌드는 ~입니다' 금지\n"
                            "- '~의 시장 특성은 다음과 같다.' 금지\n"
                            "- '~의 시장은 다음과 같다.' 금지\n"
                            "- 외래 문자·비자연스러운 어구 생성 금지\n"
                            "- 텍스트에 없는 추론/가정/숫자 생성 금지\n"
                            "- 서론·결론·장식적 문장 금지\n\n"
                            "- 결론·조언 문장 금지\n"
                            "⚠️ 반드시 지킬 것:\n"
                            "- 텍스트 기반 핵심만 3문장\n"
                            "- 한국어로 생성, 필요 시 핵심 용어만 영어 병기"
                            "- 오직 텍스트에 있는 사실만 3개의 자연스러운 한국어 문장으로 정리\n"
                            "- 전문적인 문체 유지, 단문/군더더기 없는 표현\n"
                            "- 필요한 경우에만 핵심 용어 영어 병기"
                        ),
                        (
                            "human",
                            f"{year}년의 '{country_text}' 관련 텍스트:\n\n{combined}"
                        ),
                    ]
                )

                chain = prompt | llm | StrOutputParser()
                return chain.invoke({})

            summary_2025 = get_summary(docs_2025, 2025)
            summary_2024 = get_summary(docs_2024, 2024)

        # 출력 UI
        st.markdown(f"### 🌍 {country_text} — Market Insights")

        st.write("### 📌 2025년")
        st.write(summary_2025)
        st.markdown("---")

        st.write("### 📌 2024년")
        st.write(summary_2024)

# =====================================================================
# 📌 TAB — 키워드 시각화 (Top 10 Bar + Top3 Line Chart)
# =====================================================================
with tab_keyword:

    st.subheader("Top 10 Keywords")

    import re
    from collections import Counter
    import pandas as pd
    import plotly.express as px

    # ---------------------------
    # (A) 강화된 키워드 필터링 함수
    # ---------------------------
    def extract_keywords(text):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        tokens = [t.lower() for t in tokens if len(t) > 3]

        stopwords = {
            # 일반 영어 불용어
            "that","with","this","have","from","will","into","been","more","than",
            "their","which","also","about","what","when","were","your","them","they",
            "over","only","some","make","made","like","just","very","those","while",
            "where","such","many","each","most","much","other","would","should",
            "could","might","these","both","through","across","there","after","before",
            "under","between","because","based","during","within","without","using",
            "over","well","however","even","though","still","every","including",

            # 숫자 표현
            "percent","million","billion","thousand",

            # 패션 문서에서 너무 기본적인 단어들
            "brands","brand","business","market","industry","consumer","consumers","customer",
            "customers","global","fashion","system","trend","analysis","report",
            "state","chapter","growth","people","products","product","value",
            "goods","retail","sales","year","years","company","companies",

            # 불필요 토큰
            "said","https","http","mckinsey",
        }

        tokens = [t for t in tokens if t not in stopwords]

        # 추가 필터링
        tokens = [t for t in tokens if not t.endswith("ing")]     # 동명사 제거
        tokens = [t for t in tokens if len(set(t)) > 2]           # 반복 문자 제거

        return tokens

    # ---------------------------
    # (B) 연도별 텍스트 취합
    # ---------------------------
    year_texts = {year: "" for year in [2021, 2022, 2023, 2024, 2025]}
    all_docs = list(vectorstore.docstore._dict.values())

    for d in all_docs:
        y = d.metadata.get("year")
        if y in year_texts:
            year_texts[y] += " " + d.page_content

    yearly_keyword_counts = {
        year: Counter(extract_keywords(text))
        for year, text in year_texts.items()
    }

    # ---------------------------
    # (C) 연도 선택 UI
    # ---------------------------
    selected_year = st.selectbox(
        "연도 선택",
        [2021, 2022, 2023, 2024, 2025],
        key="keyword_visual_year"
    )

    st.markdown("---")

    # ---------------------------
    # (D) Bar Chart 출력
    # ---------------------------

    top_keywords = yearly_keyword_counts[selected_year].most_common(10)

    if not top_keywords:
        st.warning("해당 연도에서 의미 있는 키워드를 찾지 못했습니다.")
        st.stop()

    df_bar = pd.DataFrame({
        "keyword": [k for k, _ in top_keywords],
        "count": [v for _, v in top_keywords],
    })

    fig = px.bar(
        df_bar,
        x="keyword",
        y="count",
        title=f"{selected_year} Keyword Top 10",
        color="count",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("Top 3 Keywords — Yearly Trend (2021–2025)")

    # ---------------------------
    # (E) 상위 3개 키워드 선택
    # ---------------------------
    top3_keywords = [k for k, _ in top_keywords[:3]]

    # ---------------------------
    # (F) Top3 키워드를 연도별로 빈도 기반 변화 계산
    # ---------------------------
    for keyword in top3_keywords:
        trend_counts = []
        for yr in [2021, 2022, 2023, 2024, 2025]:
            cnt = yearly_keyword_counts[yr][keyword]
            trend_counts.append(cnt)

        df_line = pd.DataFrame({
            "year": ["2021", "2022", "2023", "2024", "2025"],
            "count": trend_counts
        })

        df_line["year"] = df_line["year"].astype(str)

        st.write(f"🔎 {keyword}")

        fig_line = px.line(
            df_line,
            x="year",
            y="count",
            markers=True
        )

        fig_line.update_xaxes(type="category")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("---")


# =====================================================================
# 📌 TAB 5 — 대화형 챗봇 & 리포트 생성
# =====================================================================
with tab_chat:
    st.subheader("Conversational Strategy Copilot")
    st.caption("챗봇과 자유롭게 대화한 뒤, 대화 내용을 리포트로 정리할 수 있습니다.")

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_report" not in st.session_state:
        st.session_state.chat_report = ""

    # 이전 메시지 출력
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력
    user_input = st.chat_input("패션·리테일 인사이트에 대해 자유롭게 질문해보세요.")

    if user_input:
        # 사용자 메시지 추가 및 표시
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG 기반 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("AI가 SoF 리포트를 참고해 답변 중입니다..."):
                docs = hybrid_search(
                    user_input,
                    semantic_k=30,
                    keyword_k=30,
                    combined_k=12,
                )
                context = format_docs(docs[:8])
                answer = qa_chain.invoke(
                    {"question": user_input, "context": context}
                )
                st.markdown(answer)

        # 어시스턴트 메시지를 히스토리에 저장
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    st.markdown("---")
    st.markdown("### 📝 대화 내용을 리포트로 정리하기")

    col_report_btn, col_clear = st.columns([2, 1])

    with col_report_btn:
        generate_report = st.button("대화 내용으로 리포트 생성")
    with col_clear:
        clear_chat = st.button("대화 및 리포트 초기화")

    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.chat_report = ""
        st.experimental_rerun()

    if generate_report:
        if not st.session_state.chat_history:
            st.warning("먼저 챗봇과 몇 번 대화를 나눈 뒤 리포트를 생성해주세요.")
        else:
            # 대화 로그를 하나의 텍스트로 병합
            lines = []
            for msg in st.session_state.chat_history:
                role_label = "사용자" if msg["role"] == "user" else "AI"
                lines.append(f"{role_label}: {msg['content']}")

            conversation_text = "\n".join(lines)

            with st.spinner("대화 내용을 요약 리포트로 정리하는 중입니다..."):
                report = report_chain.invoke({"conversation": conversation_text})

            st.session_state.chat_report = report

    if st.session_state.chat_report:
        st.markdown("### 📄 Generated Conversation Report")
        st.write(st.session_state.chat_report)
