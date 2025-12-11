#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from matplotlib import font_manager, rcParams
import os

# 如果你要使用 OpenAI embedding
from openai import OpenAI

# --- 字型設定（依你原本的邏輯） ---
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKtc-Regular.otf")
font_manager.fontManager.addfont(FONT_PATH)
font_prop = font_manager.FontProperties(fname=FONT_PATH)

rcParams["font.family"] = font_prop.get_name()
rcParams["axes.unicode_minus"] = False


# --- 頁面設定 ---
st.set_page_config(page_title="RAG 語意分析實驗室", layout="wide")
st.title("🧬 RAG 語義探索：Embedding 相似度矩陣")


# -----------------------------------
# 📌 模型清單（8個你想加入的）
# -----------------------------------
FREE_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "免費 - 中文友善",
    "all-MiniLM-L6-v2": "免費 - 英文最佳",
    "thenlper/gte-large": "免費 - 多語大模型",
   # "LaBSE": "免費 - Google 多語雙向模型",
   # "distiluse-base-multilingual-cased-v2": "免費 - 多語舊模型",
    "BAAI/bge-base-zh-v1.5": "免費 - 中文 BGE 基礎模型（更快）",
    "BAAI/bge-large-zh-v1.5": "免費 - 中文 BGE 大模型（最新）"
}

OPENAI_MODELS = {
    "text-embedding-3-small": "付費 - 新一代高速模型",
    "text-embedding-3-large": "付費 - 高品質語意模型"
}

MODEL_OPTIONS = list(FREE_MODELS.keys()) + list(OPENAI_MODELS.keys())


# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")

    model_name = st.selectbox(
        "選擇 Embedding 模型",
        MODEL_OPTIONS,
        index=0
    )

    # 若選 OpenAI 才需要 API Key
    api_key = None
    if model_name in OPENAI_MODELS:
        api_key = st.text_input("請輸入 OpenAI API Key（若選 OpenAI 模型必填）",
                                type="password")

    # 模型總覽表
    model_df = pd.DataFrame([
        ["paraphrase-multilingual-MiniLM-L12-v2", "多語（含中文）", "免費", "多語言平行語料（50+語言翻譯對，如OPUS、Wikipedia），Sentence Transformers通用微調數據", "~2021年（發布時）"],
        ["all-MiniLM-L6-v2", "英文", "免費", "超過10億英文/多語句對（Reddit、STS基準、平行翻譯），無監督預訓練+對比微調", "~2021年（發布時）"],
        ["thenlper/gte-large", "多語（含中文）", "免費", "多階段對比學習數據（英文網頁、弱監督標記對），聚焦MTEB任務優化，未公開具體來源", "~2023年（英文優化發布）"],
        #["distiluse-base-multilingual-cased-v2", "多語", "免費", "較舊但穩定的模型"],
        ["BAAI/bge-large-zh-v1.5", "中文", "免費", "同 base，但此模型較大，可能因記憶體不足導致錯誤", "~2023年中"],
        ["BAAI/bge-base-zh-v1.5", "中文", "免費", "使用 WuDao, Zhihu, Baike 進行訓練，效果略低於 large", "~2023年中"],
        ["text-embedding-3-small", "全語言", "付費", "大規模網絡爬取數據（Common Crawl類似），多語言弱監督+監督微調，類似ada-002", "2021/9"],
        ["text-embedding-3-large", "全語言", "付費", "同small，大規模網絡數據預訓練，強調多語言檢索微調", "2021/9"]
    ], columns=["模型名稱", "語言", "類型", "說明", "訓練資料估計截止時間"])

    st.markdown("### 📘 模型總覽")
    st.table(model_df)
    
    st.info("💡 小提示：\n1.0 = 完全一樣\n0.8 以上 = 語意高度相關")


# --- 主畫面 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 請輸入測試語句 (每行一句)")

    default_text = "貓喜歡吃魚\n小貓愛吃海鮮\n今天天氣很好\n股市大漲\n我想要寫程式"
    user_input = st.text_area("在此輸入...", value=default_text, height=200)

    analyze_btn = st.button("🚀 開始分析語意", type="primary")


# ----------------------------------------------------------
# 🚀 Embedding 與分析邏輯
# ----------------------------------------------------------
if analyze_btn and user_input:

    texts = [line.strip() for line in user_input.split("\n") if line.strip()]

    if len(texts) < 2:
        st.warning("請至少輸入兩行文字來進行比較！")
    else:
        with st.spinner(f"正在載入模型：{model_name}..."):

            # ----------------------------------------------------------
            # 1️⃣ 免費模型（SentenceTransformer）
            # ----------------------------------------------------------
            if model_name in FREE_MODELS:
                model = SentenceTransformer(model_name)
                embeddings = model.encode(texts)

            # ----------------------------------------------------------
            # 2️⃣ OpenAI 付費 Embedding
            # ----------------------------------------------------------
            elif model_name in OPENAI_MODELS:

                if not api_key:
                    st.error("你選擇了 OpenAI 模型，必須輸入 API Key！")
                    st.stop()

                client = OpenAI(api_key=api_key)

                response = client.embeddings.create(
                    model=model_name,
                    input=texts
                )

                embeddings = [item.embedding for item in response.data]

            # ----------------------------------------------------------
            # 3️⃣ 計算相似度
            # ----------------------------------------------------------
            sim_matrix = cosine_similarity(embeddings)
            df_sim = pd.DataFrame(sim_matrix, index=texts, columns=texts)

        # --- 熱圖 ---
        with col2:
            st.subheader("2. 分析結果 (熱圖)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_sim, annot=True, fmt=".2f",
                        cmap="coolwarm", vmin=0, vmax=1, ax=ax)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        # --- 詳細數據表 ---
        st.markdown("---")
        st.subheader("3. 詳細數據矩陣")
        st.dataframe(
            df_sim.style.background_gradient(axis=None, cmap="Blues", vmin=0.5, vmax=1.0)
        )


# In[ ]:




