#!/usr/bin/env python
# coding: utf-8

# In[6]:


### import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer   # ⭐ 使用免費 Embedding 模型
import os
from matplotlib import font_manager, rcParams

# 尋找專案內 fonts/ 的字型檔（請把檔名改成你實際放的那個）
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKtc-Regular.otf")

# 把字型加入 matplotlib
font_manager.fontManager.addfont(FONT_PATH)
font_prop = font_manager.FontProperties(fname=FONT_PATH)

# 設定全域字型
rcParams["font.family"] = font_prop.get_name()
rcParams["axes.unicode_minus"] = False   # 避免負號變成方塊

# 如果你想確認 Cloud 上有沒有成功載入字型，可以暫時加這行：
# st.write("Using font:", font_prop.get_name())

# --- 頁面設定 ---
st.set_page_config(page_title="RAG 語意分析實驗室（免費版）", layout="wide")
st.title("🧬 免費版 RAG 語義探索：Embedding 相似度矩陣")
st.markdown("本版本使用 **SentenceTransformers（免費模型）** 計算 Embedding，不需 API Key！")


# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")

    model_name = st.selectbox(
        "選擇免費 Embedding 模型",
        [
            "paraphrase-multilingual-MiniLM-L12-v2",  # ⭐ 中文友善
            "all-MiniLM-L6-v2"  # 英文最佳
        ],
        index=0
    )

    st.info("💡 小提示：\n1.0 = 完全一樣\n0.8 以上 = 語意高度相關")


# --- 主畫面 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 請輸入測試語句 (每行一句)")

    default_text = "貓喜歡吃魚\n小貓愛吃海鮮\n今天天氣很好\n股市大漲\n我想要寫程式"
    user_input = st.text_area("在此輸入...", value=default_text, height=200)

    analyze_btn = st.button("🚀 開始分析語意", type="primary")


# --- Embedding 與分析 ---
if analyze_btn and user_input:

    # 整理輸入文字
    texts = [line.strip() for line in user_input.split("\n") if line.strip()]

    if len(texts) < 2:
        st.warning("請至少輸入兩行文字來進行比較！")
    else:
        with st.spinner("正在載入免費 Embedding 模型並計算向量..."):
            # ⭐ 載入免費模型
            model = SentenceTransformer(model_name)

            # 計算 Embeddings（完全不需 API）
            embeddings = model.encode(texts)

        # 計算相似度矩陣
        sim_matrix = cosine_similarity(embeddings)
        df_sim = pd.DataFrame(sim_matrix, index=texts, columns=texts)

        # --- 熱圖 ---
        with col2:
            st.subheader("2. 分析結果 (熱圖)")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="coolwarm",
                        vmin=0, vmax=1, ax=ax)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        # --- 詳細數據表 ---
        st.markdown("---")
        st.subheader("3. 詳細數據矩陣")

        st.dataframe(
            df_sim.style.background_gradient(axis=None, cmap="Blues", vmin=0.5, vmax=1.0)
        )


# In[ ]:




