import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- 解決 Matplotlib/Seaborn 中文顯示亂碼的設定 ---
import matplotlib.pyplot as plt

# 確保在 Windows 環境下，使用常見的中文字型（例如微軟正黑體）
# 備註：如果 'Microsoft YaHei' 不存在，它會嘗試使用列表中的下一個字型 'SimHei'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']

# 解決 Matplotlib 負號 '-' 顯示為方塊的問題
plt.rcParams['axes.unicode_minus'] = False
# ----------------------------------------------------

# --- 頁面設定 ---
st.set_page_config(page_title="RAG 語意分析實驗室", layout="wide")
st.title("🧬 RAG 語義探索：Embedding 相似度矩陣")
st.markdown("嘗試輸入不同的語句，看看 ChatGPT 眼中它們的「距離」有多近。**善意提醒，模型缺乏對2021年9月之後發生的事件的了解！")

# --- 側邊欄：設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("請輸入 OpenAI API Key", type="password", help="您的金鑰不會被儲存")
    model_name = st.selectbox("選擇模型", ["text_embedding_3_small", "text_embedding_3_large"], index=0)
    
    st.info("💡 小提示：\n1.0 代表完全一樣\n0.8 以上通常代表語意高度相關")

# --- 主畫面：輸入區 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 請輸入測試語句 (每行一句)")
    # 預設範例
    default_text = "貓喜歡吃魚\n小貓愛吃海鮮\n今天天氣很好\n股市大漲\n我想要寫程式"
    user_input = st.text_area("在此輸入...", value=default_text, height=200)
    
    analyze_btn = st.button("🚀 開始分析語意", type="primary")

# --- 邏輯處理 ---
if analyze_btn and user_input and api_key:
    try:
        client = OpenAI(api_key=api_key)
        
        # 1. 整理輸入文字 (去除空行)
        texts = [line.strip() for line in user_input.split('\n') if line.strip()]
        
        if len(texts) < 2:
            st.warning("請至少輸入兩行文字來進行比較！")
        else:
            with st.spinner('正在呼叫 OpenAI 進行 Embedding 轉換...'):
                # 2. 呼叫 OpenAI API 取得向量
                # 注意：為了簡化，這裡一次發送所有文本 (Batch request)
                model_id = model_name.replace("_", "-") # 轉換格式
                response = client.embeddings.create(
                    input=texts,
                    model=model_id
                )
                
                # 提取向量數據
                embeddings = [item.embedding for item in response.data]
                
            # 3. 計算餘弦相似度矩陣
            # 使用 scikit-learn 快速計算所有向量兩兩之間的相似度
            sim_matrix = cosine_similarity(embeddings)
            
            # 4. 轉成 DataFrame 以便展示
            df_sim = pd.DataFrame(sim_matrix, index=texts, columns=texts)

            # --- 展示結果 ---
            with col2:
                st.subheader("2. 分析結果 (熱圖)")
                # 使用 Seaborn 繪製熱圖
                fig, ax = plt.subplots(figsize=(8, 6))
                # 使用支援中文的字型設定可能會比較複雜，這裡主要看數值與顏色
                sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, ax=ax)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("3. 詳細數據矩陣")
            # 使用 Streamlit 的內建格式化功能，數值越大背景越深
            st.dataframe(df_sim.style.background_gradient(axis=None, cmap="Blues", vmin=0.5, vmax=1.0))
            
    except Exception as e:
        st.error(f"發生錯誤：{e}")
        st.error("請檢查您的 API Key 是否正確，或帳戶是否有額度。")

elif analyze_btn and not api_key:
    st.warning("⚠️ 請先在左側輸入 OpenAI API Key")