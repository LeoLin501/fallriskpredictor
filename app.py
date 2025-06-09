# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from io import BytesIO

# 設定中文字體
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# Streamlit 頁面設定
st.set_page_config(page_title="升級版 AI 跌倒風險預測", layout="wide")
st.title("🧠 升級版 AI 跌倒風險預測平台")

# Sidebar 步驟1
st.sidebar.header("步驟 1：上傳 Excel")
uploaded_file = st.sidebar.file_uploader("請選擇包含預測欄位與跌倒標籤的 Excel 檔案", type=[".xlsx"])

# 風險分數映射表
risk_score_mapping = {
    "利尿劑": 3,
    "麻醉止痛劑": 3,
    "緩瀉劑": 2,
    "鎮靜安眠藥": 3,
    "降血壓藥": 2,
    "降血糖藥": 2,
    "抗組織胺": 1,
    "肌肉鬆弛劑": 3,
    "抗憂鬱劑": 3
}

# 衛教建議映射表
risk_education_mapping = {
    "行動力": "加強平衡與下肢肌力訓練，安排復健課程，使用合適的輔具。",
    "跌倒史": "檢視居家與醫院環境，移除障礙物，鋪設防滑地墊，定期檢討跌倒事件。",
    "藥物風險分數": "請藥師進行藥物整合評估，避免多重鎮靜、降壓、利尿等高風險藥物交互作用。",
    "認知分類_輕度障礙": "加強用藥安全提醒，標示清楚藥盒，安排陪伴協助日常活動。",
    "認知分類_重度障礙": "安排 24 小時照護或安全看護，設置床邊護欄，避免單獨行動。",
    "是否獨居": "建議與家屬、社區資源聯繫，安排陪伴，安裝緊急呼叫系統。",
    "是否有家屬陪同": "鼓勵家屬學習照護技巧，共同參與病人跌倒預防教育。",
    "憂鬱症診斷": "評估心理狀態，提供心理支持或轉介心理諮商，促進正向生活型態。",
    "慢性疾病數量": "加強慢病管理，安排定期回診，注意疾病對行動力與認知的影響。",
    "環境風險分數": "改善居家與醫療環境安全，如良好照明、扶手設置、防滑設施。",
    "年齡": "加強年度跌倒風險篩檢，鼓勵定期健康檢查與功能評估。",
    "性別": "針對女性病人特別注意骨質疏鬆、營養補充，男性則注意心血管風險與活動安全。"
}


# 風險分類函數
def create_categorize_risk_fn(high, medium):
    def categorize_risk(prob):
        if prob >= high:
            return "高風險"
        elif prob >= medium:
            return "中風險"
        else:
            return "低風險"
    return categorize_risk

# 主流程
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.subheader("📋 原始資料預覽")
    st.dataframe(data.head())

    if "跌倒" not in data.columns:
        st.error("找不到 '跌倒' 欄位。請確認資料正確。")
    else:
        # 特徵處理
        data["性別"] = data["性別"].map({"男": 1, "女": 0})
        data["跌倒史"] = data["跌倒史"].astype(int)

        def cognitive_level(score):
            if score >= 24:
                return "完整"
            elif score >= 18:
                return "輕度障礙"
            else:
                return "重度障礙"

        data["認知分類"] = data["認知分數"].apply(cognitive_level)
        data = pd.get_dummies(data, columns=["認知分類"], drop_first=True)

        def calculate_drug_score(drugs):
            if pd.isna(drugs):
                return 0
            return sum([risk_score_mapping.get(d.strip(), 0) for d in str(drugs).split(",")])

        data["藥物風險分數"] = data["藥物"].apply(calculate_drug_score)

        for col in ["是否獨居", "是否有家屬陪同", "憂鬱症診斷", "慢性疾病數量", "環境風險分數"]:
            if col not in data.columns:
                data[col] = 0

        feature_cols = [
            "年齡", "性別", "行動力", "藥物風險分數", "跌倒史",
            "是否獨居", "是否有家屬陪同", "憂鬱症診斷", "慢性疾病數量", "環境風險分數"
        ] + [col for col in data.columns if col.startswith("認知分類_")]

        X = data[feature_cols]
        y = data["跌倒"]

        # SMOTE 平衡資料
        minority_class_count = X[y == 1].shape[0]
        safe_k = min(5, minority_class_count - 1) if minority_class_count > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=safe_k)

        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

        # Sidebar 步驟2
        st.sidebar.header("步驟 2：模型參數設定")
        model_choice = st.sidebar.selectbox("選擇模型", ("XGBoost", "Random Forest", "LightGBM" if LGBMClassifier else "Random Forest"))

        if st.sidebar.button("開始訓練模型"):
            # 模型選擇
            if model_choice == "XGBoost":
                model = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=300, max_depth=10)
            else:
                model = LGBMClassifier(class_weight='balanced', random_state=42)

            # 模型訓練
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            auc_score = roc_auc_score(y_test, y_proba)
            st.success(f"AUC 分數：{auc_score:.2f}")

            # ROC 曲線
            st.subheader("📉 ROC 曲線")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='darkorange')
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            # 混淆矩陣
            st.subheader("📊 混淆矩陣")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                        xticklabels=["無跌倒", "跌倒"],
                        yticklabels=["無跌倒", "跌倒"])
            ax_cm.set_xlabel("預測")
            ax_cm.set_ylabel("實際")
            ax_cm.set_title("混淆矩陣")
            st.pyplot(fig_cm)

            st.subheader("📌 特徵重要性圖")

            if model_choice == "Random Forest":
                st.info("隨機森林 → 使用 model.feature_importances_ 繪製特徵重要性 (SHAP 不相容 RF)")
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                plt.figure(figsize=(10, 6))
                plt.title("Feature Importances (Random Forest)")
                plt.bar(range(X_train.shape[1]), importances[indices])
                plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
                plt.tight_layout()
                st.pyplot(plt.gcf())

            else:
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test)

                st.info(f"{model_choice} → 使用 SHAP beeswarm 展示特徵重要性")
                plt.figure()
                shap.plots.beeswarm(shap_values, max_display=10)
                fig_shap = plt.gcf()
                st.pyplot(fig_shap)

            # 衛教建議
            st.subheader("📝 衛教建議")
            categorize_risk = create_categorize_risk_fn(0.75, 0.4)
            result_df = X_test.copy()
            result_df["實際"] = y_test.values
            result_df["預測機率"] = y_proba
            result_df["風險等級"] = result_df["預測機率"].apply(categorize_risk)


            # 進階衛教建議映射表 (移到函數外部)
            risk_education_mapping = {
                "行動力": "加強平衡與下肢肌力訓練，安排復健課程，使用合適的輔具。",
                "跌倒史": "檢視居家與醫院環境，移除障礙物，鋪設防滑地墊，定期檢討跌倒事件。",
                "藥物風險分數": "請藥師進行藥物整合評估，避免多重鎮靜、降壓、利尿等高風險藥物交互作用。",
                "認知分類_輕度障礙": "加強用藥安全提醒，標示清楚藥盒，安排陪伴協助日常活動。",
                "認知分類_重度障礙": "安排 24 小時照護或安全看護，設置床邊護欄，避免單獨行動。",
                "是否獨居": "建議與家屬、社區資源聯繫，安排陪伴，安裝緊急呼叫系統。",
                "是否有家屬陪同": "鼓勵家屬學習照護技巧，共同參與病人跌倒預防教育。",
                "憂鬱症診斷": "評估心理狀態，提供心理支持或轉介心理諮商，促進正向生活型態。",
                "慢性疾病數量": "加強慢病管理，安排定期回診，注意疾病對行動力與認知的影響。",
                "環境風險分數": "改善居家與醫療環境安全，如良好照明、扶手設置、防滑設施。",
                "年齡": "加強年度跌倒風險篩檢，鼓勵定期健康檢查與功能評估。",
                "性別": "針對女性病人特別注意骨質疏鬆、營養補充，男性則注意心血管風險與活動安全。"
            }
            # 進階衛教建議映射表
            advanced_education_mapping = {
                ("跌倒史",
                 "高風險"): "（高風險 + 跌倒史）該病患有反覆跌倒史，建議立即全面檢視環境與用藥，啟動跨團隊照護，並安排個別化防跌訓練。",
                ("跌倒史", "中風險"): "（中風險 + 跌倒史）病患有跌倒史，需加強家屬衛教與居家環境調整，建議 3 個月內追蹤。",
                ("跌倒史", "低風險"): "（低風險 + 跌倒史）病患曾有跌倒史，儘管目前低風險，應提醒持續留意行動安全。",
                ("行動力", "高風險"): "（高風險 + 行動力差）需安排復健與物理治療，並考慮使用步態輔具，嚴密監控活動安全。",
                ("認知分類_重度障礙", "高風險"): "（高風險 + 重度認知障礙）應安排 24 小時看護，強化環境保護措施，避免單獨行動，照護團隊應密切合作。"
            }


            # 定義 get_suggestion 函數
            def get_suggestion(feature_name, risk_level):
                advanced_key = (feature_name, risk_level)
                if advanced_key in advanced_education_mapping:
                    return advanced_education_mapping[advanced_key]

                base_suggestion = risk_education_mapping.get(feature_name, "建議一般健康促進與定期評估")
                if risk_level == "高風險":
                    return f"(高風險重點介入) {base_suggestion} 應立即與照護團隊討論加強監測方案，安排個別化的防跌措施，必要時啟動跨專業照護。"
                elif risk_level == "中風險":
                    return f"(中風險持續監測) {base_suggestion} 建議於未來 3 個月內安排至少一次防跌評估，並與家屬共同檢視居家與照護環境，預防風險提升。"
                else:
                    return f"(低風險健康促進) {base_suggestion} 目前屬低風險，應維持良好生活型態，定期檢視環境安全及功能狀態，預防風險變化。"


            # 展示高、中、低風險 → 用 st.expander 分區
            export_edu_list = []
            for risk_level in ["高風險", "中風險", "低風險"]:
                cases = result_df[result_df["風險等級"] == risk_level]
                if not cases.empty:
                    with st.expander(f"{risk_level} 個案 ({len(cases)})"):
                        for idx, row in cases.iterrows():
                            st.write(f"**個案編號：{idx}**")

                            if model_choice == "Random Forest":
                                # RF → 用 feature_importances_ 排前3名
                                top_features_idx = np.argsort(model.feature_importances_)[::-1][:3]
                                top_features = X_train.columns[top_features_idx]
                                for feature_name in top_features:
                                    suggestion = get_suggestion(feature_name, risk_level)
                                    st.write(f"- 針對【{feature_name}】：{suggestion}")

                            else:
                                # XGB/LGB → 用 SHAP per-case
                                case_position = list(X_test.index).index(idx)
                                shap_values_case = shap_values[case_position]
                                top_features = np.argsort(np.abs(shap_values_case.values))[-3:][::-1]
                                for feature_idx in top_features:
                                    feature_name = X_test.columns[feature_idx]
                                    suggestion = get_suggestion(feature_name, risk_level)
                                    st.write(f"- 針對【{feature_name}】：{suggestion}")


            # 下載報表功能
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)
                return output


            st.download_button("下載結果 (全部)", data=to_excel(result_df), file_name="fall_risk_predictions.xlsx")
            high_risk_cases = result_df[result_df["風險等級"] == "高風險"]
            st.download_button("下載高風險名單", data=to_excel(high_risk_cases), file_name="high_risk_cases.xlsx")

            if export_edu_list:
                df_edu = pd.DataFrame(export_edu_list)
                st.download_button("下載個案衛教建議報表", data=to_excel(df_edu), file_name="fall_risk_education.xlsx")
