"""Diabetes prediction app.

Place this file next to diabetes.csv and run: streamlit run diabetes.py
Models are trained once per configuration and reused on Streamlit reruns.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parent / "diabetes.csv"
CLASS_NAMES = {0: "Non-Diabetes (無糖尿病)", 1: "Diabetes (糖尿病)"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Outcome" not in df.columns:
        raise ValueError("diabetes.csv 缺少 Outcome 欄位")
    if df.isna().any().any():
        raise ValueError("diabetes.csv 含有缺失值，請先處理")
    if set(df["Outcome"].unique()) != {0, 1}:
        raise ValueError("Outcome 必須包含 0 與 1 兩個類別")
    return df


@st.cache_resource(show_spinner="首次使用此設定，正在訓練模型…")
def get_trained_model(classifier, train_ratio, parameters, data_fingerprint):
    """Train once per model/split/parameter/data combination and cache the fitted model."""
    df = load_data()
    x = df.drop(columns="Outcome")
    y = df["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=train_ratio,
        random_state=42,
        stratify=y,
    )

    if classifier == "Decision Tree":
        criterion, splitter = parameters
        model = DecisionTreeClassifier(
            criterion=criterion, splitter=splitter, random_state=42
        )
    elif classifier == "Logistic Regression":
        c_value, max_iter = parameters
        model = LogisticRegression(
            C=c_value, max_iter=max_iter, penalty="l2", random_state=42
        )
    elif classifier == "Random Forest":
        n_estimators, max_depth, bootstrap = parameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"未知模型：{classifier}")

    model.fit(x_train, y_train)
    return model, x_test, y_test, len(x_train)


def show_evaluation(model, x_test, y_test, selected_plots):
    y_pred = model.predict(x_test)
    st.subheader("模型評估 Model Evaluation")
    columns = st.columns(4)
    scores = [
        ("Accuracy", accuracy_score(y_test, y_pred)),
        ("Precision", precision_score(y_test, y_pred, zero_division=0)),
        ("Recall", recall_score(y_test, y_pred, zero_division=0)),
        ("F1 Score", f1_score(y_test, y_pred, zero_division=0)),
    ]
    for column, (name, score) in zip(columns, scores):
        column.metric(name, f"{score:.3f}")

    if "Confusion Matrix" in selected_plots:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=["Non-Diabetes", "Diabetes"],
            cmap="Blues",
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

    if "ROC Curve" in selected_plots:
        st.subheader("ROC Curve")
        class_one_index = list(model.classes_).index(1)
        y_probability = model.predict_proba(x_test)[:, class_one_index]
        fpr, tpr, _ = roc_curve(y_test, y_probability)
        area = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {area:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)


def show_single_prediction(model, features):
    st.subheader("單筆資料預測 Single Prediction")
    st.caption("輸入單一個案的 8 項特徵，使用上方所選的已快取模型進行預測。")
    with st.form("single_prediction_form"):
        values = {}
        left, right = st.columns(2)
        for index, feature in enumerate(features.columns):
            series = features[feature]
            default = float(series.median())
            is_integer = pd.api.types.is_integer_dtype(series)
            with (left if index % 2 == 0 else right):
                if is_integer:
                    values[feature] = st.number_input(
                        feature, min_value=0, value=max(0, int(round(default))), step=1
                    )
                else:
                    values[feature] = st.number_input(
                        feature, min_value=0.0, value=max(0.0, round(default, 3)),
                        step=0.1, format="%.3f"
                    )
        submitted = st.form_submit_button("預測 Predict", type="primary")

    if submitted:
        one_record = pd.DataFrame([values], columns=features.columns)
        predicted_class = int(model.predict(one_record)[0])
        probabilities = model.predict_proba(one_record)[0]
        probability_by_class = {
            int(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
        st.success(f"預測結果：{CLASS_NAMES[predicted_class]}")
        st.write(
            f"無糖尿病機率：**{probability_by_class[0]:.2%}** ｜ "
            f"糖尿病機率：**{probability_by_class[1]:.2%}**"
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [probability_by_class[0], probability_by_class[1]],
            labels=["Non-Diabetes", "Diabetes"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#75B9E7", "#F39C87"],
        )
        ax.axis("equal")
        ax.set_title("Single Prediction Probability Distribution")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("此結果僅供機器學習展示，不可作為醫療診斷依據。")


def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("Predicting Diabetes Web App")
    st.markdown("糖尿病預測與模型效能比較")

    try:
        df = load_data()
    except (OSError, ValueError) as exc:
        st.error(f"無法載入 diabetes.csv：{exc}")
        st.stop()

    features = df.drop(columns="Outcome")
    st.sidebar.title("Model Selection Panel")
    classifier = st.sidebar.selectbox(
        "Classifier / 選擇模型", ("Decision Tree", "Logistic Regression", "Random Forest")
    )
    train_percent = st.sidebar.slider(
        "訓練資料比例 Training Data (%)", min_value=50, max_value=90,
        value=70, step=5
    )
    st.sidebar.caption(f"訓練集：{train_percent}% ｜ 測試集：{100 - train_percent}%")

    st.sidebar.subheader("Model Parameters")
    if classifier == "Decision Tree":
        criterion = st.sidebar.radio("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.radio("Splitter", ("best", "random"))
        parameters = (criterion, splitter)
    elif classifier == "Logistic Regression":
        c_value = st.sidebar.number_input("C (Regularization)", min_value=0.01,
                                          max_value=10.0, value=1.0, step=0.01)
        max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, 500, step=50)
        parameters = (float(c_value), int(max_iter))
    else:
        n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, step=50)
        max_depth = st.sidebar.slider("Maximum depth", 1, 20, 5)
        bootstrap = st.sidebar.checkbox("Bootstrap", value=True)
        parameters = (n_estimators, max_depth, bootstrap)

    selected_plots = st.sidebar.multiselect(
        "Select evaluation plots",
        ("Confusion Matrix", "ROC Curve"),
        default=("Confusion Matrix", "ROC Curve"),
    )
    if st.sidebar.checkbox("Show raw data", value=False):
        st.subheader("Diabetes Raw Dataset")
        st.dataframe(df, use_container_width=True)

    # File metadata invalidates cached fitted models if the data changes.
    stat = DATA_PATH.stat()
    data_fingerprint = (stat.st_size, stat.st_mtime_ns)
    model, x_test, y_test, train_count = get_trained_model(
        classifier, train_percent / 100.0, parameters, data_fingerprint
    )
    st.info(
        f"目前模型：{classifier} ｜ 訓練資料 {train_count} 筆 ｜ "
        f"測試資料 {len(x_test)} 筆。相同設定會直接使用快取模型。"
    )
    show_evaluation(model, x_test, y_test, selected_plots)
    st.divider()
    show_single_prediction(model, features)


if __name__ == "__main__":
    main()
