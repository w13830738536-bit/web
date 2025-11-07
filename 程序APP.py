import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征范围定义（根据需求更新）
feature_ranges = {
    # 分类变量
    "Subjective Chewing Difficulty": {
        "type": "categorical",
        "options": [0, 1],
        "labels": ["0=No", "1=Yes"],  # 显示选项含义
        "default": 0
    },
    "Dentures": {
        "type": "categorical",
        "options": [0, 1],
        "labels": ["0=No", "1=Yes"],
        "default": 0
    },
    "Oral Health-related Self-efficacy": {
        "type": "categorical",
        "options": [0, 1],
        "labels": ["0=＞50分", "1=≤50分"],
        "default": 0
    },
    "Physical Frailty": {
        "type": "categorical",
        "options": [0, 1],
        "labels": ["0=No", "1=Yes"],
        "default": 0
    },
    "Social Support": {
        "type": "categorical",
        "options": [1, 2, 3],
        "labels": ["1=High", "2=Medium", "3=Low"],
        "default": 1
    },
    "Dietary Diversity": {
        "type": "categorical",
        "options": [1, 2, 3],
        "labels": ["1=High", "2=Medium", "3=Low"],
        "default": 1
    },
    # 连续变量（整数）
    "Oral Health Status": {
        "type": "numerical",
        "min": 0,
        "max": 20,
        "default": 0,
        "step": 1  # 确保输入为整数
    }
}

# Streamlit 界面
st.title("Oral Frailty Prediction Model")  # 更新标题为口腔衰弱预测

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=int(properties["min"]),  # 整数处理
            max_value=int(properties["max"]),
            value=int(properties["default"]),
            step=properties.get("step", 1)
        )
    elif properties["type"] == "categorical":
        # 显示带含义的选项标签
        value = st.selectbox(
            label=f"{feature}",
            options=properties["options"],
            format_func=lambda x: properties["labels"][properties["options"].index(x)]
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测功能
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，更新为口腔衰弱相关文本
    text = f"Based on feature values, predicted possibility of Oral Frailty is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)

    st.image("prediction_text.png")
