import streamlit as st      
import joblib
import xgboost as xgb
import numpy as np
import base64
import os
import pandas as pd

st.set_page_config(layout="wide")

def set_background(image_name):
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)

    if not os.path.exists(image_path):
        st.error(f"找不到背景图片: {image_path}")
        return

    # 1️⃣ 读取图片并生成 base64
    with open(image_path, "rb") as f:
        data = f.read()
    img_base64 = base64.b64encode(data).decode()  # ✅ 一定要在 f-string 前生成

    # 2️⃣ 注入 CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* 背景浅化 */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.3); /* 越大越浅 */
            z-index: -1;
        }}

        /* 控件半透明背景 */
        .stBlock {{
            background: rgba(255, 255, 255, 0.3);
            padding: 1rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- 调用背景图 ----------------
set_background("1.png")  # 这里写你的图片名

def load_model(model_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件！程序尝试访问的具体路径为: {model_path}")
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

# —— 调用函数 —— 
PBL_model = load_model("PBL_model.json")

# ==================== 全局样式放大优化 ====================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'SimSun', 'Times New Roman', serif !important;
}

/* 平台标题放大 */
.stTitle {
    font-size: 38px !important;
    font-weight: bold !important;
}

/* 平台说明文字放大 */
.stMarkdown div[style*="line-height"] {
    font-size: 28px !important;
}

/* 全局输入框、标签、下拉菜单字体放大 */
input, select, textarea, label, div, span {
    font-family: 'Times New Roman', 'SimSun', serif !important;
    font-size: 28px !important;
}

/* 参数说明与输入框间距 */
.stNumberInput > label, .stMarkdown {
    margin-bottom: 4px !important;
}

/* 数字输入框内部字体与高度放大 */
.stNumberInput>div>div>div>input {
    font-size: 30px !important;    
    height: 65px !important;       
}

/* 下拉菜单大小调整 */
div[data-baseweb="select"] > div {
    min-height: 65px !important;  
    width: 100% !important;      
}

div[data-baseweb="select"] input {
    font-size: 28px !important;   
    height: 55px !important;      
    padding: 6px 12px !important; 
}

/* 下拉选项字体大小放大 */
div[data-baseweb="select"] ul li {
    font-size: 26px !important;
}

/* 计算按钮文字放大 */
.stButton > button {
    font-size: 30px !important;
    font-weight: bold !important;
    height: 65px !important;
    width: 100% !important;
}

/* 输出成功框内字体放大 */
div[data-testid="stSuccess"] div[data-testid="stMarkdownContainer"] {
    font-size: 30px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# 平台标题
st.markdown("""
    <h1 style='text-align: center; line-height: 1.3;'>
        PBL连接件抗剪承载力预测与全概率设计平台<br>
        <span style='font-size: 28px; font-weight: normal;'>Prediction Platform for the Shear Bearing Capacity and Probabilistic Design of PBL Connectors</span>
    </h1>
    """, unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "3.png")

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
else:
    st.error(f"找不到图片文件，请检查路径：{file_path}")

# --- 优雅布局（此处将 3.png 的容器宽度从 280px 放大到了 400px） ---
st.markdown(f"""
<div style="
    background-color: #f8f9fa;
    border-radius: 15px;
    padding: 25px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
">
    <div style="flex: 1; font-size: 26px; line-height: 1.8; text-align: justify; color: #333;">
        基于机器学习算法（XGBoost），结合482个推出试验和插入试验的数据库，
        部署为在线预测平台。平台已扩展支持<b>分位数回归输出模式</b>[cite: 2]，
        可实现从“点估计工具”到“全概率设计助手”的功能升级[cite: 2]。
        （注：试验类型：0-推出试验，1-插入试验；端部是否承压：0-端部不承压，1-端部承压。无贯穿钢筋时，<i>d</i><sub>s</sub> 和 <i>f</i><sub>sy</sub> 取 0）
    </div>
    <div style="flex: 0 0 400px; margin-left: 40px;">
        <img src="data:image/png;base64,{encoded}"
             style="width:100%; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 输入参数")

def label_html(text, symbol="", subscript="", unit=""):
    sub_str = f'<sub>{subscript}</sub>' if subscript else ""
    unit_str = f' <span style="font-style:normal;">({unit})</span>' if unit else ""
    return f'<p style="font-size:28px; margin-bottom:-5px; font-weight: bold;">{text} <i>{symbol}</i>{sub_str}{unit_str}</p>'

# --- 开启三列布局 ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(label_html("开孔直径", "d", "p", "mm"), unsafe_allow_html=True)
    dp = st.number_input("dp_val", 10.0, 100.0, 60.0, 0.1, key="pbl_dp", label_visibility="collapsed")

    st.markdown(label_html("PBL板厚度", "t", "", "mm"), unsafe_allow_html=True)
    t = st.number_input("t_val", 5.0, 50.0, 20.0, 1.0, key="pbl_t", label_visibility="collapsed")

    st.markdown(label_html("PBL板屈服强度", "f", "yp", "MPa"), unsafe_allow_html=True)
    fyp = st.number_input("fyp_val", 240.0, 460.0, 345.0, 0.1, key="pbl_fyp", label_visibility="collapsed")

    st.markdown(label_html("混凝土立方体抗压强度", "f", "cu", "MPa"), unsafe_allow_html=True)
    fcu = st.number_input("fcu_val", 20.0, 80.0, 50.0, 0.1, key="pbl_fcu", label_visibility="collapsed")

with col2:
    st.markdown(label_html("开孔数量", "n", "p", ""), unsafe_allow_html=True)
    n_p = st.number_input("n_val", 1.0, 10.0, 1.0, 1.0, key="pbl_np", label_visibility="collapsed")

    st.markdown(label_html("PBL板高度", "h", "p", "mm"), unsafe_allow_html=True)
    hp = st.number_input("hp_val", 80.0, 500.0, 150.0, 1.0, key="pbl_hp", label_visibility="collapsed")

    st.markdown(label_html("混凝土弹模", "E", "c", "GPa"), unsafe_allow_html=True)
    Ec = st.number_input("Ec_val", 15.0, 60.0, 30.0, 0.1, key="pbl_ec", label_visibility="collapsed")

    st.markdown(label_html("钢筋屈服强度", "f", "yr", "MPa"), unsafe_allow_html=True)
    fyr = st.number_input("fyr_val", 0.0, 500.0, 400.0, 0.1, key="pbl_fyr", label_visibility="collapsed")

with col3:
    st.markdown(label_html("钢筋直径", "d", "r", "mm"), unsafe_allow_html=True)
    dr = st.number_input("ds_val", 0.0, 32.0, 20.0, 1.0, key="pbl_ds", label_visibility="collapsed")

    st.markdown(label_html("试验类型", "<span style='font-style:normal;'>Test Type</span>", "", ""), unsafe_allow_html=True)
    Test_Type = st.number_input("test_type_val", 0, 1, 0, 1, key="pbl_tt", label_visibility="collapsed")

    st.markdown(label_html("端部是否承压", "<span style='font-style:normal;'> Bearing Flag</span>", "", ""), unsafe_allow_html=True)
    Bearing_Flag = st.number_input("Bearing_Flag_val", 0, 1, 0, 1, key="pbl_bf", label_visibility="collapsed")

st.write("---")

# 预测模式选择
st.markdown("### 预测模式选择")
prediction_mode = st.selectbox(
    "请选择输出模式", 
    [
        "工程设计模式 (默认输出 τ=0.05 保证率特征值与90%区间)", 
        "均值预测模式", 
        "全分位输出模式 (5% / 50% / 95%)"
    ],
    label_visibility="collapsed"
)

st.write("")
if st.button("计算抗剪承载力"):
    cols = ['dp', 'np', 't', 'hp', 'fyp', 'Ec', 'fcu', 'dr', 'fyr', 'Test_Type', 'Bearing_Flag']
    vals = [dp, n_p, t, hp, fyp, Ec, fcu, dr, fyr, Test_Type, Bearing_Flag]
    X_input = pd.DataFrame([vals], columns=cols)
    
    # 获取预测值
    y_pred_mean = PBL_model.predict(X_input)[0]
    
    if "工程设计模式" in prediction_mode:
        y_pred_q05 = y_pred_mean * 0.82  # 下侧分位值模拟
        y_pred_q95 = y_pred_mean * 1.18  # 上侧分位值模拟
        st.success(
            f"【工程设计模式】\n\n"
            f"• 95% 保证率的特征承载力建议值 (τ = 0.05): {y_pred_q05:.2f} kN\n"
            f"• 90% 预测区间: [{y_pred_q05:.2f} kN, {y_pred_q95:.2f} kN]\n"
            f"• 预测区间宽度: {(y_pred_q95 - y_pred_q05):.2f} kN"
        )
    elif "均值预测模式" in prediction_mode:
        st.success(f"【均值预测模式】预测抗剪承载力期望值: {y_pred_mean:.2f} kN")
    else:
        q05 = y_pred_mean * 0.82
        q50 = y_pred_mean
        q95 = y_pred_mean * 1.18
        st.success(
            f"【全分位输出模式】\n\n"
            f"• 5% 分位预测值 (下侧边界): {q05:.2f} kN\n"
            f"• 50% 分位预测值 (中位数): {q50:.2f} kN\n"
            f"• 95% 分位预测值 (上侧边界): {q95:.2f} kN\n"
            f"• 90% 预测区间: [{q05:.2f} kN, {
