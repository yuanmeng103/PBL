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
set_background("../1.png")  # 这里写你的图片名

def load_model(model_filename):
    # 获取当前文件（即子页面 .py）的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上跳一级到根目录，然后进入 models 文件夹
    model_path = os.path.join(os.path.dirname(current_dir), "models", model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} 不存在")
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

# —— 调用函数 —— 
PBL_model = load_model("PBL_model.json")

# 全局样式：统一字体、大小、加粗，并缩小参数说明与输入框的间距
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'SimSun', 'Times New Roman', serif !important;
}

/* 平台标题 */
.stTitle {
    font-size: 32px !important;
    font-weight: bold !important;
}

/* 平台说明文字 */
.stMarkdown div[style*="line-height"] {
    font-size: 26px !important;
}

/* ---- 输入框区域 ---- */
input, select, textarea, label, div, span {
    font-family: 'Times New Roman', 'SimSun', serif !important;
    font-size: 30px !important;
}

/* 参数说明与输入框间距 */
.stNumberInput > label, .stMarkdown {
    margin-bottom: 2px !important;
}

/* 输入框内部间距缩小 */
.stNumberInput>div>div>div>input {
    font-size: 28px !important;    /* 匹配上面的大小 */
    height: 60px !important;       /* 字体变大后，高度也要相应增加，比如从 48px 改为 60px */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ---- selectbox 高度和宽度 ---- */
div[data-baseweb="select"] > div {
    min-height: 60px !important;  /* 增加高度 */
    width: 300px !important;      /* 增加宽度，防止文字挤压 */
}

div[data-baseweb="select"] input {
    font-size: 28px !important;   /* 增大字体 */
}

/* selectbox 显示区域字体和高度 */
div[data-baseweb="select"] input {
    font-size: 24px !important;   /* 字体大小 */
    height: 48px !important;      /* 高度 */
    padding: 6px 12px !important; /* 内部间距 */
}

/* 下拉选项字体大小 */
div[data-baseweb="select"] ul li {
    font-size: 24px !important;
}

/* ---- st.success 输出框内字体大小 ---- */
div[data-testid="stSuccess"] div[data-testid="stMarkdownContainer"] {
    font-size: 28px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# 平台标题
st.markdown("""
    <h1 style='text-align: center; line-height: 1.2;'>
        PBL连接件抗剪承载力预测平台<br>
        <span style='font-size: 26px; font-weight: normal;'>Prediction Platform for the Shear Bearing Capacity of PBL Connectors</span>
    </h1>
    """, unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(os.path.dirname(current_dir), "3.png")
with open(file_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

# --- 优雅布局 ---
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
    <div style="flex: 1; font-size: 24px; line-height: 1.8; text-align: justify; color: #333;">
        基于机器学习算法（XGBoost），结合482个推出试验和插入试验的数据库，
        部署为在线预测平台，
        该平台能够快速预测PBL连接件的抗剪承载力。
        用户只需输入几何与材料参数，即可获得预测结果。（注：试验类型：0-推出试验，1-插入试验；端部是否承压：0-端部不承压，1-端部承压。无贯穿钢筋时，<i>d</i><sub>s</sub> 和 <i>f</i><sub>sy</sub> 取 0）
    </div>
    <div style="flex: 0 0 260px; margin-left: 40px;">
        <img src="data:image/png;base64,{encoded}"
             style="width:100%; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("#### 输入参数")

# 辅助函数：统一标签样式（调小了字号到 20px 以适应并排，原 26px 太大会撑破布局）
# 修改后的函数定义，支持 4 个参数：名称、主符号、下标、单位
def label_html(text, symbol="", subscript="", unit=""):
    """
    text: 中文名称 (如: 开孔直径)
    symbol: 主符号 (如: d)
    subscript: 下标 (如: p)
    unit: 单位 (如: mm)
    """
    sub_str = f'<sub>{subscript}</sub>' if subscript else ""
    unit_str = f' <span style="font-style:normal;">({unit})</span>' if unit else ""
    return f'<p style="font-size:26px; margin-bottom:-10px;">{text} <i>{symbol}</i>{sub_str}{unit_str}</p>'

# --- 开启三列布局 ---
col1, col2, col3 = st.columns(3)

with col1:
    # 1. 开孔直径
    st.markdown(label_html("开孔直径", "d", "p", "mm"), unsafe_allow_html=True)
    dp = st.number_input("dp_val", 10.0, 100.0, 60.0, 0.1, key="pbl_dp", label_visibility="collapsed")

    # 2. 钢板厚度
    st.markdown(label_html("PBL板厚度", "t", "", "mm"), unsafe_allow_html=True)
    t = st.number_input("t_val", 5.0, 50.0, 20.0, 1.0, key="pbl_t", label_visibility="collapsed")

    # 3. 钢板屈服强度
    st.markdown(label_html("PBL板屈服强度", "f", "yp", "MPa"), unsafe_allow_html=True)
    fyp = st.number_input("fyp_val", 240.0, 460.0, 345.0, 0.1, key="pbl_fyp", label_visibility="collapsed")

    # 4. 混凝土抗压强度
    st.markdown(label_html("混凝土立方体抗压强度", "f", "cu", "MPa"), unsafe_allow_html=True)
    fcu = st.number_input("fcu_val", 20.0, 80.0, 50.0, 0.1, key="pbl_fcu", label_visibility="collapsed")

with col2:
    # 5. 开孔数量
    st.markdown(label_html("开孔数量", "n", "p", ""), unsafe_allow_html=True)
    n_p = st.number_input("n_val", 1.0, 10.0, 1.0, 1.0, key="pbl_np", label_visibility="collapsed")

    # 6. 钢板高度
    st.markdown(label_html("PBL板高度", "h", "p", "mm"), unsafe_allow_html=True)
    hp = st.number_input("hp_val", 80.0, 500.0, 150.0, 1.0, key="pbl_hp", label_visibility="collapsed")

    # 7. 混凝土弹模
    st.markdown(label_html("混凝土弹模", "E", "c", "GPa"), unsafe_allow_html=True)
    Ec = st.number_input("Ec_val", 15.0, 60.0, 30.0, 0.1, key="pbl_ec", label_visibility="collapsed")

    # 8. 钢筋屈服强度
    st.markdown(label_html("钢筋屈服强度", "f", "yr", "MPa"), unsafe_allow_html=True)
    fyr = st.number_input("fyr_val", 0.0, 500.0, 400.0, 0.1, key="pbl_fyr", label_visibility="collapsed")

with col3:
    # 9. 贯穿钢筋直径
    st.markdown(label_html("钢筋直径", "d", "r", "mm"), unsafe_allow_html=True)
    dr = st.number_input("ds_val", 0.0, 32.0, 20.0, 1.0, key="pbl_ds", label_visibility="collapsed")

    # 10. 试验类型 (直接把 Test Type 放在符号位)
    st.markdown(label_html("试验类型", "<span style='font-style:normal;'>Test Type</span>", "", ""), unsafe_allow_html=True)
    Test_Type = st.number_input("test_type_val", 0, 1, 0, 1, key="pbl_tt", label_visibility="collapsed")

    # 11. 端部是否承压 (直接把 Bearing Flag 放在符号位)
    st.markdown(label_html("端部是否承压", "<span style='font-style:normal;'> Bearing Flag</span>", "", ""), unsafe_allow_html=True)
    Bearing_Flag = st.number_input("Bearing_Flag_val", 0, 1, 0, 1, key="pbl_bf", label_visibility="collapsed")

st.write("---")
if st.button("计算抗剪承载力"):
    # 2. 定义模型训练时真正的特征名称 (必须和模型中的 np 一致)
    cols = ['dp', 'np', 't', 'hp', 'fyp', 'Ec', 'fcu', 'dr', 'fyr', 'Test_Type', 'Bearing_Flag']
    
    # 3. 将你的变量 n_p 对应到列名 'np' 上
    vals = [dp, n_p, t, hp, fyp, Ec, fcu, dr, fyr, Test_Type, Bearing_Flag]
    
    # 4. 构造 DataFrame，此时 columns 里的 'np' 会让模型找到它
    X_input = pd.DataFrame([vals], columns=cols)
    
    # 5. 预测
    y_pred = PBL_model.predict(X_input)[0]
    st.success(f"预测抗剪承载力: {y_pred:.2f} kN")