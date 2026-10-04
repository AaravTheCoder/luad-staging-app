import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io

def gemini_summary(prob, stage, confidence, mode_label, age, sex, smoking, symptoms, history):
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return None, "GEMINI_API_KEY not found in secrets"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        prompt = f"""You are a clinical AI assistant summarizing a lung adenocarcinoma staging model result for a research demo.
Write 2-3 concise sentences in a neutral clinical tone. Do not diagnose. End with a note that this is for research only.

Patient: {age}-year-old {sex.lower()}, {smoking.lower()} smoker. Symptoms: {symptoms}. History: {history}.
Model output: {stage} (P={prob:.3f}, {confidence*100:.1f}% confidence). Mode: {mode_label}.

Summary:"""
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, str(e)

st.set_page_config(
    page_title="LUAD Staging AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
body, .stApp { background-color: #0f1117; color: #eee; }
.block-container { padding: 2rem 3rem; }
h1 { color: #3498db; font-size: 2.2rem; }
h2, h3 { color: #ccc; }
.result-box {
    border-radius: 16px; padding: 2rem; text-align: center;
    margin-top: 1rem;
}
.early  { background: linear-gradient(135deg, #1a3a2a, #1e5c3a); border: 2px solid #2ecc71; }
.late   { background: linear-gradient(135deg, #3a1a1a, #5c1e1e); border: 2px solid #e74c3c; }
.metric-card {
    background: #1a1d27; border-radius: 12px; padding: 1.2rem;
    text-align: center; margin: 0.5rem 0;
}
.metric-value { font-size: 2rem; font-weight: bold; color: #3498db; }
.metric-label { font-size: 0.85rem; color: #aaa; margin-top: 0.3rem; }
.warning-box {
    background: #2a2200; border: 1px solid #f39c12; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.85rem; color: #f39c12; margin-top: 0.5rem;
}
.stButton > button {
    background: #3498db; color: white; border: none;
    border-radius: 8px; padding: 0.6rem 2rem;
    font-size: 1rem; font-weight: bold; width: 100%;
}
.stButton > button:hover { background: #2980b9; }
div[data-testid="stFileUploader"] { background: #1a1d27; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🫁 LUAD Staging AI")
st.markdown("**Multimodal deep learning for lung adenocarcinoma staging** · CT scan + RNA-seq fusion")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown("### 🩻 CT Scan <span style='font-size:0.75rem;color:#aaa'>(optional)</span>", unsafe_allow_html=True)
    ct_file = st.file_uploader("Upload chest CT slice", type=["png", "jpg", "jpeg", "dcm"],
                                label_visibility="collapsed")
    if ct_file:
        try:
            img_bytes = ct_file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(pil_img, caption="Uploaded CT slice", use_container_width=True)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            pil_img = None
    else:
        st.info("Upload a 2D axial CT slice (PNG / JPG) — or run genomic-only")
        pil_img = None

with col2:
    st.markdown("### 🧬 RNA-seq Expression <span style='font-size:0.75rem;color:#aaa'>(optional)</span>", unsafe_allow_html=True)
    rna_file = st.file_uploader("Upload RNA-seq CSV", type=["csv", "tsv"],
                                 label_visibility="collapsed")
    rna_dict = {}
    if rna_file:
        try:
            sep = "\t" if rna_file.name.endswith(".tsv") else ","
            rna_df = pd.read_csv(rna_file, sep=sep, index_col=0)
            if rna_df.shape[0] == 1:
                rna_dict = rna_df.iloc[0].to_dict()
            elif rna_df.shape[1] == 1:
                rna_dict = rna_df.iloc[:, 0].to_dict()
            else:
                rna_dict = rna_df.iloc[0].to_dict()
            st.success(f"Loaded {len(rna_dict):,} gene expression values")
            st.caption("Format: genes as columns (or rows), one patient per file")
        except Exception as e:
            st.error(f"Could not parse RNA-seq file: {e}")
    else:
        st.info("Upload RNA-seq CSV — or run imaging-only")
        st.caption("Expected: ~18,514 genes matching TCGA/CPTAC naming (HGNC symbols)")

with col3:
    st.markdown("### 👤 Patient Info")
    age     = st.slider("Age", 30, 90, 62)
    sex     = st.radio("Sex", ["Male", "Female"], horizontal=True)
    smoking = st.selectbox("Smoking history", ["Never", "Former", "Current"])
    symptoms = st.selectbox("Presenting symptoms", [
        "None",
        "Mild (occasional cough)",
        "Persistent (chronic cough, fatigue)",
        "Severe (hemoptysis, weight loss, dyspnea)"
    ], index=2)
    history = st.selectbox("Relevant medical history", [
        "None relevant",
        "Family history of lung cancer",
        "Occupational exposure (asbestos, radon)",
        "Prior pulmonary condition"
    ], index=2)
    st.caption("Patient info is displayed alongside the prediction for clinical context. It does not affect the model output.")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    predict_btn = st.button("🔬 Run Staging Prediction")

if predict_btn:
    if pil_img is None and not rna_dict:
        st.error("Please upload at least a CT scan or an RNA-seq file.")
    else:
        # Determine mode
        has_ct  = pil_img is not None
        has_rna = bool(rna_dict)
        if has_ct and has_rna:
            mode_label = "🔬 Multimodal (CT + RNA-seq)"
            mode_color = "#3498db"
        elif has_ct:
            mode_label = "🩻 Imaging Only (CT)"
            mode_color = "#9b59b6"
        else:
            mode_label = "🧬 Genomic Only (RNA-seq)"
            mode_color = "#2ecc71"

        # Fill missing modality with blank inputs
        if not has_ct:
            pil_img = Image.new("RGB", (224, 224), color=0)
        if not has_rna:
            rna_dict = {}

        with st.spinner("Running model inference..."):
            try:
                from inference import predict
                result = predict(pil_img, rna_dict)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        prob       = result["probability"]
        stage      = result["stage"]
        confidence = result["confidence"]
        cam        = result["gradcam"]
        attribution = result["attribution"]
        missing    = result["missing_genes"]
        total      = result["total_genes"]

        st.markdown("## Results")
        st.markdown(f'<div style="margin-bottom:0.8rem"><span style="background:#1a1d27;border:1px solid {mode_color};border-radius:20px;padding:0.3rem 1rem;font-size:0.9rem;color:{mode_color}">{mode_label}</span></div>', unsafe_allow_html=True)

        # ── Summary box ──────────────────────────────────────────────────────
        box_class = "late" if prob >= 0.5 else "early"
        color     = "#e74c3c" if prob >= 0.5 else "#2ecc71"
        emoji     = "🔴" if prob >= 0.5 else "🟢"
        st.markdown(f"""
        <div class="result-box {box_class}">
            <div style="font-size:3.5rem">{emoji}</div>
            <div style="font-size:1.8rem; font-weight:bold; color:{color}; margin:0.5rem 0">{stage}</div>
            <div style="font-size:1rem; color:#ccc">P(Stage III/IV) = <b style="color:{color}">{prob:.3f}</b> &nbsp;·&nbsp; Confidence: <b>{confidence*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        if has_rna and missing > total * 0.3:
            st.markdown(f"""<div class="warning-box">
            ⚠️ {missing:,} of {total:,} genes were missing from your RNA-seq file and were zero-filled.
            Predictions may be less reliable. Ensure gene names use HGNC symbols (e.g. EGFR, KRAS, TP53).
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Metric cards ─────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{prob:.3f}</div><div class="metric-label">P(Stage III/IV)</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence*100:.1f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{age}</div><div class="metric-label">Patient Age</div></div>', unsafe_allow_html=True)
        with m4:
            genes_used = (total - missing) if has_rna else "—"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{genes_used if isinstance(genes_used, str) else f"{genes_used:,}"}</div><div class="metric-label">Genes Matched</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Grad-CAM + Genomic attribution ───────────────────────────────────
        show_cam   = has_ct
        show_genes = has_rna and np.any(attribution != 0)

        if show_cam or show_genes:
            viz1, viz2 = st.columns(2) if (show_cam and show_genes) else (st.columns(1)[0], None), None
            if show_cam and show_genes:
                viz1, viz2 = st.columns(2)
            elif show_cam:
                viz1 = st.container()
                viz2 = None
            else:
                viz1 = None
                viz2 = st.container()

            if show_cam and viz1:
                with viz1:
                    st.markdown("#### 🔥 Grad-CAM Saliency (ResNet-18 layer4)")
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    fig.patch.set_facecolor("#0f1117")
                    img_arr = np.array(pil_img.resize((224, 224)))
                    axes[0].imshow(img_arr, cmap="gray")
                    axes[0].set_title("Input CT", color="white", fontsize=10)
                    axes[0].axis("off")
                    axes[1].imshow(img_arr, cmap="gray")
                    axes[1].imshow(cam, cmap="jet", alpha=0.45)
                    axes[1].set_title("Grad-CAM", color="white", fontsize=10)
                    axes[1].axis("off")
                    for ax in axes:
                        ax.set_facecolor("#0f1117")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption("Red = regions most influential to the prediction")

            if show_genes and viz2:
                with viz2:
                    st.markdown("#### 🧬 Top Genomic Dimensions (grad × input)")
                    top_idx  = np.argsort(np.abs(attribution))[::-1][:12]
                    top_vals = attribution[top_idx]
                    colors   = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    fig2.patch.set_facecolor("#0f1117")
                    ax2.set_facecolor("#0f1117")
                    ax2.barh([f"PC dim {i}" for i in top_idx[::-1]], top_vals[::-1], color=colors[::-1])
                    ax2.axvline(0, color="white", linewidth=0.8, linestyle="--")
                    ax2.set_xlabel("Attribution (grad × input)", color="#aaa")
                    ax2.tick_params(colors="white")
                    for spine in ax2.spines.values():
                        spine.set_edgecolor("#333")
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                    st.caption("Red = pushes toward Stage III/IV · Blue = pushes toward Stage I/II")

        # ── Gemini clinical summary ───────────────────────────────────────────
        st.markdown("---")
        with st.spinner("Generating AI clinical summary..."):
            summary, err = gemini_summary(prob, stage, confidence, mode_label, age, sex, smoking, symptoms, history)
        if summary:
            st.markdown(f"""
            <div style="background:#1a1d27;border-left:3px solid #3498db;border-radius:8px;
                        padding:1rem 1.2rem;margin-top:0.5rem;color:#ccc;font-size:0.95rem;line-height:1.6">
                <div style="font-size:0.75rem;color:#3498db;font-weight:bold;margin-bottom:0.4rem;letter-spacing:0.05em">
                    AI CLINICAL SUMMARY
                </div>
                {summary}
            </div>
            """, unsafe_allow_html=True)
        elif err:
            st.caption(f"_AI summary unavailable: {err}_")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.85rem">
<a href="https://github.com/AaravTheCoder/luad-staging-app" style="color:#3498db">GitHub</a>
&nbsp;·&nbsp; ResNet-18 (RadiologyNET) + ComBat + PCA · 3 cohorts · 5×5-fold CV
</div>
""", unsafe_allow_html=True)
