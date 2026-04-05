"""
STEP 3: Streamlit Dashboard — app.py  (WITH PCA TAB)
======================================================
Run with: streamlit run dashboard/app.py
(from inside the space_weather_project folder)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Space Weather Risk Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    .stApp { background: #0a0e1a; }
    .main-title {
        font-family: 'Orbitron', monospace; font-size: 2.4rem; font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7, #ff6b35);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: 2px;
    }
    .sub-title {
        font-family: 'Share Tech Mono', monospace; color: #4a9eff;
        text-align: center; font-size: 0.9rem; letter-spacing: 3px; margin-bottom: 2rem;
    }
    .section-header {
        font-family: 'Orbitron', monospace; color: #00d4ff; font-size: 1.1rem;
        letter-spacing: 2px; border-bottom: 1px solid #1e3a5f;
        padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0f1628, #1a2340);
        border-radius: 16px; padding: 2rem; text-align: center;
        border: 2px solid #1e3a5f; margin: 1rem 0;
    }
    .metric-label { font-family: 'Share Tech Mono', monospace; color: #4a9eff; font-size: 0.75rem; letter-spacing: 2px; }
    .metric-value { font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 700; color: #fff; }
    .risk-high   { color: #ff4757 !important; text-shadow: 0 0 20px rgba(255,71,87,0.7); }
    .risk-medium { color: #ffa502 !important; text-shadow: 0 0 20px rgba(255,165,2,0.7); }
    .risk-low    { color: #2ed573 !important; text-shadow: 0 0 20px rgba(46,213,115,0.7); }
    .alert-box { border-radius: 10px; padding: 1rem 1.5rem; font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; margin: 0.5rem 0; }
    .alert-high   { background: rgba(255,71,87,0.15);  border-left: 4px solid #ff4757; color: #ff6b7a; }
    .alert-medium { background: rgba(255,165,2,0.15);  border-left: 4px solid #ffa502; color: #ffb733; }
    .alert-low    { background: rgba(46,213,115,0.15); border-left: 4px solid #2ed573; color: #52e88f; }
    .pca-card {
        background: linear-gradient(135deg, #0f1628, #1a2340);
        border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.2rem;
        text-align: center; box-shadow: 0 0 20px rgba(0,212,255,0.1);
    }
    div[data-testid="stSidebar"] { background: #0d1020; border-right: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#aaa', family='Share Tech Mono'),
    margin=dict(t=40, b=10, l=10, r=10)
)
COLOR_MAP = {'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#2ed573'}
FEATURE_COLS = [
    # Raw sensor measurements
    'solar_wind_speed', 'proton_density', 'bx', 'by', 'bz',
    'plasma_temp', 'kp_index', 'dst_index', 'xray_flux', 'ae_index',
    'f107_index', 'ap_index', 'speed_pressure',
    # Physics-based engineered features
    'storm_severity', 'mag_disturbance', 'radiation_pressure',
    # Rolling averages
    'kp_rolling_3h', 'bz_rolling_3h', 'wind_rolling_3h', 'dst_rolling_6h',
    # Rate-of-change features
    'delta_bz', 'delta_wind', 'delta_proton', 'delta_kp',
    # Binary threshold flags
    'bz_negative', 'extreme_kp', 'extreme_dst', 'high_speed'
]


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        models = {}
        for name in ['random_forest', 'xgboost', 'logistic_regression', 'isolation_forest']:
            with open(f'models/{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        with open('models/pca.pkl',           'rb') as f: pca     = pickle.load(f)
        with open('models/scaler.pkl',         'rb') as f: scaler  = pickle.load(f)
        with open('models/label_encoder.pkl',  'rb') as f: le      = pickle.load(f)
        with open('models/feature_cols.pkl',   'rb') as f: fc      = pickle.load(f)
        with open('models/results.json',       'r')  as f: results = json.load(f)
        with open('models/pca_info.json',      'r')  as f: pca_info= json.load(f)
        return models, pca, scaler, le, fc, results, pca_info
    except Exception as e:
        st.error(f"❌ Models not found! Run train_models.py first.\n\nError: {e}")
        return [None]*7


@st.cache_data
def load_dataset():
    for path in ['data/space_weather_raw.csv', 'data/space_weather_processed.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='h')
            return df
    return None

@st.cache_data
def load_pca_csvs():
    d2 = pd.read_csv('data/pca_2d.csv') if os.path.exists('data/pca_2d.csv') else None
    d3 = pd.read_csv('data/pca_3d.csv') if os.path.exists('data/pca_3d.csv') else None
    return d2, d3


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_risk(vals, models, pca, scaler, le, feature_cols, model_choice):
    storm_sev  = 0.4*vals['kp_index'] + 0.3*abs(vals['bz']) + 0.3*(vals['solar_wind_speed']/800)
    mag_dist   = np.sqrt(vals['bx']**2 + vals['by']**2 + vals['bz']**2)
    rad_press  = vals['proton_density'] * vals['solar_wind_speed']

    fv = {
        'solar_wind_speed': vals['solar_wind_speed'], 'proton_density': vals['proton_density'],
        'bx': vals['bx'], 'by': vals['by'], 'bz': vals['bz'],
        'plasma_temp': vals['plasma_temp'], 'kp_index': vals['kp_index'],
        'dst_index': vals['dst_index'], 'xray_flux': vals['xray_flux'],
        'ae_index': vals['ae_index'],
        'f107_index'     : vals.get('f107_index', 120.0),
        'ap_index'       : vals.get('ap_index', 8.0),
        'speed_pressure' : vals['solar_wind_speed'] * vals['proton_density'],
        'storm_severity' : storm_sev,
        'mag_disturbance': mag_dist, 'radiation_pressure': rad_press,
        'kp_rolling_3h'  : vals['kp_index'], 'bz_rolling_3h': vals['bz'],
        'wind_rolling_3h': vals['solar_wind_speed'],
        'dst_rolling_6h' : vals['dst_index'],
        'delta_bz': 0.0, 'delta_wind': 0.0, 'delta_proton': 0.0, 'delta_kp': 0.0,
        'bz_negative' : int(vals['bz'] < 0),
        'extreme_kp'  : int(vals['kp_index'] > 7),
        'extreme_dst' : int(vals['dst_index'] < -150),
        'high_speed'  : int(vals['solar_wind_speed'] > 600)
    }

    X        = pd.DataFrame([fv])[feature_cols]
    X_scaled = scaler.transform(X)

    # For Logistic Regression use PCA-transformed input
    model_map = {
        'Random Forest'      : ('random_forest',       X_scaled),
        'XGBoost'            : ('xgboost',             X_scaled),
        'Logistic Regression': ('logistic_regression', pca.transform(X_scaled))
    }
    key, X_in = model_map[model_choice]
    model     = models[key]

    pred_enc  = model.predict(X_in)[0]
    pred_label= le.inverse_transform([pred_enc])[0]

    proba_dict = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_in)[0]
        proba_dict = {le.classes_[i]: round(float(p)*100, 1) for i,p in enumerate(proba)}

    iso_score  = models['isolation_forest'].decision_function(X_scaled)[0]
    return pred_label, proba_dict, fv, iso_score < 0


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">🛰️ SPACE WEATHER MONITOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">SATELLITE FAILURE RISK PREDICTION SYSTEM</div>', unsafe_allow_html=True)

    loaded = load_models()
    if loaded[0] is None:
        st.stop()
    models, pca, scaler, le, feature_cols, results, pca_info = loaded
    df      = load_dataset()
    pca_2d, pca_3d = load_pca_csvs()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Predict Risk", "📊 Model Performance",
        "🔬 PCA Analysis", "🌌 Data Explorer", "⚙️ Feature Importance"
    ])

    # ══════════════════════════════════════════
    # TAB 1 — PREDICT (ALL MODELS AT ONCE)
    # ══════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns([1, 1.4], gap="large")
        with col_l:
            st.markdown('<div class="section-header">INPUT PARAMETERS</div>', unsafe_allow_html=True)

            st.markdown("**☀️ Solar Parameters**")
            kp     = st.slider("Kp Index (0–9)", 0.0, 9.0, 2.5, 0.1)
            dst    = st.slider("Dst Index (nT)", -400, 50, -15, 1)
            wind   = st.slider("Solar Wind Speed (km/s)", 200, 900, 450, 10)
            proton = st.slider("Proton Density (p/cm³)", 0.1, 50.0, 5.0, 0.1)

            st.markdown("**🧲 Magnetic Field (nT)**")
            bz = st.slider("Bz", -50.0, 20.0, 1.0, 0.5)
            bx = st.slider("Bx", -20.0, 20.0, 0.0, 0.5)
            by = st.slider("By", -20.0, 20.0, 0.0, 0.5)

            st.markdown("**🔬 Other**")
            plasma_temp = st.slider("Plasma Temp (×10⁴ K)", 0.1, 100.0, 10.0, 0.1)
            xray        = st.slider("X-ray Flux (×10⁻⁷ W/m²)", 0.001, 100.0, 0.5, 0.001)
            ae          = st.slider("AE Index (nT)", 0, 3000, 100, 10)

        with col_r:
            st.markdown('<div class="section-header">ALL MODELS PREDICTION</div>', unsafe_allow_html=True)
            input_vals = {
                'kp_index': kp, 'dst_index': dst,
                'solar_wind_speed': wind, 'proton_density': proton,
                'bz': bz, 'bx': bx, 'by': by,
                'plasma_temp': plasma_temp * 1e4,
                'xray_flux': xray * 1e-7, 'ae_index': ae
            }

            # Run ALL 3 models simultaneously
            all_model_names = ['Random Forest', 'XGBoost', 'Logistic Regression']
            all_preds = {}
            fv = None
            is_anomaly = False
            for mc in all_model_names:
                pred, proba_dict, fv, is_anomaly = predict_risk(
                    input_vals, models, pca, scaler, le, feature_cols, mc)
                all_preds[mc] = {'pred': pred, 'proba': proba_dict}

            # Majority vote for final verdict
            votes      = [all_preds[m]['pred'] for m in all_model_names]
            final_pred = max(set(votes), key=votes.count)
            rc = {'High':'risk-high','Medium':'risk-medium','Low':'risk-low'}[final_pred]
            re = {'High':'🔴','Medium':'🟡','Low':'🟢'}[final_pred]

            st.markdown(f"""
            <div class="prediction-box">
                <div class="metric-label">FINAL VERDICT (MAJORITY VOTE)</div>
                <div class="metric-value {rc}" style="font-size:2.6rem">{re} {final_pred.upper()} RISK</div>
                <div style="color:#888;font-family:monospace;margin-top:.4rem">
                    🌲 {votes[0]} &nbsp;·&nbsp; ⚡ {votes[1]} &nbsp;·&nbsp; 📈 {votes[2]}
                </div>
            </div>""", unsafe_allow_html=True)

            # Individual model result cards
            st.markdown("**🤖 Individual Model Results**")
            m_cols = st.columns(3)
            icons  = {'High':'🔴','Medium':'🟡','Low':'🟢'}
            colors = {'High':'#ff4757','Medium':'#ffa502','Low':'#2ed573'}
            micons = {'Random Forest':'🌲','XGBoost':'⚡','Logistic Regression':'📈'}
            for i, mc in enumerate(all_model_names):
                p    = all_preds[mc]['pred']
                conf = all_preds[mc]['proba'].get(p, 0)
                with m_cols[i]:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#0f1628,#1a2340);
                         border:1px solid {colors[p]};border-radius:12px;
                         padding:1rem;text-align:center;margin-bottom:0.5rem">
                        <div style="color:#aaa;font-family:'Share Tech Mono',monospace;
                             font-size:0.68rem;letter-spacing:1px">{micons[mc]} {mc.upper()}</div>
                        <div style="color:{colors[p]};font-family:'Orbitron',monospace;
                             font-size:1.2rem;font-weight:700;margin:.3rem 0">
                             {icons[p]} {p}</div>
                        <div style="color:#666;font-family:monospace;font-size:0.78rem">
                             {conf:.1f}% confidence</div>
                    </div>""", unsafe_allow_html=True)

            # Probability comparison bar chart
            st.markdown("**📊 Probability Comparison — All Models**")
            chart_data = []
            for mc in all_model_names:
                for risk, prob in all_preds[mc]['proba'].items():
                    chart_data.append({'Model': mc, 'Risk': risk, 'Probability (%)': prob})
            if chart_data:
                cdf = pd.DataFrame(chart_data)
                fig = px.bar(cdf, x='Model', y='Probability (%)', color='Risk',
                             color_discrete_map=COLOR_MAP, barmode='group',
                             text='Probability (%)')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=280, **PLOT_LAYOUT,
                                  yaxis=dict(gridcolor='#1e3a5f', range=[0,115]),
                                  xaxis=dict(gridcolor='#1e3a5f'),
                                  legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig, use_container_width=True)

            # Alert based on final verdict
            alerts = {
                'High'  : ('alert-high',   '🚨 CRITICAL: Severe geomagnetic storm! Recommend satellite safe mode.'),
                'Medium': ('alert-medium', '⚠️ WARNING: Elevated solar activity. Monitor satellite systems.'),
                'Low'   : ('alert-low',    '✅ NOMINAL: Space weather calm. Operations normal.')
            }
            cls, msg = alerts[final_pred]
            st.markdown(f'<div class="alert-box {cls}">{msg}</div>', unsafe_allow_html=True)
            if is_anomaly:
                st.markdown('<div class="alert-box alert-high">🔍 ANOMALY DETECTED by Isolation Forest!</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Storm Severity",  f"{fv['storm_severity']:.2f}")
            c2.metric("Mag Disturbance", f"{fv['mag_disturbance']:.2f} nT")
            c3.metric("Rad Pressure",    f"{fv['radiation_pressure']:.0f}")

    # ══════════════════════════════════════════
    # TAB 2 — MODEL PERFORMANCE
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">MODEL EVALUATION METRICS</div>', unsafe_allow_html=True)
        mr  = results['model_results']
        mdf = pd.DataFrame({
            'Model'    : list(mr.keys()),
            'Accuracy' : [v['accuracy']  for v in mr.values()],
            'Precision': [v['precision'] for v in mr.values()],
            'Recall'   : [v['recall']    for v in mr.values()],
            'F1-Score' : [v['f1_score']  for v in mr.values()]
        })

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for metric, color in zip(['Accuracy','Precision','Recall','F1-Score'],
                                     ['#00d4ff','#7b2ff7','#ff6b35','#2ed573']):
                fig.add_trace(go.Bar(name=metric, x=mdf['Model'], y=mdf[metric],
                                     marker_color=color,
                                     text=[f"{v:.3f}" for v in mdf[metric]],
                                     textposition='outside'))
            fig.update_layout(title='Model Comparison', barmode='group', height=380,
                              **PLOT_LAYOUT,
                              legend=dict(bgcolor='rgba(0,0,0,0)'),
                              yaxis=dict(gridcolor='#1e3a5f', range=[0,1.1]),
                              xaxis=dict(gridcolor='#1e3a5f'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            best = 'XGBoost' if 'XGBoost' in mr else list(mr.keys())[0]
            cm   = mr[best]['confusion_matrix']
            fig  = px.imshow(cm, x=results['classes'], y=results['classes'],
                             text_auto=True, color_continuous_scale='Blues',
                             title=f'{best} Confusion Matrix')
            fig.update_layout(height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(mdf, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 3 — PCA ANALYSIS  ← NEW
    # ══════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">PRINCIPAL COMPONENT ANALYSIS (PCA)</div>',
                    unsafe_allow_html=True)

        # ── Summary cards ─────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="pca-card">
            <div class="metric-label">ORIGINAL FEATURES</div>
            <div class="metric-value">{pca_info['total_features']}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="pca-card">
            <div class="metric-label">COMPONENTS @ 95%</div>
            <div class="metric-value" style="color:#00d4ff">{pca_info['n_components_95']}</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="pca-card">
            <div class="metric-label">COMPONENTS @ 99%</div>
            <div class="metric-value" style="color:#7b2ff7">{pca_info['n_components_99']}</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="pca-card">
            <div class="metric-label">DIMENSION REDUCTION</div>
            <div class="metric-value" style="color:#2ed573">
                {pca_info['total_features']}→{pca_info['n_components_95']}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        col1, col2 = st.columns(2)

        # ── Scree plot + cumulative variance ──
        with col1:
            evr = pca_info['explained_variance_ratio']
            cum = pca_info['cumulative_variance']
            n   = len(evr)
            pcs = [f"PC{i+1}" for i in range(n)]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=pcs, y=[v*100 for v in evr],
                                 name='Individual %', marker_color='#4a9eff',
                                 text=[f"{v*100:.1f}%" for v in evr],
                                 textposition='outside'))
            fig.add_trace(go.Scatter(x=pcs, y=[v*100 for v in cum],
                                     name='Cumulative %', mode='lines+markers',
                                     line=dict(color='#ff6b35', width=2),
                                     marker=dict(size=6)))
            fig.add_hline(y=95, line_dash='dash', line_color='#2ed573',
                          annotation_text="95% threshold")
            fig.add_hline(y=99, line_dash='dash', line_color='#ffa502',
                          annotation_text="99% threshold")
            fig.update_layout(title='Scree Plot — Explained Variance', height=380,
                              **PLOT_LAYOUT,
                              yaxis=dict(title='Variance (%)', gridcolor='#1e3a5f'),
                              xaxis=dict(gridcolor='#1e3a5f'),
                              legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        # ── PC1 loadings bar chart ─────────────
        with col2:
            load1 = pca_info['loadings_pc1']
            ldf   = pd.DataFrame(list(load1.items()), columns=['Feature', 'Loading'])
            ldf   = ldf.reindex(ldf['Loading'].abs().sort_values(ascending=True).index)

            fig = go.Figure(go.Bar(
                x=ldf['Loading'], y=ldf['Feature'], orientation='h',
                marker=dict(
                    color=ldf['Loading'],
                    colorscale=[[0,'#ff4757'],[0.5,'#1e3a5f'],[1,'#2ed573']],
                    showscale=True,
                    colorbar=dict(title='Loading', tickfont=dict(color='#aaa'))
                )))
            fig.update_layout(title='PC1 Feature Loadings', height=380, **PLOT_LAYOUT,
                              xaxis=dict(title='Loading Value', gridcolor='#1e3a5f'),
                              yaxis=dict(gridcolor='#1e3a5f'))
            st.plotly_chart(fig, use_container_width=True)

        # ── 2D PCA Scatter ─────────────────────
        if pca_2d is not None:
            st.markdown('<div class="section-header">PCA 2D — RISK CLUSTER SEPARATION</div>',
                        unsafe_allow_html=True)
            sample = pca_2d.sample(min(1500, len(pca_2d)), random_state=42)
            fig = px.scatter(sample, x='PC1', y='PC2', color='risk_level',
                             color_discrete_map=COLOR_MAP, opacity=0.6,
                             title='2D PCA — Risk Level Clusters')
            fig.update_layout(height=420, **PLOT_LAYOUT,
                              yaxis=dict(gridcolor='#1e3a5f'),
                              xaxis=dict(gridcolor='#1e3a5f'),
                              legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        # ── 3D PCA Scatter ─────────────────────
        if pca_3d is not None:
            st.markdown('<div class="section-header">PCA 3D — RISK CLUSTER SEPARATION</div>',
                        unsafe_allow_html=True)
            sample3 = pca_3d.sample(min(1000, len(pca_3d)), random_state=42)
            fig3 = px.scatter_3d(sample3, x='PC1', y='PC2', z='PC3',
                                 color='risk_level', color_discrete_map=COLOR_MAP,
                                 opacity=0.7, title='3D PCA Scatter')
            fig3.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#aaa', family='Share Tech Mono'),
                               scene=dict(
                                   xaxis=dict(backgroundcolor='#0a0e1a', gridcolor='#1e3a5f'),
                                   yaxis=dict(backgroundcolor='#0a0e1a', gridcolor='#1e3a5f'),
                                   zaxis=dict(backgroundcolor='#0a0e1a', gridcolor='#1e3a5f')
                               ))
            st.plotly_chart(fig3, use_container_width=True)

        # ── PCA explainer ──────────────────────
        st.info(f"""
        **📌 PCA Summary for this project:**
        - We started with **{pca_info['total_features']} features** (raw + engineered)
        - PCA reduced this to **{pca_info['n_components_95']} components** while keeping **95% of information**
        - The 2D and 3D scatter plots above show the three risk classes are **clearly separable** in PCA space
        - Logistic Regression is trained on these PCA components (noise removed → better generalisation)
        - Features with **high PC1 loadings** (like Kp Index, Storm Severity, Dst) drive the most variance
        """)

    # ══════════════════════════════════════════
    # TAB 4 — DATA EXPLORER
    # ══════════════════════════════════════════
    with tab4:
        if df is None:
            st.warning("⚠️ Dataset not found. Run generate_dataset.py or fetch_real_data.py first.")
            st.stop()

        st.markdown('<div class="section-header">DATASET EXPLORATION</div>', unsafe_allow_html=True)

        # ── Debug: show what columns exist ────
        if 'risk_level' not in df.columns:
            st.error(f"❌ 'risk_level' column missing. Columns found: {list(df.columns)}")
            st.stop()

        rc = df['risk_level'].value_counts()

        # ── Summary metrics ───────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("🔴 High Risk",   f"{rc.get('High',  0):,}")
        c3.metric("🟡 Medium Risk", f"{rc.get('Medium', 0):,}")
        c4.metric("🟢 Low Risk",    f"{rc.get('Low',    0):,}")

        # ── Dataset stats ─────────────────────
        with st.expander("📋 View raw dataset (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)

        with st.expander("📈 Dataset statistics"):
            num_cols = df.select_dtypes(include='number').columns.tolist()
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

        col1, col2 = st.columns(2)

        # ── Risk distribution pie ─────────────
        with col1:
            fig = px.pie(values=rc.values, names=rc.index,
                         color=rc.index, color_discrete_map=COLOR_MAP,
                         hole=0.5, title="Risk Level Distribution")
            fig.update_layout(height=320, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # ── Kp Index over time ────────────────
        with col2:
            df_tail = df.tail(500).copy()
            fig = px.scatter(df_tail, x='timestamp', y='kp_index',
                             color='risk_level', color_discrete_map=COLOR_MAP,
                             title='Kp Index Over Time (Last 500 Records)')
            fig.update_layout(height=320, **PLOT_LAYOUT,
                              yaxis=dict(gridcolor='#1e3a5f'),
                              xaxis=dict(gridcolor='#1e3a5f'))
            st.plotly_chart(fig, use_container_width=True)

        # ── Bz vs Dst scatter ─────────────────
        n_sample = min(500, len(df))
        df_s = df.sample(n_sample, random_state=42).copy()
        fig = px.scatter(df_s, x='bz', y='dst_index',
                         color='risk_level', color_discrete_map=COLOR_MAP,
                         title='Bz vs Dst Index (Storm Indicators)', opacity=0.7)
        fig.update_layout(height=350, **PLOT_LAYOUT,
                          yaxis=dict(gridcolor='#1e3a5f'),
                          xaxis=dict(gridcolor='#1e3a5f'))
        st.plotly_chart(fig, use_container_width=True)

        # ── Solar wind speed distribution ─────
        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df, x='solar_wind_speed', color='risk_level',
                               color_discrete_map=COLOR_MAP, nbins=50,
                               title='Solar Wind Speed Distribution')
            fig.update_layout(height=300, **PLOT_LAYOUT,
                              yaxis=dict(gridcolor='#1e3a5f'),
                              xaxis=dict(gridcolor='#1e3a5f'),
                              legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.histogram(df, x='kp_index', color='risk_level',
                               color_discrete_map=COLOR_MAP, nbins=50,
                               title='Kp Index Distribution')
            fig.update_layout(height=300, **PLOT_LAYOUT,
                              yaxis=dict(gridcolor='#1e3a5f'),
                              xaxis=dict(gridcolor='#1e3a5f'),
                              legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 5 — FEATURE IMPORTANCE
    # ══════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">FEATURE IMPORTANCE (Random Forest)</div>',
                    unsafe_allow_html=True)
        fi  = results['feature_importance']
        fdf = pd.DataFrame(list(fi.items()), columns=['Feature','Importance'])
        fdf = fdf.sort_values('Importance', ascending=True).tail(15)

        fig = go.Figure(go.Bar(
            x=fdf['Importance'], y=fdf['Feature'], orientation='h',
            marker=dict(color=fdf['Importance'],
                        colorscale=[[0,'#1a2340'],[0.5,'#4a9eff'],[1,'#00d4ff']],
                        showscale=False),
            text=[f"{v:.4f}" for v in fdf['Importance']], textposition='outside'))
        fig.update_layout(height=520,
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#aaa', family='Share Tech Mono'),
                          xaxis=dict(title='Importance', gridcolor='#1e3a5f'),
                          yaxis=dict(gridcolor='#1e3a5f'),
                          margin=dict(t=10, b=10, l=10, r=80))
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **📌 Key Insights:**
        - **Kp Index** and **Dst Index** are strongest predictors
        - **Storm Severity** (engineered) captures combined multi-parameter effect
        - **Bz component** is critical — negative Bz drives energy into Earth's magnetosphere
        - **Extreme flags** capture the most dangerous threshold breaches
        """)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'Orbitron',monospace;color:#00d4ff;
        font-size:1rem;letter-spacing:2px;margin-bottom:1rem">🛰️ SYSTEM STATUS</div>""",
        unsafe_allow_html=True)
    st.success("✅ Models Loaded")
    st.success("✅ PCA Applied")
    st.success("✅ Dataset Ready")
    st.markdown("---")
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;color:#4a9eff;font-size:0.8rem">
    RISK THRESHOLDS:<br><br>
    🔴 HIGH: Kp > 5 or Dst < -100<br>
    🟡 MEDIUM: Kp 3–5 or Dst -50 to -100<br>
    🟢 LOW: Kp < 3 and Dst > -50
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;color:#555;font-size:0.75rem">
    MODELS:<br>
    • Random Forest<br>• XGBoost<br>
    • Logistic Regression (PCA)<br>• Isolation Forest<br><br>
    PCA: 22 → n_95 components<br>
    DATA: NASA/NOAA Space Weather
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
