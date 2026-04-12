import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EU AI Act Compliance Classifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }a
    .sub-header {
        font-size: 1rem;
        color: #cccccc;
        margin-bottom: 2rem;
    }
    .risk-prohibited {
        background-color: #5c1a1a;
        border-left: 5px solid #e74c3c;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .risk-high {
        background-color: #5c3a0e;
        border-left: 5px solid #e67e22;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .risk-limited {
        background-color: #4a4000;
        border-left: 5px solid #f1c40f;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .risk-minimal {
        background-color: #0e3d22;
        border-left: 5px solid #2ecc71;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #444;
        color: #ffffff;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .legal-chunk {
        background: #1a2a4a;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        border-left: 3px solid #3498db;
        color: #e0e0e0;
    }
    .footer {
        text-align: center;
        color: #aaaaaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, '..', 'models')

    try:
        with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
            models['ml_model'] = pickle.load(f)
        with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            models['ml_vectorizer'] = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            models['label_encoder'] = pickle.load(f)
        with open(os.path.join(model_dir, 'rag_vectorizer.pkl'), 'rb') as f:
            models['rag_vectorizer'] = pickle.load(f)
        with open(os.path.join(model_dir, 'rag_chunks.pkl'), 'rb') as f:
            models['rag_chunks'] = pickle.load(f)
        models['status'] = 'loaded'
    except Exception as e:
        models['status'] = f'error: {e}'

    return models


# ─────────────────────────────────────────────
# CLASSIFICATION RULES
# ─────────────────────────────────────────────
PROHIBITED_KEYWORDS = [
    'social scor', 'citizen scor', 'social credit', 'behavior scor',
    'subliminal', 'manipulat', 'deepfake', 'synthetic media', 'fake video',
    'disinformation', 'emotion recogni', 'emotion detect',
    'real.time biometric', 'mass surveillance'
]

HIGH_RISK_KEYWORDS = {
    'Annex III — Category 1 (Biometric)': ['facial recognition', 'biometric', 'fingerprint', 'face recogni', 'identity verif'],
    'Annex III — Category 2 (Critical Infrastructure)': ['autonomous vehicle', 'self.driving', 'autopilot', 'power grid', 'electricity grid', 'aircraft', 'railway'],
    'Annex III — Category 3 (Education)': ['student', 'school', 'university', 'admission', 'teacher evaluat', 'grading'],
    'Annex III — Category 4 (Employment)': ['hiring', 'recruitment', 'resume', 'employ', 'worker', 'scheduling algorithm', 'performance monitor'],
    'Annex III — Category 5 (Essential Services)': ['credit scor', 'loan', 'insurance', 'medical', 'surgery', 'hospital', 'patient', 'diagnosis'],
    'Annex III — Category 6 (Law Enforcement)': ['police', 'crime predict', 'criminal', 'recidivism', 'sentencing', 'court', 'profiling'],
    'Annex III — Category 7 (Migration)': ['migration', 'asylum', 'border control', 'visa', 'refugee'],
    'Annex III — Category 8 (Democracy)': ['election', 'voting', 'democratic', 'political campaign', 'propaganda']
}

LIMITED_RISK_KEYWORDS = [
    'chatbot', 'virtual assistant', 'recommendation', 'content filter',
    'spam filter', 'advertisement', 'sentiment analys', 'translation'
]

COMPLIANCE_REQUIREMENTS = {
    'Prohibited': {
        'action': 'CEASE IMMEDIATELY',
        'color': '#e74c3c',
        'emoji': '🔴',
        'requirements': [
            'This AI system is explicitly banned under EU AI Act Annex I',
            'Immediate discontinuation of the system is required',
            'Regulatory authorities must be notified',
            'Legal counsel should be consulted immediately'
        ],
        'deadline': 'Immediate — no grace period',
        'css_class': 'risk-prohibited'
    },
    'High Risk': {
        'action': 'COMPLIANCE REQUIRED BEFORE DEPLOYMENT',
        'color': '#e67e22',
        'emoji': '🟠',
        'requirements': [
            'Conduct conformity assessment before market placement',
            'Implement human oversight mechanisms',
            'Establish risk management system (Article 9)',
            'Ensure data governance and quality (Article 10)',
            'Maintain technical documentation (Article 11)',
            'Enable automatic logging of events (Article 12)',
            'Ensure transparency and provision of information (Article 13)',
            'Design for human oversight (Article 14)',
            'Ensure accuracy, robustness, and cybersecurity (Article 15)',
            'Register system in EU database before deployment (Article 49)'
        ],
        'deadline': 'Full compliance required by August 2026',
        'css_class': 'risk-high'
    },
    'Limited Risk': {
        'action': 'TRANSPARENCY OBLIGATIONS APPLY',
        'color': '#f1c40f',
        'emoji': '🟡',
        'requirements': [
            'Disclose to users that they are interacting with an AI system (Article 52)',
            'Label AI-generated content appropriately',
            'Ensure users can opt out where applicable'
        ],
        'deadline': 'Applies from August 2026',
        'css_class': 'risk-limited'
    },
    'Minimal Risk': {
        'action': 'NO MANDATORY REQUIREMENTS',
        'color': '#2ecc71',
        'emoji': '🟢',
        'requirements': [
            'No mandatory compliance obligations under EU AI Act',
            'Voluntary adherence to AI codes of conduct recommended',
            'Good practice: maintain internal documentation'
        ],
        'deadline': 'No mandatory deadline',
        'css_class': 'risk-minimal'
    }
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def classify_risk(text):
    text_lower = str(text).lower()
    for keyword in PROHIBITED_KEYWORDS:
        if re.search(keyword, text_lower):
            return 'Prohibited', 'Annex I', f'Matches prohibited pattern: "{keyword}"'
    for category, keywords in HIGH_RISK_KEYWORDS.items():
        for keyword in keywords:
            if re.search(keyword, text_lower):
                return 'High Risk', category, f'Matches high risk pattern: "{keyword}"'
    for keyword in LIMITED_RISK_KEYWORDS:
        if re.search(keyword, text_lower):
            return 'Limited Risk', 'Article 52', f'Matches limited risk pattern: "{keyword}"'
    return 'Minimal Risk', 'Not applicable', 'No regulated pattern detected'


def hybrid_classify(text, models):
    cleaned = clean_text(text)
    risk, annex, reason = classify_risk(cleaned)
    if risk != 'Minimal Risk':
        return risk, annex, reason
    if models['status'] == 'loaded':
        vec = models['ml_vectorizer'].transform([cleaned])
        pred_enc = models['ml_model'].predict(vec)
        pred = models['label_encoder'].inverse_transform(pred_enc)[0]
        return pred, 'ML Model', 'Predicted by trained XGBoost classifier'
    return risk, annex, reason


def retrieve_chunks(query, models, top_k=3):
    if models['status'] != 'loaded':
        return []
    chunks_df = models['rag_chunks']
    query_vec = models['rag_vectorizer'].transform([query])
    chunk_matrix = models['rag_vectorizer'].transform(chunks_df['text'].tolist())
    similarities = cosine_similarity(query_vec, chunk_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:
            results.append({
                'title': chunks_df.iloc[idx]['title'],
                'text': chunks_df.iloc[idx]['text'][:400],
                'similarity': round(float(similarities[idx]), 4)
            })
    return results


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/320px-Flag_of_Europe.svg.png", width=80)
    st.markdown("### ⚖️ EU AI Act Classifier")
    st.markdown("---")

    st.markdown("**Navigation**")
    page = st.radio("", [
        "🔍 Compliance Checker",
        "📊 Policy Dashboard",
        "📚 About the EU AI Act"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Risk Level Guide**")
    st.markdown("🔴 **Prohibited** — Banned completely")
    st.markdown("🟠 **High Risk** — Strict compliance")
    st.markdown("🟡 **Limited Risk** — Transparency only")
    st.markdown("🟢 **Minimal Risk** — No obligations")

    st.markdown("---")
    st.markdown("**Built by**")
    st.markdown("[Jayesh Ranghera](https://github.com/jayeshranghera)")
    st.markdown("*AI Policy & Governance Analyst*")


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
models = load_models()


# ═════════════════════════════════════════════
# PAGE 1 — COMPLIANCE CHECKER
# ═════════════════════════════════════════════
if page == "🔍 Compliance Checker":

    st.markdown('<p class="main-header">⚖️ EU AI Act Compliance Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Check if your AI system complies with the EU Artificial Intelligence Act (Regulation 2024/1689)</p>', unsafe_allow_html=True)

    # Input
    st.markdown("### Describe Your AI System")
    description = st.text_area(
        "",
        placeholder="Example: Our AI system automatically screens job applications and ranks candidates based on their resume and online profile data...",
        height=120,
        label_visibility="collapsed"
    )

    # Example buttons
    st.markdown("**Quick Examples:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📄 Resume Screening"):
            description = "An AI system that automatically screens and ranks job applicants based on resume analysis"
    with col2:
        if st.button("🏥 Medical Diagnosis"):
            description = "AI tool that analyzes medical imaging to detect cancer and assist doctors in diagnosis"
    with col3:
        if st.button("💬 Customer Chatbot"):
            description = "A chatbot that answers customer queries and provides product recommendations"
    with col4:
        if st.button("🚗 Autonomous Vehicle"):
            description = "Self-driving car system that navigates public roads without human intervention"

    st.markdown("---")

    # Classify button
    if st.button("🔍 Analyse Compliance", type="primary", use_container_width=True):
        if not description or len(description.strip()) < 20:
            st.warning("Please enter a description of at least 20 characters.")
        else:
            with st.spinner("Analysing against EU AI Act..."):

                # Classify
                risk_level, annex_ref, reason = hybrid_classify(description, models)
                req = COMPLIANCE_REQUIREMENTS[risk_level]

                # ── Risk Banner ──────────────────────────────
                st.markdown(f"""
                <div class="{req['css_class']}">
                    <h2>{req['emoji']} {risk_level.upper()}</h2>
                    <p><strong>Legal Reference:</strong> {annex_ref}</p>
                    <p><strong>Classification Basis:</strong> {reason}</p>
                    <p><strong>Required Action:</strong> {req['action']}</p>
                </div>
                """, unsafe_allow_html=True)

                # ── Two columns ──────────────────────────────
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown('<p class="section-title">📋 Compliance Requirements</p>', unsafe_allow_html=True)
                    for i, r in enumerate(req['requirements'], 1):
                        st.markdown(f"**{i}.** {r}")

                    st.markdown('<p class="section-title">⏰ Deadline</p>', unsafe_allow_html=True)
                    st.info(req['deadline'])

                with col_right:
                    st.markdown('<p class="section-title">📖 Relevant EU AI Act Sections</p>', unsafe_allow_html=True)
                    chunks = retrieve_chunks(description, models, top_k=3)
                    if chunks:
                        for chunk in chunks:
                            st.markdown(f"""
                            <div class="legal-chunk">
                                <strong>{chunk['title']}</strong>
                                <br><em>Relevance: {chunk['similarity']}</em>
                                <br><br>{chunk['text']}...
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Legal section retrieval unavailable.")

                # ── Compliance Checklist ─────────────────────
                if risk_level == 'High Risk':
                    st.markdown("---")
                    st.markdown("### ✅ Compliance Checklist")
                    st.markdown("*Track your compliance progress:*")
                    checklist_items = [
                        "Risk management system established (Article 9)",
                        "Data governance procedures in place (Article 10)",
                        "Technical documentation prepared (Article 11)",
                        "Automatic logging enabled (Article 12)",
                        "Transparency information provided (Article 13)",
                        "Human oversight mechanisms designed (Article 14)",
                        "Conformity assessment completed",
                        "System registered in EU database (Article 49)"
                    ]
                    cols = st.columns(2)
                    for i, item in enumerate(checklist_items):
                        with cols[i % 2]:
                            st.checkbox(item, key=f"check_{i}")


# ═════════════════════════════════════════════
# PAGE 2 — POLICY DASHBOARD
# ═════════════════════════════════════════════
elif page == "📊 Policy Dashboard":

    st.markdown('<p class="main-header">📊 Policy Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">EU AI Act compliance gap analysis across industries</p>', unsafe_allow_html=True)

    # Load policy data
    policy_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'policy_gap_analysis.csv')

    if os.path.exists(policy_data_path):
        policy_df = pd.read_csv(policy_data_path)
    else:
        # Fallback data
        policy_df = pd.DataFrame([
            {"description": "Resume screening AI", "sector": "Employment", "risk_level": "High Risk"},
            {"description": "Loan approval AI", "sector": "Finance", "risk_level": "High Risk"},
            {"description": "University admission AI", "sector": "Education", "risk_level": "High Risk"},
            {"description": "Facial recognition border", "sector": "Migration", "risk_level": "High Risk"},
            {"description": "Customer chatbot", "sector": "Customer Service", "risk_level": "Limited Risk"},
            {"description": "Predictive policing", "sector": "Law Enforcement", "risk_level": "High Risk"},
            {"description": "Medical diagnosis AI", "sector": "Healthcare", "risk_level": "High Risk"},
            {"description": "Autonomous vehicle", "sector": "Transport", "risk_level": "High Risk"},
            {"description": "Content recommendation", "sector": "Media", "risk_level": "Limited Risk"},
            {"description": "Social scoring system", "sector": "Public Admin", "risk_level": "Prohibited"},
            {"description": "Insurance premium AI", "sector": "Insurance", "risk_level": "High Risk"},
            {"description": "Student monitoring AI", "sector": "Education", "risk_level": "High Risk"},
            {"description": "Asylum risk assessment", "sector": "Migration", "risk_level": "High Risk"},
            {"description": "Deepfake generator", "sector": "Media", "risk_level": "Prohibited"},
            {"description": "Smart home assistant", "sector": "Consumer", "risk_level": "Minimal Risk"},
            {"description": "Grid management AI", "sector": "Energy", "risk_level": "High Risk"},
            {"description": "Employee monitoring", "sector": "Employment", "risk_level": "High Risk"},
            {"description": "Spam filter", "sector": "IT Services", "risk_level": "Minimal Risk"},
            {"description": "Court sentencing AI", "sector": "Justice", "risk_level": "High Risk"},
            {"description": "Music recommendation", "sector": "Entertainment", "risk_level": "Minimal Risk"},
        ])

    # ── KPI Metrics ──────────────────────────────
    total = len(policy_df)
    high_risk = len(policy_df[policy_df['risk_level'] == 'High Risk'])
    prohibited = len(policy_df[policy_df['risk_level'] == 'Prohibited'])
    non_compliant_pct = round((high_risk + prohibited) / total * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total AI Systems Analysed", total)
    with col2:
        st.metric("High Risk Systems", high_risk, delta=f"{round(high_risk/total*100)}% of total")
    with col3:
        st.metric("Prohibited Systems", prohibited)
    with col4:
        st.metric("Non-Compliant Rate", f"{non_compliant_pct}%")

    st.markdown("---")

    # ── Charts ──────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Risk Level Distribution")
        risk_counts = policy_df['risk_level'].value_counts()
        colors = {
            'Prohibited': '#e74c3c',
            'High Risk': '#e67e22',
            'Limited Risk': '#f1c40f',
            'Minimal Risk': '#2ecc71'
        }
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        bar_colors = [colors.get(r, '#95a5a6') for r in risk_counts.index]
        ax1.bar(risk_counts.index, risk_counts.values, color=bar_colors, edgecolor='white', linewidth=1.5)
        ax1.set_ylabel('Number of AI Systems')
        ax1.set_title('EU AI Act Risk Distribution', fontweight='bold')
        for i, v in enumerate(risk_counts.values):
            ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        sns.despine()
        st.pyplot(fig1)

    with col_b:
        st.markdown("#### Non-Compliance Rate by Sector")
        non_c = policy_df[policy_df['risk_level'].isin(['High Risk', 'Prohibited'])]
        sector_rate = (non_c['sector'].value_counts() / policy_df['sector'].value_counts() * 100).dropna().sort_values()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(sector_rate.index, sector_rate.values, color='#e67e22', edgecolor='white')
        ax2.set_xlabel('Non-Compliance Rate (%)')
        ax2.set_title('Sectors at Risk', fontweight='bold')
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        sns.despine()
        st.pyplot(fig2)

    # ── Policy Findings ──────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Key Policy Findings")

    col1, col2 = st.columns(2)
    with col1:
        st.error(f"**{non_compliant_pct}% of common AI use cases** fall under High Risk or Prohibited categories under the EU AI Act.")
        st.warning("**Education, Healthcare, and Law Enforcement** sectors show 100% non-compliance rate — every AI system in these sectors requires strict compliance measures.")
    with col2:
        st.info("**Indian IT companies** building AI products for EU clients face significant compliance exposure — especially in HR tech, fintech, and healthtech.")
        st.success("**Compliance deadline: August 2026** — organisations must act now to avoid regulatory penalties and market access restrictions.")

    # ── Data Table ───────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Full Analysis Table")
    risk_filter = st.multiselect(
        "Filter by Risk Level:",
        options=['Prohibited', 'High Risk', 'Limited Risk', 'Minimal Risk'],
        default=['Prohibited', 'High Risk']
    )
    filtered_df = policy_df[policy_df['risk_level'].isin(risk_filter)] if risk_filter else policy_df
    st.dataframe(filtered_df[['description', 'sector', 'risk_level']].reset_index(drop=True), use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═════════════════════════════════════════════
elif page == "📚 About the EU AI Act":

    st.markdown('<p class="main-header">📚 About the EU AI Act</p>', unsafe_allow_html=True)

    st.markdown("""
    ### What is the EU AI Act?
    The **EU Artificial Intelligence Act** (Regulation 2024/1689) is the world's first comprehensive legal framework for artificial intelligence.
    Passed in **June 2024**, it establishes binding rules for AI systems deployed in the European Union.

    ---

    ### Risk Classification Framework

    | Risk Level | Definition | Examples |
    |-----------|------------|---------|
    | 🔴 **Prohibited** | Banned completely | Social scoring, deepfakes for manipulation, real-time public biometric surveillance |
    | 🟠 **High Risk** | Strict compliance required | Hiring AI, medical diagnosis, credit scoring, autonomous vehicles, court sentencing |
    | 🟡 **Limited Risk** | Transparency obligations | Chatbots, recommendation systems, deepfake disclosure |
    | 🟢 **Minimal Risk** | No mandatory obligations | Spam filters, music recommendations, smart home assistants |

    ---

    ### Key Compliance Deadlines

    | Date | Requirement |
    |------|------------|
    | **February 2025** | Prohibited AI systems banned |
    | **August 2025** | GPAI model rules apply |
    | **August 2026** | High Risk AI systems must comply |
    | **August 2027** | Embedded AI systems must comply |

    ---

    ### About This Tool

    This classifier was built as part of a policy research project to bridge the gap between complex EU regulation and organisations that need to understand their compliance obligations.

    **Methodology:**
    - Rule-based classifier grounded in Annex I and Annex III of the EU AI Act
    - XGBoost ML model trained on 514 real-world AI incidents (AIID dataset)
    - RAG pipeline using official EU AI Act PDF (529 chunks indexed)
    - Hybrid approach: rules take priority, ML model as fallback

    **Built by:** Jayesh Ranghera — Applied AI Policy & Governance Analyst
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📄 Official Source**")
        st.markdown("[EU AI Act Full Text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)")
    with col2:
        st.markdown("**💻 GitHub**")
        st.markdown("[View Project Code](https://github.com/jayeshranghera/eu-ai-act-compliance-classifier)")
    with col3:
        st.markdown("**📊 Data Source**")
        st.markdown("[AI Incident Database](https://incidentdatabase.ai)")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by Jayesh Ranghera | EU AI Act Compliance Classifier | 2026<br>
    <em>This tool is for informational purposes only and does not constitute legal advice.</em>
</div>
""", unsafe_allow_html=True)
