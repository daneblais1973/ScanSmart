"""
Professional Layout Styles for ScanSmart Trading Platform
Enhanced with consistent theming and professional UI components
"""

import streamlit as st

def get_professional_dark_theme():
    """Get professional dark theme CSS styles"""
    return inject_professional_styles()

def inject_professional_styles():
    """Inject comprehensive professional CSS styles"""
    
    css = """
    <style>
    /* ===== ROOT VARIABLES ===== */
    :root {
        /* Colors - Professional dark theme */
        --primary-bg: #0f172a;
        --secondary-bg: #1e293b;
        --accent-bg: #334155;
        --card-bg: rgba(255, 255, 255, 0.03);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --border-color: #334155;
        --border-light: #475569;
        
        /* Accent colors */
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-purple: #8b5cf6;
        
        /* Spacing */
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        --spacing-2xl: 3rem;
        
        /* Border radius */
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    /* ===== BASE STYLES ===== */
    .stApp {
        background: var(--primary-bg);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
    }

    /* ===== LAYOUT UTILITIES ===== */
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 var(--spacing-md);
    }

    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--spacing-lg);
    }

    .grid-3 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: var(--spacing-lg);
    }

    .grid-4 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: var(--spacing-lg);
    }

    /* ===== CARDS ===== */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        margin-bottom: var(--spacing-lg);
        backdrop-filter: blur(10px);
    }

    .card-header {
        display: flex;
        justify-content: between;
        align-items: center;
        margin-bottom: var(--spacing-md);
        padding-bottom: var(--spacing-sm);
        border-bottom: 1px solid var(--border-light);
    }

    /* ===== TYPOGRAPHY ===== */
    .h1 { font-size: 2.5rem; font-weight: 700; margin: var(--spacing-xl) 0 var(--spacing-lg) 0; }
    .h2 { font-size: 2rem; font-weight: 600; margin: var(--spacing-lg) 0 var(--spacing-md) 0; }
    .h3 { font-size: 1.5rem; font-weight: 600; margin: var(--spacing-lg) 0 var(--spacing-md) 0; }
    .h4 { font-size: 1.25rem; font-weight: 500; margin: var(--spacing-md) 0; }

    .text-muted { color: var(--text-muted); }
    .text-secondary { color: var(--text-secondary); }

    /* ===== BUTTONS ===== */
    .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: var(--spacing-sm) var(--spacing-lg);
        border-radius: var(--radius-md);
        font-weight: 500;
        font-size: 0.875rem;
        text-decoration: none;
        cursor: pointer;
        border: none;
        transition: all 0.2s ease;
        gap: var(--spacing-sm);
    }

    .btn-primary {
        background: var(--accent-blue);
        color: white;
    }

    .btn-primary:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }

    .btn-secondary {
        background: var(--secondary-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    .btn-secondary:hover {
        background: var(--accent-bg);
    }

    /* ===== TABLES ===== */
    .table {
        width: 100%;
        border-collapse: collapse;
        background: var(--secondary-bg);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    .table th {
        background: var(--accent-bg);
        color: var(--text-primary);
        font-weight: 600;
        padding: var(--spacing-md);
        text-align: left;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .table td {
        padding: var(--spacing-md);
        border-bottom: 1px solid var(--border-color);
        font-size: 0.875rem;
    }

    .table tr:hover {
        background: var(--accent-bg);
    }

    /* ===== METRICS ===== */
    .metric {
        text-align: center;
        padding: var(--spacing-lg);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
        margin-bottom: var(--spacing-xs);
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-change {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: var(--spacing-xs);
    }

    .change-positive { color: var(--accent-green); }
    .change-negative { color: var(--accent-red); }

    /* ===== STATUS INDICATORS ===== */
    .status {
        display: inline-flex;
        align-items: center;
        gap: var(--spacing-sm);
        padding: var(--spacing-xs) var(--spacing-sm);
        border-radius: var(--radius-md);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-online {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ===== STREAMLIT OVERRIDES ===== */
    /* Remove default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Fix spacing issues */
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: var(--spacing-xl) !important;
    }

    /* Style Streamlit buttons consistently */
    .stButton > button {
        border-radius: var(--radius-md);
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* Style inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
    }

    /* Style metrics */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
    }
    
    /* Ensure all metric text is white */
    [data-testid="metric-container"] {
        color: white !important;
    }
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    [data-testid="metric-container"] label {
        color: white !important;
    }
    [data-testid="metric-container"] > div > div {
        color: white !important;
    }
    [data-testid="metric-container"] * {
        color: white !important;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .grid-2, .grid-3, .grid-4 {
            grid-template-columns: 1fr;
        }
        
        .container {
            padding: 0 var(--spacing-sm);
        }
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def create_metric_card(title, value, change=None, icon=None):
    """Create a professional metric card"""
    change_html = ""
    if change is not None:
        change_class = "change-positive" if change >= 0 else "change-negative"
        change_symbol = "+" if change >= 0 else ""
        change_html = f'<div class="metric-change {change_class}">{change_symbol}{change}</div>'
    
    icon_html = f'<div style="font-size: 2rem; margin-bottom: var(--spacing-sm);">{icon}</div>' if icon else ""
    
    return f"""
    <div class="card">
        {icon_html}
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {change_html}
    </div>
    """

def create_data_table(headers, rows):
    """Create a professional data table"""
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    rows_html = "".join(
        f"<tr>{''.join(f'<td>{cell}</td>' for cell in row)}</tr>"
        for row in rows
    )
    
    return f"""
    <div class="card">
        <div class="table-container">
            <table class="table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
    """

def get_finviz_professional_styles():
    """Professional Finviz-style clean layout optimized for institutional trading interface"""
    return """
    <style>
    /* True Finviz background - proper gray tone like the real site */
    .stApp {
        background-color: #e8e8e8 !important;
    }
    
    .main .block-container {
        background-color: #e8e8e8 !important;
        color: #000000 !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100% !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    
    /* Professional Finviz-style headers */
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
        font-weight: 600 !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
        margin-bottom: 0.5rem !important;
    }
    
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1.0rem !important; }
    
    /* Clean text styling like Finviz */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stText {
        color: #333333 !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
        font-size: 0.9rem !important;
    }
    
    /* Finviz-style professional sidebar */
    section[data-testid="stSidebar"] {
        background-color: #e8e8e8 !important;
        border-right: 2px solid #d0d0d0 !important;
        padding-top: 1rem !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #333333 !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    
    /* Professional Finviz-style buttons */
    .stButton button {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border: 1px solid #888888 !important;
        border-radius: 0px !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.6rem !important;
        transition: all 0.15s ease !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    
    .stButton button:hover {
        background-color: #f8f9fa !important;
        border-color: #adb5bd !important;
        color: #333333 !important;
    }
    
    /* Primary button in Finviz blue accent */
    .stButton button[kind="primary"] {
        background-color: #007bff !important;
        border-color: #007bff !important;
        color: #ffffff !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
        color: #ffffff !important;
    }
    
    /* True Finviz-style table styling with colored rows */
    .stDataFrame {
        background-color: #ffffff !important;
        border: 2px solid #808080 !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
        font-size: 0.75rem !important;
    }
    
    .stDataFrame table {
        border-collapse: collapse !important;
    }
    
    .stDataFrame th {
        background-color: #d0d0d0 !important;
        color: #000000 !important;
        border: 1px solid #808080 !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        padding: 0.3rem !important;
        text-align: left !important;
    }
    
    .stDataFrame td {
        color: #000000 !important;
        border: 1px solid #c0c0c0 !important;
        padding: 0.3rem !important;
        font-size: 0.75rem !important;
        background-color: #ffffff !important;
    }
    
    .stDataFrame tr:nth-child(even) td {
        background-color: #f8f8f8 !important;
    }
    
    /* Professional metric cards exactly like Finviz */
    div[data-testid="metric-container"] {
        background-color: #f8f8f8 !important;
        border: 1px solid #888888 !important;
        border-radius: 0px !important;
        padding: 0.4rem !important;
        color: #000000 !important;
        box-shadow: none !important;
        margin: 0.2rem !important;
    }
    
    div[data-testid="metric-container"] * {
        color: #333333 !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 0.8rem !important;
        color: #6c757d !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }
    
    /* Clean input styling */
    .stTextInput input, .stSelectbox select {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #ced4da !important;
        border-radius: 2px !important;
        font-size: 0.85rem !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    
    /* Finviz-style tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #d0d0d0 !important;
        border-bottom: 2px solid #808080 !important;
        border-radius: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #6c757d !important;
        background-color: transparent !important;
        border-radius: 0 !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        font-family: 'Arial', 'Helvetica', sans-serif !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #007bff !important;
        border-bottom: 2px solid #007bff !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Professional progress bars */
    .stProgress .st-bo {
        background-color: #e9ecef !important;
    }
    
    .stProgress .st-bp {
        background-color: #007bff !important;
    }
    
    /* Clean status messages */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
        border-left: 4px solid #28a745 !important;
        font-size: 0.85rem !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        border: 1px solid #bee5eb !important;
        border-left: 4px solid #17a2b8 !important;
        font-size: 0.85rem !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
        border-left: 4px solid #dc3545 !important;
        font-size: 0.85rem !important;
    }
    
    /* Compact spacing like Finviz */
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    /* Tighter metric card spacing */
    .stColumns {
        gap: 0.3rem !important;
    }
    
    /* Clean column spacing */
    .stColumn {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
    
    /* Professional divider */
    hr {
        border: none !important;
        border-top: 1px solid #dee2e6 !important;
        margin: 1rem 0 !important;
    }
    
    /* Data density optimization */
    .stDataFrame div[data-testid="stDataFrame"] > div {
        font-size: 0.8rem !important;
    }
    
    /* Status indicators */
    .positive { color: #28a745 !important; font-weight: 600 !important; }
    .negative { color: #dc3545 !important; font-weight: 600 !important; }
    .neutral { color: #6c757d !important; font-weight: 500 !important; }
    </style>
    """