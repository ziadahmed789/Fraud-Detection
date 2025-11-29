import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import time

# ==========================================
# ⚙️ إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Fraud Monitor Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص CSS لتحسين شكل الكروت (Metrics)
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .stMetricLabel {font-weight: bold; color: #555;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔌 الاتصال بقاعدة البيانات (SQLAlchemy)
# ==========================================
def get_data():
    try:
        # استخدام SQLAlchemy + PyMySQL
        # التنسيق: mysql+pymysql://user:password@host/database
        db_connection_str = 'mysql+pymysql://root:root@localhost/transactions'
        db_connection = create_engine(db_connection_str)
        
        # هنجيب آخر 2000 عملية عشان الداش بورد تبقى خفيفة وسريعة
        query = "SELECT * FROM clean_data ORDER BY Time DESC LIMIT 2000"
        
        with db_connection.connect() as conn:
            df = pd.read_sql(query, conn)
            
        return df
    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        return pd.DataFrame()

# ==========================================
# 🎛️ القائمة الجانبية (Sidebar Filters)
# ==========================================
st.sidebar.title("🎛️ Controls")
st.sidebar.markdown("---")

# زرار التحديث اليدوي
if st.sidebar.button('🔄 Live Refresh', use_container_width=True):
    st.rerun()

# جلب البيانات
df = get_data()

if not df.empty:
    df['Time'] = pd.to_datetime(df['Time'])

    # -- فلتر المدينة (Location) --
    all_locations = ['All'] + list(df['Location'].unique())
    selected_location = st.sidebar.selectbox("📍 Filter by City:", all_locations)

    if selected_location != 'All':
        df_display = df[df['Location'] == selected_location]
    else:
        df_display = df

    # -- فلتر إظهار الاحتيال فقط --
    show_fraud_only = st.sidebar.checkbox("🚨 Show Fraud Only")
    if show_fraud_only:
        df_display = df_display[df_display['Is_Fraud'] == 'YES']

    st.sidebar.markdown(f"**Showing:** {len(df_display)} Transactions")

    # ==========================================
    # 📊 واجهة الداش بورد الرئيسية
    # ==========================================
    st.title("🛡️ SecurePay | Real-Time Monitor")
    st.markdown(f"Last updated: **{time.strftime('%H:%M:%S')}**")
    st.markdown("---")

    # --- KPIs (مؤشرات الأداء) ---
    total = len(df_display)
    fraud = df_display[df_display['Is_Fraud'] == 'YES']
    fraud_count = len(fraud)
    fraud_amount = fraud['Amount'].sum()
    fraud_percentage = (fraud_count / total * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📦 Total Transactions", f"{total:,}")
    # العداد ده هينور أحمر لو النسبة زادت
    col2.metric("🚨 Fraud Cases", f"{fraud_count}", delta=f"{fraud_percentage:.1f}% Rate", delta_color="inverse")
    col3.metric("💸 Fraud Amount", f"${fraud_amount:,.0f}")
    col4.metric("🏙️ Active Locations", df_display['Location'].nunique())

    # --- الصف الأول: الرسومات البيانية ---
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📉 Transaction Volume & Fraud (Timeline)")
        # Area Chart يوضح حجم التعاملات مع الوقت
        # Resample بالساعة '1h'
        df_time = df_display.set_index('Time').resample('1h')['Amount'].sum().reset_index()
        
        fig_area = px.area(df_display.sort_values('Time'), x='Time', y='Amount', color='Is_Fraud',
                           color_discrete_map={'YES': '#FF4B4B', 'NO': '#00CC96'}, # أحمر وأخضر
                           template="plotly_white") # خلفية بيضاء
        st.plotly_chart(fig_area, use_container_width=True)

    with c2:
        st.subheader("🌍 Fraud by Location")
        if not fraud.empty:
            fraud_by_loc = fraud['Location'].value_counts().reset_index()
            fraud_by_loc.columns = ['Location', 'Count']
            fig_bar = px.bar(fraud_by_loc, x='Location', y='Count', color='Count',
                             color_continuous_scale='Reds', template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("✅ No fraud detected in selected view.")

    # --- الصف الثاني: التحليلات ---
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("📱 Suspicious Devices")
        if not fraud.empty:
            # Donut Chart للأجهزة المستخدمة في الاحتيال
            # التعديل هنا: استخدمنا px.pie بدلاً من px.donut (لأن donut مش موجودة)
            fig_pie = px.pie(fraud, names='Device', title='Devices used in Fraud',
                               hole=0.4, template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available.")

    with c4:
        st.subheader("🏪 High-Risk Merchants")
        if not fraud.empty:
            # Top 5 Merchants
            fraud_merch = fraud['Merchant'].value_counts().head(5).reset_index()
            fraud_merch.columns = ['Merchant', 'Count']
            fig_merch = px.bar(fraud_merch, y='Merchant', x='Count', orientation='h',
                               title="Top 5 Merchants with Fraud", color='Count',
                               color_continuous_scale='Reds', template="plotly_white")
            st.plotly_chart(fig_merch, use_container_width=True)
        else:
            st.info("No data available.")

    # --- جدول البيانات الحية ---
    st.subheader("📋 Recent Suspicious Activity (Live Feed)")
    
    if not fraud.empty:
        # تلوين المبالغ العالية بالأحمر
        styled_df = fraud[['TransactionID', 'UserID', 'Amount', 'Location', 'Time', 'Merchant']].head(10).style\
            .background_gradient(cmap='Reds', subset=['Amount'])\
            .format({'Amount': "${:,.2f}"})
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.success("✅ System Clean. No recent fraud detected.")

    # تحديث تلقائي كل 10 ثواني
    time.sleep(10)
    st.rerun()

else:
    st.warning("⚠️ Waiting for data... Please ensure the Spark Pipeline is running.")