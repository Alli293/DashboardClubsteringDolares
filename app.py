# ============================================================
# DASHBOARD DE ANÁLISIS DE CLUSTERS SALARIALES - DUAL (USD + CRC)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard de Clusters Salariales - Dual (USD + CRC)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS GLOBAL MEJORADO
# ============================================================

st.markdown("""
<style>
    /* ESTILOS ADAPTATIVOS PARA MODO CLARO/OSCURO */
    .main-header {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card-dark {
        background: linear-gradient(145deg, #2d2d2d, #252525);
        border: 1px solid #444;
        color: white;
    }
    
    .currency-toggle {
        background-color: #4ECDC4;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .currency-toggle:hover {
        background-color: #45b7d1;
        transform: translateY(-2px);
    }
    
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    
    /* Animación para transición de moneda */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS DUAL
# ============================================================

@st.cache_data
def load_dual_data():
    """Cargar datos de ambos análisis (USD y CRC)"""
    try:
        # Cargar datos USD
        df_usd = pd.read_csv('analisis_completo_crc.csv')  # Ajustar nombre según tu archivo
        
        # Cargar datos CRC
        df_crc = pd.read_csv('resultados_finales_clusters_crc.csv')  # Archivo de clusters CRC
        
        # Preparar datos USD
        if 'cluster_nombre' in df_usd.columns:
            cluster_usd = df_usd.groupby('cluster_nombre').agg({
                'salario_limpio': ['mean', 'min', 'max', 'count'],
                'Categora_refinada': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).round(0)
            
            cluster_usd.columns = ['salario_promedio', 'salario_min', 'salario_max', 'n_empleos', 'categoria_principal']
            cluster_usd = cluster_usd.sort_values('salario_promedio', ascending=False)
        else:
            cluster_usd = pd.DataFrame()
        
        # Preparar datos CRC
        if 'cluster_final' in df_crc.columns:
            # Asegurar que tenemos la columna de salario CRC
            if 'crc_limpio' not in df_crc.columns and 'salario_limpio' in df_crc.columns:
                df_crc = df_crc.rename(columns={'salario_limpio': 'crc_limpio'})
            
            cluster_crc = df_crc.groupby('cluster_final').agg({
                'crc_limpio': ['mean', 'min', 'max', 'count'],
                'categoria_refinada_crc': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).round(0)
            
            cluster_crc.columns = ['salario_promedio', 'salario_min', 'salario_max', 'n_empleos', 'categoria_principal']
            cluster_crc = cluster_crc.sort_values('salario_promedio', ascending=False)
        else:
            cluster_crc = pd.DataFrame()
        
        return df_usd, df_crc, cluster_usd, cluster_crc
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.info("""
        **Soluciones posibles:**
        1. Asegúrate de que los archivos estén en la misma carpeta que el dashboard
        2. Verifica los nombres de los archivos:
           - Para USD: 'analisis_completo_crc.csv' 
           - Para CRC: 'resultados_finales_clusters_crc.csv'
        3. Si los nombres son diferentes, ajusta el código
        """)
        return None, None, None, None

# ============================================================
# FUNCIONES DE UTILIDAD PARA FORMATO DE MONEDA
# ============================================================

def format_currency(value, currency_type='USD'):
    """Formatear valor según tipo de moneda"""
    if pd.isna(value):
        return "N/A"
    
    if currency_type == 'CRC':
        return f"₡{value:,.0f}"
    else:  # USD
        return f"${value:,.0f}"

def convert_crc_to_usd(crc_value, exchange_rate=550):
    """Convertir CRC a USD usando tasa de cambio aproximada"""
    return crc_value / exchange_rate

def get_currency_data(data_type, df, cluster_summary, exchange_rate=550):
    """Obtener datos formateados según tipo de moneda"""
    if data_type == 'CRC':
        return df, cluster_summary, 'CRC', '₡'
    else:
        # Convertir CRC a USD si es necesario
        if 'crc_limpio' in df.columns:
            df_usd = df.copy()
            df_usd['salario_usd'] = df_usd['crc_limpio'] / exchange_rate
            
            # Actualizar cluster summary para USD
            cluster_usd = cluster_summary.copy()
            cluster_usd['salario_promedio'] = cluster_usd['salario_promedio'] / exchange_rate
            cluster_usd['salario_min'] = cluster_usd['salario_min'] / exchange_rate
            cluster_usd['salario_max'] = cluster_usd['salario_max'] / exchange_rate
            
            return df_usd, cluster_usd, 'USD', '$'
        else:
            return df, cluster_summary, 'USD', '$'

# ============================================================
# INICIALIZACIÓN
# ============================================================

# Título principal con diseño mejorado
st.markdown("""
<div class="main-header fade-in">
    <h1> Dashboard de Análisis de Clusters Salariales</h1>
    <p style="opacity: 0.9; margin-bottom: 0;">Análisis Dual: Colones Costarricenses (CRC) y Dólares (USD)</p>
    <p style="font-size: 14px; opacity: 0.8;">
        Clustering inteligente de empleos por categoría y salario • Tasa de cambio: ₡550 = $1 USD
    </p>
</div>
""", unsafe_allow_html=True)

# Cargar datos
df_usd, df_crc, cluster_usd, cluster_crc = load_dual_data()

if df_usd is None or df_crc is None:
    st.stop()

# ============================================================
# SIDEBAR - CONFIGURACIÓN DUAL
# ============================================================

with st.sidebar:
    st.header(" Configuración del Dashboard")
    
    # Selector de moneda
    st.subheader(" Moneda de Visualización")
    currency_type = st.radio(
        "Selecciona la moneda para visualizar:",
        ["Colones Costarricenses (CRC)", "Dólares Estadounidenses (USD)"],
        index=0,
        help="Visualiza los datos en colones o en dólares (conversión automática)"
    )
    
    # Tasa de cambi personalizable
    st.subheader(" Tasa de Cambio")
    exchange_rate = st.slider(
        "Tasa de cambio CRC/USD:",
        min_value=500,
        max_value=600,
        value=550,
        step=10,
        help="Ajusta la tasa de cambio para la conversión CRC → USD"
    )
    
    st.markdown(f"**Equivalencia:** ₡1,000,000 ≈ ${(1000000/exchange_rate):,.0f} USD")
    
    # Selección de dataset principal
    st.subheader(" Dataset Principal")
    dataset_choice = st.radio(
        "Dataset para análisis:",
        ["CRC Original (mejor clusters)", "USD Convertido"],
        index=0,
        help="Selecciona el dataset base para el análisis"
    )
    
    st.markdown("---")
    
    # Configuración de análisis
    st.subheader(" Configuración de Análisis")
    
    # Filtro por rango salarial dinámico
    st.write("**Filtro Salarial:**")
    
    # Determinar el dataset activo para filtros
    if dataset_choice == "CRC Original (mejor clusters)":
        active_df = df_crc.copy()
        salary_col = 'crc_limpio'
        default_max = int(df_crc[salary_col].max())
        default_min = int(df_crc[salary_col].min())
    else:
        active_df = df_usd.copy()
        salary_col = 'salario_limpio'
        default_max = int(df_usd[salary_col].max())
        default_min = int(df_usd[salary_col].min())
    
    min_salary, max_salary = st.slider(
        f"Rango salarial ({'CRC' if dataset_choice == 'CRC Original (mejor clusters)' else 'USD'}):",
        min_value=default_min,
        max_value=default_max,
        value=(default_min, default_max),
        step=100 if dataset_choice == 'CRC Original (mejor clusters)' else 1
    )
    
    # Filtro por número de clusters
    st.write("**Número de Clusters:**")
    n_clusters = st.slider(
        "Máximo clusters a mostrar:",
        min_value=3,
        max_value=15,
        value=7,
        step=1
    )
    
    # Opciones de visualización
    st.write("**Opciones de Visualización:**")
    show_details = st.checkbox("Mostrar detalles avanzados", value=True)
    use_animations = st.checkbox("Animaciones en gráficos", value=True)
    color_scheme = st.selectbox(
        "Paleta de colores:",
        ["viridis", "plasma", "inferno", "Set3", "Set2", "tab20"],
        index=0
    )
    
    st.markdown("---")
    
    # Métricas rápidas en sidebar
    st.subheader(" Resumen Rápido")
    
    if dataset_choice == "CRC Original (mejor clusters)":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Empleos", len(df_crc))
        with col2:
            st.metric("Clusters", len(cluster_crc))
        
        if currency_type == "Colones Costarricenses (CRC)":
            st.metric("Salario Promedio", f"₡{df_crc['crc_limpio'].mean():,.0f}")
            st.metric("Salario Máximo", f"₡{df_crc['crc_limpio'].max():,.0f}")
        else:
            st.metric("Salario Promedio", f"${(df_crc['crc_limpio'].mean()/exchange_rate):,.0f}")
            st.metric("Salario Máximo", f"${(df_crc['crc_limpio'].max()/exchange_rate):,.0f}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Empleos", len(df_usd))
        with col2:
            st.metric("Clusters", len(cluster_usd))
        
        st.metric("Salario Promedio", f"${df_usd['salario_limpio'].mean():,.0f}")
        st.metric("Salario Máximo", f"${df_usd['salario_limpio'].max():,.0f}")

# ============================================================
# FILTRADO DE DATOS Y PREPARACIÓN
# ============================================================

# Determinar dataset activo
if dataset_choice == "CRC Original (mejor clusters)":
    df_active = df_crc.copy()
    cluster_active = cluster_crc.copy()
    salary_column = 'crc_limpio'
    cluster_column = 'cluster_final'
    currency_symbol = '₡' if currency_type == "Colones Costarricenses (CRC)" else '$'
    is_crc_original = True
else:
    df_active = df_usd.copy()
    cluster_active = cluster_usd.copy()
    salary_column = 'salario_limpio'
    cluster_column = 'cluster_nombre'
    currency_symbol = '$'
    is_crc_original = False

# Aplicar filtros
df_filtered = df_active[
    (df_active[salary_column] >= min_salary) & 
    (df_active[salary_column] <= max_salary)
]

# Limitar clusters mostrados
if len(cluster_active) > n_clusters:
    cluster_filtered = cluster_active.head(n_clusters)
else:
    cluster_filtered = cluster_active

# Convertir a USD si se solicita
if currency_type == "Dólares Estadounidenses (USD)" and is_crc_original:
    df_filtered = df_filtered.copy()
    df_filtered['salary_display'] = df_filtered[salary_column] / exchange_rate
    
    cluster_filtered = cluster_filtered.copy()
    cluster_filtered['salario_promedio'] = cluster_filtered['salario_promedio'] / exchange_rate
    cluster_filtered['salario_min'] = cluster_filtered['salario_min'] / exchange_rate
    cluster_filtered['salario_max'] = cluster_filtered['salario_max'] / exchange_rate
    salary_column = 'salary_display'
else:
    df_filtered['salary_display'] = df_filtered[salary_column]

# ============================================================
# SECCIÓN 1: RESUMEN EJECUTIVO DUAL
# ============================================================

st.header(" Resumen Ejecutivo Dual")

# Crear columnas para métricas
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_salary = df_filtered[salary_column].mean()
    st.metric(
        label="Salario Promedio",
        value=f"{currency_symbol}{avg_salary:,.0f}",
        delta=f"Moneda: {currency_type.split('(')[-1].replace(')', '')}"
    )

with col2:
    st.metric(
        label="Empleos Analizados",
        value=f"{len(df_filtered)}",
        delta=f"{len(df_filtered)/len(df_active)*100:.1f}% del total"
    )

with col3:
    top_cluster = cluster_filtered.iloc[0]
    st.metric(
        label="Cluster Mejor Pagado",
        value=top_cluster.name,
        delta=f"{currency_symbol}{top_cluster['salario_promedio']:,.0f}"
    )

with col4:
    if len(cluster_filtered) > 1:
        best_value_cluster = cluster_filtered.iloc[1]
    else:
        best_value_cluster = cluster_filtered.iloc[0]
    
    st.metric(
        label="Mejor Relación Valor",
        value=best_value_cluster.name,
        delta=f"{best_value_cluster['n_empleos']} empleos"
    )

# Información adicional sobre conversión
if currency_type == "Dólares Estadounidenses (USD)" and is_crc_original:
    st.info(f"""
    **Información de Conversión:**
    - Tasa de cambio utilizada: ₡{exchange_rate} = $1 USD
    - Salario promedio original: ₡{df_crc['crc_limpio'].mean():,.0f}
    - Salario máximo original: ₡{df_crc['crc_limpio'].max():,.0f}
    """)

st.markdown("---")

# ============================================================
# SECCIÓN 2: COMPARACIÓN DUAL CRC vs USD
# ============================================================

st.header(" Comparación Dual: CRC vs USD")

col_dual1, col_dual2 = st.columns(2)

with col_dual1:
    # Comparativa de rangos salariales
    st.subheader("Rangos Salariales Comparados")
    
    comparison_data = []
    for cluster_id, row in cluster_filtered.iterrows():
        # Datos en CRC (original)
        if is_crc_original:
            crc_avg = row['salario_promedio'] * exchange_rate if currency_type == "Dólares Estadounidenses (USD)" else row['salario_promedio']
            usd_avg = row['salario_promedio'] if currency_type == "Dólares Estadounidenses (USD)" else row['salario_promedio'] / exchange_rate
        else:
            crc_avg = row['salario_promedio'] * exchange_rate
            usd_avg = row['salario_promedio']
        
        comparison_data.append({
            'Cluster': cluster_id,
            'CRC Promedio': crc_avg,
            'USD Promedio': usd_avg,
            'Empleos': row['n_empleos']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Gráfico de comparación
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='CRC',
        x=comparison_df['Cluster'],
        y=comparison_df['CRC Promedio'],
        text=[f"₡{x:,.0f}" for x in comparison_df['CRC Promedio']],
        textposition='outside',
        marker_color='#2E86AB'
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='USD',
        x=comparison_df['Cluster'],
        y=comparison_df['USD Promedio'],
        text=[f"${x:,.0f}" for x in comparison_df['USD Promedio']],
        textposition='outside',
        marker_color='#F18F01'
    ))
    
    fig_comparison.update_layout(
        height=400,
        title="Comparación Salarial CRC vs USD",
        barmode='group',
        xaxis_title="Cluster",
        yaxis_title="Salario Promedio",
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)

with col_dual2:
    # Tabla de conversión detallada
    st.subheader("Tabla de Conversión por Cluster")
    
    display_comparison = comparison_df.copy()
    display_comparison['CRC Promedio'] = display_comparison['CRC Promedio'].apply(lambda x: f"₡{x:,.0f}")
    display_comparison['USD Promedio'] = display_comparison['USD Promedio'].apply(lambda x: f"${x:,.0f}")
    display_comparison['Tasa Aplicada'] = f"₡{exchange_rate}"
    
    st.dataframe(
        display_comparison[['Cluster', 'CRC Promedio', 'USD Promedio', 'Empleos', 'Tasa Aplicada']],
        use_container_width=True,
        hide_index=True
    )
    
    # Estadísticas de conversión
    st.subheader("Estadísticas de Conversión")
    
    if is_crc_original:
        original_avg_crc = df_crc['crc_limpio'].mean()
        converted_avg_usd = original_avg_crc / exchange_rate
    else:
        original_avg_usd = df_usd['salario_limpio'].mean()
        converted_avg_crc = original_avg_usd * exchange_rate
    
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        if is_crc_original:
            st.metric("Promedio CRC", f"₡{original_avg_crc:,.0f}")
        else:
            st.metric("Promedio USD", f"${original_avg_usd:,.0f}")
    
    with col_stats2:
        if is_crc_original:
            st.metric("Equivalente USD", f"${converted_avg_usd:,.0f}")
        else:
            st.metric("Equivalente CRC", f"₡{converted_avg_crc:,.0f}")

st.markdown("---")

# ============================================================
# SECCIÓN 3: PIRÁMIDE SALARIAL INTERACTIVA DUAL
# ============================================================

st.header(" Pirámide Salarial por Cluster")

col_pyramid1, col_pyramid2 = st.columns([2, 1])

with col_pyramid1:
    # Preparar datos para la pirámide
    clusters_ordered = cluster_filtered.sort_values('salario_promedio', ascending=True).index
    salarios_avg = cluster_filtered.loc[clusters_ordered]['salario_promedio'].values
    counts = cluster_filtered.loc[clusters_ordered]['n_empleos'].values
    categorias = cluster_filtered.loc[clusters_ordered]['categoria_principal'].values
    
    # Crear gráfico de barras horizontales
    fig_pyramid = go.Figure()
    
    # Añadir barras
    fig_pyramid.add_trace(go.Bar(
        y=[str(c) for c in clusters_ordered],
        x=salarios_avg,
        orientation='h',
        marker_color=px.colors.sequential.Viridis_r[:len(clusters_ordered)],
        text=[f"{currency_symbol}{s:,.0f}" for s in salarios_avg],
        textposition='outside',
        name='Salario Promedio',
        hovertemplate='<b>%{y}</b><br>' +
                     f'Salario: {currency_symbol}%{{x:,.0f}}<br>' +
                     'Empleos: %{customdata[0]}<br>' +
                     'Categoría: %{customdata[1]}<extra></extra>',
        customdata=np.column_stack((counts, categorias))
    ))
    
    # Añadir línea de promedio general
    promedio_general = df_filtered[salary_column].mean()
    fig_pyramid.add_vline(
        x=promedio_general,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio General: {currency_symbol}{promedio_general:,.0f}",
        annotation_position="top right"
    )
    
    # Configurar layout
    fig_pyramid.update_layout(
        height=500,
        title=f"Distribución Salarial por Cluster ({currency_type.split('(')[-1].replace(')', '')})",
        xaxis_title=f"Salario Promedio Mensual ({currency_symbol})",
        yaxis_title="Cluster",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_pyramid, use_container_width=True)

with col_pyramid2:
    # Gráfico de donut para distribución
    fig_donut = go.Figure()
    
    fig_donut.add_trace(go.Pie(
        labels=[str(c) for c in cluster_filtered.index],
        values=cluster_filtered['n_empleos'],
        hole=0.4,
        marker_colors=px.colors.sequential.Viridis[:len(cluster_filtered)],
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>' +
                     'Empleos: %{value}<br>' +
                     'Porcentaje: %{percent}<br>' +
                     f'Salario: {currency_symbol}%{{customdata[0]:,.0f}}<extra></extra>',
        customdata=cluster_filtered['salario_promedio'].values
    ))
    
    fig_donut.update_layout(
        height=400,
        title="Distribución de Empleos",
        showlegend=False,
        margin=dict(t=40, b=0, l=0, r=0),
        annotations=[dict(
            text=f"Total: {len(df_filtered)}",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)
    
    # Mostrar tabla resumen
    st.subheader("Resumen por Cluster")
    summary_table = cluster_filtered.copy()
    summary_table['salario_promedio'] = summary_table['salario_promedio'].apply(lambda x: f"{currency_symbol}{x:,.0f}")
    st.dataframe(
        summary_table[['salario_promedio', 'n_empleos', 'categoria_principal']],
        use_container_width=True,
        height=300
    )

st.markdown("---")

# ============================================================
# SECCIÓN 4: ANÁLISIS DE ESTABILIDAD SALARIAL
# ============================================================

st.header(" Análisis de Salario vs Estabilidad")

# Calcular métricas de estabilidad
stability_data = []
for cluster, row in cluster_filtered.iterrows():
    cluster_df = df_filtered[df_filtered[cluster_column] == cluster] if cluster_column in df_filtered.columns else pd.DataFrame()
    
    if len(cluster_df) > 0:
        # Calcular estabilidad (menor rango = mayor estabilidad)
        salary_range = row['salario_max'] - row['salario_min']
        stability_score = 100 - (salary_range / row['salario_promedio']) * 20 if row['salario_promedio'] > 0 else 0
        
        stability_data.append({
            'cluster': str(cluster),
            'salario_promedio': row['salario_promedio'],
            'n_empleos': row['n_empleos'],
            'estabilidad': min(100, max(0, stability_score)),
            'rango_salarial': salary_range,
            'categoria': row['categoria_principal']
        })

if stability_data:
    stability_df = pd.DataFrame(stability_data)
    
    # Crear gráfico de burbujas
    fig_bubbles = px.scatter(
        stability_df,
        x='salario_promedio',
        y='estabilidad',
        size='n_empleos',
        color='cluster',
        hover_name='cluster',
        hover_data=['categoria', 'rango_salarial'],
        size_max=60,
        color_discrete_sequence=px.colors.qualitative.Set3[:len(stability_df)],
        title=f"Relación Salario-Estabilidad por Cluster ({currency_type.split('(')[-1].replace(')', '')})"
    )
    
    # Añadir líneas de referencia
    fig_bubbles.add_vline(
        x=promedio_general,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Promedio General"
    )
    
    fig_bubbles.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Límite Estabilidad"
    )
    
    # Mejorar layout
    fig_bubbles.update_layout(
        height=600,
        xaxis_title=f"Salario Promedio ({currency_symbol})",
        yaxis_title="Índice de Estabilidad (0-100)",
        hovermode='closest',
        showlegend=True
    )
    
    st.plotly_chart(fig_bubbles, use_container_width=True)
    
    # Análisis de estabilidad
    col_stab1, col_stab2 = st.columns(2)
    
    with col_stab1:
        st.subheader("Clusters más Estables")
        if len(stability_df) > 0:
            stable_clusters = stability_df.nlargest(3, 'estabilidad')
            for _, row in stable_clusters.iterrows():
                st.metric(
                    label=row['cluster'],
                    value=f"{row['estabilidad']:.1f} puntos",
                    delta=f"{currency_symbol}{row['salario_promedio']:,.0f}"
                )
    
    with col_stab2:
        st.subheader("Clusters con Mayor Salario")
        if len(stability_df) > 0:
            high_salary_clusters = stability_df.nlargest(3, 'salario_promedio')
            for _, row in high_salary_clusters.iterrows():
                st.metric(
                    label=row['cluster'],
                    value=f"{currency_symbol}{row['salario_promedio']:,.0f}",
                    delta=f"{row['estabilidad']:.1f} estabilidad"
                )
else:
    st.warning("No hay suficientes datos para calcular la estabilidad por cluster.")

st.markdown("---")

# ============================================================
# SECCIÓN 5: TRAYECTORIAS PROFESIONALES DUAL
# ============================================================

st.header(" Trayectorias Profesionales y Evolución Salarial")

# Definir trayectorias basadas en clusters (ajustar según tus datos)
trayectorias = {
    'Asistente → Director (Ciencias)': {
        'etapas': ['Asistente', 'Coordinador', 'Gerente', 'Director'],
        'salarios_crc': [531250, 1888889, 3699777, 5764062],
        'clusters': ['C5', 'C5', 'C3/C4', 'C1'],
        'color': '#2E86AB'
    },
    'Desarrollador Senior → Líder': {
        'etapas': ['Senior Dev', 'Tech Lead'],
        'salarios_crc': [3825000, 6120000],
        'clusters': ['C2', 'C1'],
        'color': '#A23B72'
    },
    'Académico → Industria': {
        'etapas': ['Académico', 'Industria Entry', 'Industria Senior'],
        'salarios_crc': [237354, 1500000, 3500000],
        'clusters': ['C7', 'C5', 'C3'],
        'color': '#F18F01'
    }
}

# Crear gráfico de líneas
fig_trayectorias = go.Figure()

# Añadir cada trayectoria
for nombre, datos in trayectorias.items():
    # Convertir salarios según moneda seleccionada
    if currency_type == "Dólares Estadounidenses (USD)":
        salarios = [s / exchange_rate for s in datos['salarios_crc']]
    else:
        salarios = datos['salarios_crc']
    
    fig_trayectorias.add_trace(go.Scatter(
        x=datos['etapas'],
        y=salarios,
        mode='lines+markers+text',
        name=nombre,
        line=dict(color=datos['color'], width=3),
        marker=dict(size=12),
        text=[f"{currency_symbol}{s:,.0f}" for s in salarios],
        textposition="top center",
        hoverinfo='text+name',
        hovertext=[f"{etapa}<br>Salario: {currency_symbol}{salario:,.0f}<br>Cluster: {cluster}" 
                  for etapa, salario, cluster in zip(datos['etapas'], salarios, datos['clusters'])]
    ))

# Configurar layout
fig_trayectorias.update_layout(
    height=500,
    title=f"Evolución Salarial en Trayectorias Profesionales ({currency_type.split('(')[-1].replace(')', '')})",
    xaxis_title="Etapa Profesional",
    yaxis_title=f"Salario Mensual ({currency_symbol})",
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_trayectorias, use_container_width=True)

# Mostrar estadísticas de crecimiento
st.subheader("Crecimiento Salarial por Trayectoria")
cols_tray = st.columns(len(trayectorias))

for idx, (nombre, datos) in enumerate(trayectorias.items()):
    with cols_tray[idx]:
        if currency_type == "Dólares Estadounidenses (USD)":
            salarios = [s / exchange_rate for s in datos['salarios_crc']]
        else:
            salarios = datos['salarios_crc']
        
        if len(salarios) > 1:
            crecimiento = ((salarios[-1] - salarios[0]) / salarios[0]) * 100
            st.metric(
                label=nombre,
                value=f"+{crecimiento:.0f}%",
                delta=f"De {currency_symbol}{salarios[0]:,.0f} a {currency_symbol}{salarios[-1]:,.0f}"
            )

st.markdown("---")

# ============================================================
# SECCIÓN 6: DISTRIBUCIÓN DE CATEGORÍAS
# ============================================================

st.header(" Distribución de Categorías por Cluster")

# Determinar columna de categoría
categoria_col = 'categoria_refinada_crc' if is_crc_original else 'Categora_refinada'

if categoria_col in df_filtered.columns:
    # Preparar datos para heatmap
    top_categories = df_filtered[categoria_col].value_counts().head(10).index.tolist()
    heatmap_data = []
    
    for cluster in cluster_filtered.index:
        if cluster_column in df_filtered.columns:
            cluster_df = df_filtered[df_filtered[cluster_column] == cluster]
        else:
            cluster_df = pd.DataFrame()
        
        row = []
        for category in top_categories:
            count = len(cluster_df[cluster_df[categoria_col] == category])
            row.append(count)
        heatmap_data.append(row)
    
    # Crear heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=top_categories,
        y=[str(c) for c in cluster_filtered.index],
        colorscale='YlOrRd',
        hoverongaps=False,
        text=heatmap_data,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>Cluster: %{y}</b><br>' +
                     '<b>Categoría: %{x}</b><br>' +
                     'Empleos: %{z}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        height=500,
        title="Concentración de Categorías por Cluster",
        xaxis_title="Categorías Laborales",
        yaxis_title="Clusters",
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Distribución de categorías principales
    st.subheader("Categorías Principales por Cluster")
    for cluster, row in cluster_filtered.iterrows():
        col_cat1, col_cat2 = st.columns([1, 4])
        with col_cat1:
            st.markdown(f"**{cluster}**")
        with col_cat2:
            st.write(f"{row['categoria_principal']} ({currency_symbol}{row['salario_promedio']:,.0f} promedio)")
else:
    st.warning(f"No se encontró la columna de categorías: {categoria_col}")

st.markdown("---")

# ============================================================
# SECCIÓN 7: PERFIL DETALLADO DE CLUSTERS
# ============================================================

st.header(" Perfil Detallado de Clusters")

# Seleccionar cluster para análisis detallado
if len(cluster_filtered) > 0:
    selected_cluster = st.selectbox(
        "Selecciona un cluster para análisis detallado:",
        options=[str(c) for c in cluster_filtered.index]
    )
    
    if selected_cluster:
        # Obtener datos del cluster seleccionado
        cluster_data = cluster_filtered.loc[selected_cluster]
        
        if cluster_column in df_filtered.columns:
            cluster_jobs = df_filtered[df_filtered[cluster_column] == selected_cluster]
        else:
            cluster_jobs = pd.DataFrame()
        
        # Mostrar métricas del cluster
        col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
        
        with col_detail1:
            st.metric(
                "Salario Promedio",
                f"{currency_symbol}{cluster_data['salario_promedio']:,.0f}",
                f"Rango: {currency_symbol}{cluster_data['salario_min']:,.0f} - {currency_symbol}{cluster_data['salario_max']:,.0f}"
            )
        
        with col_detail2:
            st.metric("Total Empleos", cluster_data['n_empleos'])
        
        with col_detail3:
            # Calcular estabilidad
            salary_range = cluster_data['salario_max'] - cluster_data['salario_min']
            stability = 100 - (salary_range / cluster_data['salario_promedio']) * 20 if cluster_data['salario_promedio'] > 0 else 0
            st.metric("Estabilidad", f"{stability:.1f}%")
        
        with col_detail4:
            if categoria_col in df_filtered.columns and len(cluster_jobs) > 0:
                unique_cats = cluster_jobs[categoria_col].nunique()
                diversity = (unique_cats / len(cluster_jobs)) * 100
                st.metric("Diversidad", f"{diversity:.1f}%")
            else:
                st.metric("Categoría Principal", cluster_data['categoria_principal'])
        
        # Mostrar empleos del cluster
        if len(cluster_jobs) > 0:
            st.subheader(f"Empleos en Cluster {selected_cluster}")
            
            # Crear dataframe para mostrar
            display_jobs = cluster_jobs.copy()
            
            if 'Título' in display_jobs.columns:
                display_cols = ['Título', 'Empresa', categoria_col, salary_column]
            elif 'Categora' in display_jobs.columns:
                display_cols = ['Categora', 'Empresa', salary_column]
            else:
                display_cols = [c for c in display_jobs.columns if c not in [cluster_column, 'crc_limpio', 'salario_limpio']][:4] + [salary_column]
            
            # Formatear salario
            display_jobs['Salario'] = display_jobs[salary_column].apply(lambda x: f"{currency_symbol}{x:,.0f}")
            
            st.dataframe(
                display_jobs[display_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Mostrar estadísticas adicionales
            if len(cluster_jobs) > 5:
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.write("**Distribución de Categorías:**")
                    if categoria_col in cluster_jobs.columns:
                        cat_counts = cluster_jobs[categoria_col].value_counts().head(5)
                        for cat, count in cat_counts.items():
                            percentage = (count / len(cluster_jobs)) * 100
                            st.write(f"• {cat}: {count} ({percentage:.1f}%)")
                
                with col_stats2:
                    st.write("**Estadísticas Salariales:**")
                    st.write(f"• Mediana: {currency_symbol}{cluster_jobs[salary_column].median():,.0f}")
                    st.write(f"• Desviación: {currency_symbol}{cluster_jobs[salary_column].std():,.0f}")
                    st.write(f"• Coef. Variación: {(cluster_jobs[salary_column].std() / cluster_jobs[salary_column].mean() * 100):.1f}%")
else:
    st.warning("No hay clusters disponibles para análisis detallado.")

st.markdown("---")

# ============================================================
# SECCIÓN 8: RECOMENDACIONES Y PLAN DE ACCIÓN
# ============================================================

st.header(" Ejemplo practicos de aplicaciones de este cluster")

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.subheader("Para Buscadores de Empleo")
    
    if len(cluster_filtered) >= 3:
        # Recomendar clusters según nivel
        top_cluster = cluster_filtered.iloc[0]
        mid_cluster = cluster_filtered.iloc[len(cluster_filtered)//2]
        entry_cluster = cluster_filtered.iloc[-1]
        
        st.info(f"""
        **📍 Según tu nivel de experiencia:**
        
        **Experiencia Avanzada (5+ años):**
        • **Cluster {top_cluster.name}** - {currency_symbol}{top_cluster['salario_promedio']:,.0f}
        • {top_cluster['categoria_principal']}
        
        **Experiencia Intermedia (2-5 años):**
        • **Cluster {mid_cluster.name}** - {currency_symbol}{mid_cluster['salario_promedio']:,.0f}
        • {mid_cluster['categoria_principal']}
        
        **Entry-Level (0-2 años):**
        • **Cluster {entry_cluster.name}** - {currency_symbol}{entry_cluster['salario_promedio']:,.0f}
        • {entry_cluster['categoria_principal']}
        """)

with col_rec2:
    st.subheader("Para Negociación Salarial")
    
    if len(stability_df) > 0:
        # Encontrar el mejor balance salario/estabilidad
        stability_df['score'] = (
            stability_df['salario_promedio'] / stability_df['salario_promedio'].max() * 0.6 +
            stability_df['estabilidad'] / 100 * 0.4
        )
        
        best_balance = stability_df.loc[stability_df['score'].idxmax()]
        
        st.success(f"""
        ** Estrategia de Negociación:**
        
        **Para máximo salario:**
        • Objetivo: **{currency_symbol}{cluster_filtered.iloc[0]['salario_promedio']:,.0f}+**
        • Cluster: {cluster_filtered.iloc[0].name}
        
        **Para mejor balance:**
        • Objetivo: **{currency_symbol}{best_balance['salario_promedio']:,.0f}**
        • Cluster: {best_balance['cluster']}
        • Estabilidad: {best_balance['estabilidad']:.1f}%
        
        **Rango recomendado por cluster:**
        • Usar percentil 75% del rango como objetivo
        """)

# Plan de mejora continua
st.subheader(" Plan de Mejora del Análisis")
col_plan1, col_plan2, col_plan3 = st.columns(3)

with col_plan1:
    st.markdown("""
    **🔴 ALTA PRIORIDAD:**
    1. Aumentar dataset a >200 muestras
    2. Añadir variable "años experiencia"
    3. Validar outliers extremos
    """)

with col_plan2:
    st.markdown("""
    **🟡 MEDIA PRIORIDAD:**
    1. Segmentar por ubicación geográfica
    2. Incluir datos de otras fuentes
    3. Añadir análisis de habilidades
    """)

with col_plan3:
    st.markdown("""
    **🟢 BAJA PRIORIDAD:**
    1. Análisis temporal (evolución)
    2. Comparación con benchmarks
    3. Modelo predictivo salarial
    """)

st.markdown("---")

# ============================================================
# SECCIÓN 9: EXPORTACIÓN DE DATOS DUAL
# ============================================================

st.header(" Exportación de Resultados")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    # Exportar datos filtrados
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Datos Filtrados",
        data=csv_data,
        file_name=f"datos_filtrados_{currency_type[:3]}.csv",
        mime="text/csv",
        help="Exporta los datos actualmente filtrados"
    )

with col_exp2:
    # Exportar resumen de clusters
    summary_export = cluster_filtered.copy()
    summary_export.index = [str(c) for c in summary_export.index]
    csv_summary = summary_export.to_csv(index=True).encode('utf-8')
    st.download_button(
        label=" Resumen Clusters",
        data=csv_summary,
        file_name=f"resumen_clusters_{currency_type[:3]}.csv",
        mime="text/csv",
        help="Exporta el resumen estadístico de clusters"
    )

with col_exp3:
    # Exportar análisis de trayectorias
    trayectorias_export = []
    for nombre, datos in trayectorias.items():
        if currency_type == "Dólares Estadounidenses (USD)":
            salarios = [s / exchange_rate for s in datos['salarios_crc']]
        else:
            salarios = datos['salarios_crc']
        
        trayectorias_export.append({
            'Trayectoria': nombre,
            'Etapas': ' → '.join(datos['etapas']),
            'Clusters': ' → '.join(datos['clusters']),
            'Salario Inicial': salarios[0],
            'Salario Final': salarios[-1],
            'Crecimiento %': ((salarios[-1] - salarios[0]) / salarios[0]) * 100,
            'Moneda': currency_type.split('(')[-1].replace(')', '')
        })
    
    trayectorias_df = pd.DataFrame(trayectorias_export)
    csv_trayectorias = trayectorias_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Trayectorias",
        data=csv_trayectorias,
        file_name=f"trayectorias_{currency_type[:3]}.csv",
        mime="text/csv",
        help="Exporta el análisis de trayectorias profesionales"
    )

# ============================================================
# FOOTER CON INFORMACIÓN DUAL
# ============================================================

st.markdown("---")

st.markdown(f"""
** Información del Análisis Actual:**

**Dataset:** {dataset_choice}
**Moneda:** {currency_type}
**Tasa CRC/USD:** ₡{exchange_rate} = $1
**Empleos analizados:** {len(df_filtered)} de {len(df_active)} totales
**Clusters mostrados:** {len(cluster_filtered)} de {len(cluster_active)} totales
**Rango salarial filtrado:** {currency_symbol}{min_salary:,.0f} - {currency_symbol}{max_salary:,.0f}

** Métricas Clave:**
• Salario promedio filtrado: {currency_symbol}{df_filtered[salary_column].mean():,.0f}
• Cluster mejor pagado: {cluster_filtered.iloc[0].name} ({currency_symbol}{cluster_filtered.iloc[0]['salario_promedio']:,.0f})
• Total empleos en clusters mostrados: {cluster_filtered['n_empleos'].sum()}

**Limitaciones y Consideraciones:**
1. Basado en {len(df_active)} muestras totales
2. Validar con datos adicionales para decisiones críticas
3. Considerar factores adicionales: experiencia, educación, ubicación
4. Actualizar análisis periódicamente con nuevos datos
""")

# Información sobre archivos generados
with st.expander(" Archivos Disponibles para Descarga"):
    st.write("""
    **Archivos CSV disponibles en el sistema:**
    1. **analisis_completo_crc.csv** - Dataset completo con análisis CRC
    2. **resultados_finales_clusters_crc.csv** - Clusters finales CRC
    3. **empleos_clusterizados_refinados_crc.csv** - Clusters refinados
    4. **resumen_clusters_crc.csv** - Resumen estadístico
    
    **Archivos de reporte:**
    1. **REPORTE_FINAL_CLUSTERS_CRC.txt** - Reporte ejecutivo completo
    2. **PLAN_ACCION_CLUSTERS_CRC.txt** - Plan de acción detallado
    
    **Visualizaciones generadas:**
    1. **analisis_final_clusters_crc.png** - Gráficos finales
    2. **piramide_salarial_profesional.png** - Pirámide salarial
    3. **clustering_mejorado_crc.png** - Clustering mejorado
    """)
