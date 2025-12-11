# ============================================================
# DASHBOARD DE ANÁLISIS DE CLUSTERS SALARIALES - DUAL (USD + CRC)
# VERSIÓN CON IDENTIFICADORES POR CATEGORÍA PREDOMINANTE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard de Análisis de Clusters Salariales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS GLOBAL MEJORADO
# ============================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .cluster-label {
        font-weight: bold;
        background: linear-gradient(45deg, #4ECDC4, #45B7D1);
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
    
    .category-tag {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 4px 12px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
    
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
        # Cargar datos CRC (archivo principal)
        df_crc = pd.read_csv('resultados_finales_clusters_crc.csv')
        
        # Cargar datos refinados si existen
        try:
            df_refinado = pd.read_csv('empleos_clusterizados_refinados_crc.csv')
            # Si tiene mejor estructura, usarlo como principal
            if 'cluster_final' in df_refinado.columns:
                df_crc = df_refinado
        except:
            pass
        
        # Preparar datos CRC - Verificar columnas reales
        # Tu dataset tiene 'cluster_final' o 'cluster_nombre'
        cluster_col = 'cluster_final' if 'cluster_final' in df_crc.columns else 'cluster_nombre'
        salary_col = 'crc_limpio' if 'crc_limpio' in df_crc.columns else 'salario_limpio'
        categoria_col = 'categoria_refinada_crc' if 'categoria_refinada_crc' in df_crc.columns else 'Categora_refinada'
        
        # Agrupar para obtener categoría predominante por cluster
        cluster_crc = df_crc.groupby(cluster_col).agg({
            salary_col: ['mean', 'min', 'max', 'count'],
            categoria_col: lambda x: x.value_counts().index[0] if len(x) > 0 else 'Sin categoría'
        }).round(0)
        
        cluster_crc.columns = ['salario_promedio', 'salario_min', 'salario_max', 'n_empleos', 'categoria_principal']
        cluster_crc = cluster_crc.sort_values('salario_promedio', ascending=False)
        
        # Crear identificador descriptivo para cada cluster
        cluster_crc['cluster_id_descriptivo'] = cluster_crc.apply(
            lambda row: f"{row['categoria_principal'][:25]}... (₡{row['salario_promedio']:,.0f})" 
            if len(row['categoria_principal']) > 25 else 
            f"{row['categoria_principal']} (₡{row['salario_promedio']:,.0f})", axis=1
        )
        
        # Crear identificador corto para visualizaciones
        cluster_crc['cluster_id_corto'] = cluster_crc.apply(
            lambda row: f"#{row.name} • {row['categoria_principal'].split('(')[0].strip()[:15]}..." 
            if '(' in row['categoria_principal'] else 
            f"#{row.name} • {row['categoria_principal'][:15]}...", axis=1
        )
        
        # Simular datos USD convirtiendo CRC
        df_usd = df_crc.copy()
        if salary_col in df_usd.columns:
            df_usd['salario_usd'] = df_usd[salary_col] / 550  # Conversión aproximada
        
        # Crear cluster summary para USD
        cluster_usd = cluster_crc.copy()
        cluster_usd['salario_promedio'] = cluster_usd['salario_promedio'] / 550
        cluster_usd['salario_min'] = cluster_usd['salario_min'] / 550
        cluster_usd['salario_max'] = cluster_usd['salario_max'] / 550
        
        return df_usd, df_crc, cluster_usd, cluster_crc
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.info("""
        **Archivos esperados:**
        1. resultados_finales_clusters_crc.csv (PRINCIPAL)
        2. empleos_clusterizados_refinados_crc.csv (OPCIONAL)
        """)
        return None, None, None, None

# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

def crear_etiqueta_cluster(row, moneda='CRC'):
    """Crear etiqueta descriptiva para cluster"""
    if moneda == 'CRC':
        salario = f"₡{row['salario_promedio']:,.0f}"
    else:
        salario = f"${row['salario_promedio']:,.0f}"
    
    # Extraer categoría base (sin nivel si está presente)
    categoria = row['categoria_principal']
    if '(' in categoria:
        categoria_base = categoria.split('(')[0].strip()
    else:
        categoria_base = categoria
    
    # Crear etiqueta descriptiva
    etiqueta = f"{categoria_base[:20]}... | {row['n_empleos']} empleos | {salario}"
    
    return etiqueta

def crear_etiqueta_corta(row):
    """Crear etiqueta corta para gráficos"""
    categoria = row['categoria_principal']
    if '(' in categoria:
        categoria_base = categoria.split('(')[0].strip()
        nivel = categoria.split('(')[1].replace(')', '')
    else:
        categoria_base = categoria[:15]
        nivel = ""
    
    # Extraer palabras clave
    palabras = categoria_base.split()[:3]
    etiqueta_corta = " ".join(palabras)
    
    if nivel:
        etiqueta_corta = f"{etiqueta_corta} ({nivel[:5]})"
    
    return etiqueta_corta[:20]

# ============================================================
# INICIALIZACIÓN
# ============================================================

st.markdown("""
<div class="main-header fade-in">
    <h1>💰 Dashboard de Análisis de Clusters Salariales</h1>
    <p style="opacity: 0.9; margin-bottom: 0;">Clasificación por Categoría Predominante y Nivel Salarial</p>
    <p style="font-size: 14px; opacity: 0.8;">
        Cada cluster identificado por su categoría principal y salario promedio
    </p>
</div>
""", unsafe_allow_html=True)

# Cargar datos
df_usd, df_crc, cluster_usd, cluster_crc = load_dual_data()

if df_usd is None or df_crc is None:
    st.stop()

# ============================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("💰 Moneda de Visualización")
    currency_type = st.radio(
        "Selecciona la moneda:",
        ["Colones Costarricenses (CRC)", "Dólares Estadounidenses (USD)"],
        index=0
    )
    
    st.subheader("💱 Tasa de Cambio")
    exchange_rate = st.slider(
        "Tasa CRC/USD:",
        min_value=500,
        max_value=600,
        value=550,
        step=10
    )
    
    st.markdown("---")
    
    st.subheader("📊 Filtros")
    
    # Determinar dataset activo para filtros
    active_df = df_crc.copy()
    salary_col = 'crc_limpio' if 'crc_limpio' in df_crc.columns else 'salario_limpio'
    
    min_salary, max_salary = st.slider(
        f"Rango salarial (CRC):",
        min_value=int(active_df[salary_col].min()),
        max_value=int(active_df[salary_col].max()),
        value=(int(active_df[salary_col].min()), int(active_df[salary_col].max())),
        step=10000
    )
    
    # Filtro por número de clusters
    n_clusters = st.slider(
        "Máximo clusters a mostrar:",
        min_value=3,
        max_value=min(15, len(cluster_crc)),
        value=min(7, len(cluster_crc)),
        step=1
    )
    
    st.markdown("---")
    
    st.subheader("📈 Métricas Rápidas")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Empleos", len(df_crc))
    with col2:
        st.metric("Clusters", len(cluster_crc))
    
    if currency_type == "Colones Costarricenses (CRC)":
        st.metric("Salario Promedio", f"₡{df_crc[salary_col].mean():,.0f}")
    else:
        st.metric("Salario Promedio", f"${(df_crc[salary_col].mean()/exchange_rate):,.0f}")

# ============================================================
# PREPARACIÓN DE DATOS Y ETIQUETAS
# ============================================================

# Determinar dataset activo
df_active = df_crc.copy()
cluster_active = cluster_crc.copy()
salary_column = 'crc_limpio' if 'crc_limpio' in df_crc.columns else 'salario_limpio'
cluster_column = 'cluster_final' if 'cluster_final' in df_crc.columns else 'cluster_nombre'
currency_symbol = '₡' if currency_type == "Colones Costarricenses (CRC)" else '$'
is_crc_original = True

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

# Crear etiquetas para visualización
cluster_filtered['etiqueta_descriptiva'] = cluster_filtered.apply(
    lambda row: crear_etiqueta_cluster(row, 'CRC' if currency_type == "Colones Costarricenses (CRC)" else 'USD'), 
    axis=1
)

cluster_filtered['etiqueta_corta'] = cluster_filtered.apply(crear_etiqueta_corta, axis=1)

# Convertir a USD si se solicita
if currency_type == "Dólares Estadounidenses (USD)":
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
# SECCIÓN 1: RESUMEN EJECUTIVO CON IDENTIFICADORES DESCRIPTIVOS
# ============================================================

st.header("🎯 Resumen Ejecutivo - Clusters por Categoría Predominante")

# Crear columnas para métricas
col1, col2, col3 = st.columns(3)

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
        value=top_cluster['etiqueta_corta'],
        delta=f"{currency_symbol}{top_cluster['salario_promedio']:,.0f}"
    )

# Mostrar lista de clusters con sus categorías predominantes
st.subheader("📋 Clusters Identificados (por categoría predominante)")

for idx, (cluster_id, row) in enumerate(cluster_filtered.iterrows(), 1):
    st.markdown(f"""
    **{idx}. {row['etiqueta_corta']}**
    - **Categoría principal:** {row['categoria_principal']}
    - **Salario promedio:** {currency_symbol}{row['salario_promedio']:,.0f}
    - **Rango:** {currency_symbol}{row['salario_min']:,.0f} - {currency_symbol}{row['salario_max']:,.0f}
    - **Empleos:** {row['n_empleos']} ({row['n_empleos']/len(df_filtered)*100:.1f}%)
    """)

st.markdown("---")

# ============================================================
# SECCIÓN 2: PIRÁMIDE SALARIAL CON CATEGORÍAS COMO IDENTIFICADORES
# ============================================================

st.header("📊 Pirámide Salarial por Categoría Predominante")

col_pyramid1, col_pyramid2 = st.columns([2, 1])

with col_pyramid1:
    # Preparar datos para la pirámide
    clusters_ordenados = cluster_filtered.sort_values('salario_promedio', ascending=True)
    
    salarios_avg = clusters_ordenados['salario_promedio'].values
    counts = clusters_ordenados['n_empleos'].values
    etiquetas_cortas = clusters_ordenados['etiqueta_corta'].values
    categorias_completas = clusters_ordenados['categoria_principal'].values
    
    # Crear gráfico de barras horizontales
    fig_pyramid = go.Figure()
    
    fig_pyramid.add_trace(go.Bar(
        y=etiquetas_cortas,  # Usar etiquetas cortas en eje Y
        x=salarios_avg,
        orientation='h',
        marker_color=px.colors.sequential.Viridis_r[:len(clusters_ordenados)],
        text=[f"{currency_symbol}{s:,.0f}" for s in salarios_avg],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                     f'Salario: {currency_symbol}%{{x:,.0f}}<br>' +
                     'Empleos: %{customdata[0]}<br>' +
                     'Categoría completa: %{customdata[1]}<extra></extra>',
        customdata=np.column_stack((counts, categorias_completas))
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
    
    fig_pyramid.update_layout(
        height=max(400, len(clusters_ordenados) * 40),
        title="Distribución Salarial por Categoría Predominante",
        xaxis_title=f"Salario Promedio ({currency_symbol})",
        yaxis_title="Categoría Principal",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_pyramid, use_container_width=True)

with col_pyramid2:
    # Gráfico de donut para distribución
    fig_donut = go.Figure()
    
    fig_donut.add_trace(go.Pie(
        labels=etiquetas_cortas,
        values=counts,
        hole=0.4,
        marker_colors=px.colors.sequential.Viridis[:len(clusters_ordenados)],
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>' +
                     'Empleos: %{value}<br>' +
                     'Porcentaje: %{percent}<br>' +
                     f'Salario: {currency_symbol}%{{customdata[0]:,.0f}}<extra></extra>',
        customdata=salarios_avg
    ))
    
    fig_donut.update_layout(
        height=400,
        title="Distribución de Empleos por Categoría",
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
    st.subheader("Resumen por Categoría")
    summary_table = clusters_ordenados.copy()
    summary_table['salario_promedio'] = summary_table['salario_promedio'].apply(
        lambda x: f"{currency_symbol}{x:,.0f}"
    )
    st.dataframe(
        summary_table[['categoria_principal', 'salario_promedio', 'n_empleos']],
        use_container_width=True,
        height=300
    )

st.markdown("---")

# ============================================================
# SECCIÓN 3: ANÁLISIS DE ESTABILIDAD POR CATEGORÍA
# ============================================================

st.header("📈 Análisis de Estabilidad por Categoría")

# Calcular métricas de estabilidad
stability_data = []
for cluster_id, row in cluster_filtered.iterrows():
    cluster_df = df_filtered[df_filtered[cluster_column] == cluster_id] if cluster_column in df_filtered.columns else pd.DataFrame()
    
    if len(cluster_df) > 0:
        salary_range = row['salario_max'] - row['salario_min']
        stability_score = 100 - (salary_range / row['salario_promedio']) * 20 if row['salario_promedio'] > 0 else 0
        
        stability_data.append({
            'categoria': row['etiqueta_corta'],
            'categoria_completa': row['categoria_principal'],
            'salario_promedio': row['salario_promedio'],
            'n_empleos': row['n_empleos'],
            'estabilidad': min(100, max(0, stability_score)),
            'rango_salarial': salary_range
        })

if stability_data:
    stability_df = pd.DataFrame(stability_data)
    
    # Crear gráfico de burbujas
    fig_bubbles = px.scatter(
        stability_df,
        x='salario_promedio',
        y='estabilidad',
        size='n_empleos',
        color='categoria',
        hover_name='categoria',
        hover_data=['categoria_completa', 'rango_salarial'],
        size_max=60,
        color_discrete_sequence=px.colors.qualitative.Set3[:len(stability_df)],
        title="Relación Salario-Estabilidad por Categoría"
    )
    
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
        st.subheader("Categorías más Estables")
        if len(stability_df) > 0:
            stable_cats = stability_df.nlargest(3, 'estabilidad')
            for _, row in stable_cats.iterrows():
                st.metric(
                    label=row['categoria'],
                    value=f"{row['estabilidad']:.1f} puntos",
                    delta=f"{currency_symbol}{row['salario_promedio']:,.0f}"
                )
    
    with col_stab2:
        st.subheader("Categorías Mejor Pagadas")
        if len(stability_df) > 0:
            high_salary_cats = stability_df.nlargest(3, 'salario_promedio')
            for _, row in high_salary_cats.iterrows():
                st.metric(
                    label=row['categoria'],
                    value=f"{currency_symbol}{row['salario_promedio']:,.0f}",
                    delta=f"{row['estabilidad']:.1f} estabilidad"
                )
else:
    st.warning("No hay suficientes datos para calcular la estabilidad.")

st.markdown("---")

# ============================================================
# SECCIÓN 4: DISTRIBUCIÓN DETALLADA POR CATEGORÍA
# ============================================================

st.header("🏷️ Distribución Detallada por Categoría")

# Determinar columna de categoría
categoria_col = 'categoria_refinada_crc' if 'categoria_refinada_crc' in df_filtered.columns else 'Categora_refinada'

if categoria_col in df_filtered.columns:
    # Preparar datos para heatmap
    top_categories = df_filtered[categoria_col].value_counts().head(10).index.tolist()
    heatmap_data = []
    categorias_cluster = []
    
    for cluster_id, row in cluster_filtered.iterrows():
        cluster_df = df_filtered[df_filtered[cluster_column] == cluster_id] if cluster_column in df_filtered.columns else pd.DataFrame()
        
        row_data = []
        for category in top_categories:
            count = len(cluster_df[cluster_df[categoria_col] == category])
            row_data.append(count)
        heatmap_data.append(row_data)
        categorias_cluster.append(row['etiqueta_corta'])
    
    # Crear heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[cat[:20] + '...' if len(cat) > 20 else cat for cat in top_categories],
        y=categorias_cluster,
        colorscale='YlOrRd',
        hoverongaps=False,
        text=heatmap_data,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>Categoría Cluster: %{y}</b><br>' +
                     '<b>Categoría Empleo: %{x}</b><br>' +
                     'Empleos: %{z}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        height=max(500, len(categorias_cluster) * 40),
        title="Concentración de Categorías Específicas por Cluster",
        xaxis_title="Categorías Específicas",
        yaxis_title="Categoría Principal del Cluster",
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Distribución de subcategorías
    st.subheader("Composición Interna de Cada Cluster")
    
    for cluster_id, row in cluster_filtered.iterrows():
        with st.expander(f"🔍 {row['etiqueta_corta']} - {row['n_empleos']} empleos"):
            if cluster_column in df_filtered.columns:
                cluster_df = df_filtered[df_filtered[cluster_column] == cluster_id]
                if len(cluster_df) > 0:
                    subcats = cluster_df[categoria_col].value_counts().head(5)
                    
                    col_sub1, col_sub2 = st.columns(2)
                    with col_sub1:
                        for subcat, count in subcats.items():
                            porcentaje = (count / len(cluster_df)) * 100
                            st.write(f"• **{subcat}:** {count} ({porcentaje:.1f}%)")
                    
                    with col_sub2:
                        # Estadísticas del cluster
                        st.write(f"**Salario promedio:** {currency_symbol}{cluster_df[salary_column].mean():,.0f}")
                        st.write(f"**Rango:** {currency_symbol}{cluster_df[salary_column].min():,.0f} - {currency_symbol}{cluster_df[salary_column].max():,.0f}")
                        st.write(f"**Desviación:** {currency_symbol}{cluster_df[salary_column].std():,.0f}")
else:
    st.warning(f"No se encontró la columna de categorías detalladas.")

st.markdown("---")

# ============================================================
# SECCIÓN 5: PERFIL DETALLADO POR CATEGORÍA
# ============================================================

st.header("🔍 Perfil Detallado por Categoría")

# Crear selector de categorías/clusters
categoria_options = {row['etiqueta_corta']: cluster_id for cluster_id, row in cluster_filtered.iterrows()}

if categoria_options:
    selected_categoria_nombre = st.selectbox(
        "Selecciona una categoría para análisis detallado:",
        options=list(categoria_options.keys())
    )
    
    selected_cluster_id = categoria_options[selected_categoria_nombre]
    cluster_data = cluster_filtered.loc[selected_cluster_id]
    
    if cluster_column in df_filtered.columns:
        cluster_jobs = df_filtered[df_filtered[cluster_column] == selected_cluster_id]
    else:
        cluster_jobs = pd.DataFrame()
    
    # Mostrar encabezado del perfil
    st.markdown(f"""
    ### 📋 Perfil: {cluster_data['etiqueta_corta']}
    **Categoría completa:** {cluster_data['categoria_principal']}
    """)
    
    # Métricas principales
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    
    with col_metrics1:
        st.metric(
            "Salario Promedio",
            f"{currency_symbol}{cluster_data['salario_promedio']:,.0f}",
            f"Rango: {currency_symbol}{cluster_data['salario_min']:,.0f} - {currency_symbol}{cluster_data['salario_max']:,.0f}"
        )
    
    with col_metrics2:
        st.metric("Total Empleos", cluster_data['n_empleos'])
    
    with col_metrics3:
        salary_range = cluster_data['salario_max'] - cluster_data['salario_min']
        stability = 100 - (salary_range / cluster_data['salario_promedio']) * 20 if cluster_data['salario_promedio'] > 0 else 0
        st.metric("Estabilidad", f"{stability:.1f}%")
    
    with col_metrics4:
        if len(cluster_jobs) > 0:
            unique_cats = cluster_jobs[categoria_col].nunique() if categoria_col in cluster_jobs.columns else 1
            diversity = (unique_cats / len(cluster_jobs)) * 100
            st.metric("Diversidad", f"{diversity:.1f}%")
    
    # Mostrar empleos del cluster
    if len(cluster_jobs) > 0:
        st.subheader("📝 Empleos Representativos")
        
        # Mostrar primeros 10 empleos
        display_jobs = cluster_jobs.copy()
        
        # Identificar columnas disponibles
        available_cols = []
        for col in ['Título', 'Empresa', 'Categora', categoria_col, 'Descripción']:
            if col in display_jobs.columns:
                available_cols.append(col)
        
        # Asegurar que tenemos la columna de salario
        display_jobs['Salario'] = display_jobs[salary_column].apply(lambda x: f"{currency_symbol}{x:,.0f}")
        
        st.dataframe(
            display_jobs[available_cols + ['Salario']].head(10),
            use_container_width=True,
            hide_index=True
        )
        
        # Análisis adicional
        if len(cluster_jobs) > 5:
            st.subheader("📊 Análisis Adicional")
            
            col_anal1, col_anal2 = st.columns(2)
            
            with col_anal1:
                st.write("**Distribución de Subcategorías:**")
                if categoria_col in cluster_jobs.columns:
                    subcat_counts = cluster_jobs[categoria_col].value_counts().head(5)
                    for subcat, count in subcat_counts.items():
                        percentage = (count / len(cluster_jobs)) * 100
                        st.write(f"• {subcat}: {count} ({percentage:.1f}%)")
            
            with col_anal2:
                st.write("**Estadísticas Salariales Detalladas:**")
                st.write(f"• **Mediana:** {currency_symbol}{cluster_jobs[salary_column].median():,.0f}")
                st.write(f"• **Moda:** {currency_symbol}{cluster_jobs[salary_column].mode().iloc[0] if not cluster_jobs[salary_column].mode().empty else 'N/A':,.0f}")
                st.write(f"• **Percentil 25:** {currency_symbol}{cluster_jobs[salary_column].quantile(0.25):,.0f}")
                st.write(f"• **Percentil 75:** {currency_symbol}{cluster_jobs[salary_column].quantile(0.75):,.0f}")
                st.write(f"• **Coef. Variación:** {(cluster_jobs[salary_column].std() / cluster_jobs[salary_column].mean() * 100):.1f}%")

st.markdown("---")

# ============================================================
# SECCIÓN 6: RECOMENDACIONES POR CATEGORÍA
# ============================================================

st.header("🎯 Recomendaciones Estratégicas por Categoría")

# Analizar clusters para recomendaciones
if len(cluster_filtered) >= 3:
    # Identificar clusters por nivel
    top_clusters = cluster_filtered.head(2)
    mid_clusters = cluster_filtered.iloc[2:-2] if len(cluster_filtered) > 4 else cluster_filtered.iloc[1:-1]
    entry_clusters = cluster_filtered.tail(2)
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.subheader("💼 Para Profesionales con Experiencia")
        for _, row in top_clusters.iterrows():
            st.markdown(f"""
            **{row['etiqueta_corta']}**
            • **Objetivo salarial:** {currency_symbol}{row['salario_promedio']*1.1:,.0f}+
            • **Recomendación:** Posiciones de liderazgo o especialización avanzada
            • **Oportunidades:** {row['n_empleos']} empleos disponibles
            """)
    
    with col_rec2:
        st.subheader("📈 Para Desarrollo Profesional")
        if len(mid_clusters) > 0:
            for _, row in mid_clusters.iterrows():
                st.markdown(f"""
                **{row['etiqueta_corta']}**
                • **Rango objetivo:** {currency_symbol}{row['salario_promedio']*0.9:,.0f} - {currency_symbol}{row['salario_promedio']*1.1:,.0f}
                • **Recomendación:** Desarrollo de habilidades específicas
                • **Crecimiento potencial:** +{(row['salario_promedio']/entry_clusters['salario_promedio'].mean()*100-100):.0f}%
                """)

# Trayectorias profesionales identificadas
st.subheader("🛤️ Trayectorias Profesionales Identificadas")

# Analizar posibles progresiones
if len(cluster_filtered) > 1:
    # Buscar categorías relacionadas
    categorias_por_nivel = {}
    for _, row in cluster_filtered.iterrows():
        categoria_base = row['categoria_principal'].split('(')[0].strip() if '(' in row['categoria_principal'] else row['categoria_principal']
        nivel = row['categoria_principal'].split('(')[1].replace(')', '') if '(' in row['categoria_principal'] else 'General'
        
        if categoria_base not in categorias_por_nivel:
            categorias_por_nivel[categoria_base] = []
        categorias_por_nivel[categoria_base].append((nivel, row['salario_promedio'], row['etiqueta_corta']))
    
    # Mostrar trayectorias identificadas
    for categoria_base, niveles in categorias_por_nivel.items():
        if len(niveles) > 1:
            niveles_ordenados = sorted(niveles, key=lambda x: x[1])
            st.markdown(f"""
            **Trayectoria: {categoria_base}**
            """)
            
            for i, (nivel, salario, etiqueta) in enumerate(niveles_ordenados):
                st.write(f"{i+1}. **{etiqueta}** - {currency_symbol}{salario:,.0f}")
            
            crecimiento = ((niveles_ordenado[-1][1] - niveles_ordenado[0][1]) / niveles_ordenado[0][1]) * 100
            st.write(f"**Crecimiento total:** +{crecimiento:.0f}%")

st.markdown("---")

# ============================================================
# SECCIÓN 7: EXPORTACIÓN DE RESULTADOS
# ============================================================

st.header("📤 Exportación de Resultados")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Exportar datos filtrados
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Datos Filtrados",
        data=csv_data,
        file_name="datos_filtrados_categorias.csv",
        mime="text/csv",
        help="Exporta los datos actualmente filtrados"
    )

with col_exp2:
    # Exportar resumen de clusters
    summary_export = cluster_filtered.copy()
    summary_export.index = summary_export['etiqueta_corta']
    csv_summary = summary_export[['categoria_principal', 'salario_promedio', 'salario_min', 'salario_max', 'n_empleos']].to_csv(index=True).encode('utf-8')
    st.download_button(
        label="📊 Descargar Resumen por Categoría",
        data=csv_summary,
        file_name="resumen_categorias.csv",
        mime="text/csv",
        help="Exporta el resumen por categoría predominante"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(f"""
**📋 Información del Análisis:**

**Moneda:** {currency_type}
**Empleos analizados:** {len(df_filtered)} de {len(df_active)} totales
**Categorías identificadas:** {len(cluster_filtered)}
**Rango salarial:** {currency_symbol}{df_filtered[salary_column].min():,.0f} - {currency_symbol}{df_filtered[salary_column].max():,.0f}

**🎯 Identificadores de Clusters:**
- Cada cluster identificado por su categoría predominante
- Etiqueta corta para visualización: primeras palabras de la categoría
- Información completa disponible en tooltips y tablas
- Los clusters están ordenados por salario promedio descendente

**⚠️ Consideraciones:**
1. Los clusters se definen por similitud semántica y salario
2. La categoría predominante es la más frecuente en cada cluster
3. Los salarios en USD son conversiones aproximadas
4. Validar con datos adicionales para decisiones críticas
""")
