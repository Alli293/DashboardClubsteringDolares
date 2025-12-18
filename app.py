# ============================================================ 
# DASHBOARD DE ANÁLISIS DE CLUSTERS SALARIALES - SOLO USD
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
    page_title="Dashboard de Análisis de Clusters Salariales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS GLOBAL
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
    
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS
# ============================================================

@st.cache_data
def load_data():
    """Cargar datos de análisis en USD"""
    try:
        df = pd.read_csv('empleos_analisis_final.csv')
        
        # Renombrar columna de cluster si es necesario
        if 'cluster_nombre' in df.columns:
            # Usar Categora como nombre del cluster
            # Encontrar la categoría dominante por cluster
            cluster_categories = {}
            for cluster in df['cluster_nombre'].unique():
                cluster_data = df[df['cluster_nombre'] == cluster]
                if 'Categora' in cluster_data.columns:
                    dominant_category = cluster_data['Categora'].value_counts().index[0]
                    cluster_categories[cluster] = dominant_category
            
            # Renombrar clusters con su categoría dominante
            # Asegurarnos de que no hay duplicados
            category_counts = {}
            final_cluster_names = {}
            
            for cluster, category in cluster_categories.items():
                if category not in category_counts:
                    final_cluster_names[cluster] = category
                    category_counts[category] = 1
                else:
                    # Si hay duplicados, agregar número
                    final_cluster_names[cluster] = f"{category} {category_counts[category]}"
                    category_counts[category] += 1
            
            df['cluster_nombre'] = df['cluster_nombre'].map(final_cluster_names)
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# ============================================================
# INICIALIZACIÓN
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>Dashboard de Análisis de Clusters Salariales</h1>
    <p style="opacity: 0.9; margin-bottom: 0;">Análisis de empleos por categoría y salario en USD</p>
</div>
""", unsafe_allow_html=True)

# Cargar datos
df = load_data()

if df is None:
    st.stop()

# ============================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================

with st.sidebar:
    st.header("Configuración del Dashboard")
    
    # Filtro por rango salarial
    st.write("**Filtro Salarial:**")
    default_min = int(df['salario_limpio'].min())
    default_max = int(df['salario_limpio'].max())
    
    min_salary, max_salary = st.slider(
        "Rango salarial (USD):",
        min_value=default_min,
        max_value=default_max,
        value=(default_min, default_max),
        step=100
    )
    
    # Filtro por categoría si existe
    if 'Categora' in df.columns:
        categorias = sorted(df['Categora'].unique())
        selected_categorias = st.multiselect(
            "Filtrar por categoría:",
            options=categorias,
            default=categorias[:min(5, len(categorias))]
        )
    
    st.markdown("---")
    
    # Métricas rápidas en sidebar
    st.subheader("Resumen Rápido")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Empleos", len(df))
    with col2:
        if 'cluster_nombre' in df.columns:
            st.metric("Clusters", df['cluster_nombre'].nunique())
    
    st.metric("Salario Promedio", f"${df['salario_limpio'].mean():,.0f}")
    st.metric("Salario Máximo", f"${df['salario_limpio'].max():,.0f}")

# ============================================================
# FILTRADO DE DATOS
# ============================================================

# Aplicar filtros
df_filtered = df[
    (df['salario_limpio'] >= min_salary) & 
    (df['salario_limpio'] <= max_salary)
]

if 'Categora' in df.columns and 'selected_categorias' in locals() and selected_categorias:
    df_filtered = df_filtered[df_filtered['Categora'].isin(selected_categorias)]

# ============================================================
# SECCIÓN 1: RESUMEN EJECUTIVO
# ============================================================

st.header("Resumen Ejecutivo")

# Crear columnas para métricas
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_salary = df_filtered['salario_limpio'].mean()
    st.metric(
        label="Salario Promedio",
        value=f"${avg_salary:,.0f}"
    )

with col2:
    st.metric(
        label="Empleos Analizados",
        value=f"{len(df_filtered)}",
        delta=f"{len(df_filtered)/len(df)*100:.1f}% del total"
    )

with col3:
    if 'cluster_nombre' in df_filtered.columns:
        cluster_avg = df_filtered.groupby('cluster_nombre')['salario_limpio'].mean()
        top_cluster = cluster_avg.idxmax()
        top_salary = cluster_avg.max()
        st.metric(
            label="Cluster Mejor Pagado",
            value=top_cluster,
            delta=f"${top_salary:,.0f}"
        )
    else:
        st.metric("Categorías", df_filtered['Categora'].nunique())

with col4:
    st.metric(
        label="Rango Salarial",
        value=f"${min_salary:,.0f}-${max_salary:,.0f}"
    )

st.markdown("---")

# ============================================================
# SECCIÓN 2: DISTRIBUCIÓN SALARIAL POR CATEGORÍA
# ============================================================

if 'Categora' in df_filtered.columns:
    st.header("Salario Promedio por Categoría")
    st.write("Distribución de salarios promedio organizada por categorías laborales.")
    
    # Calcular salario promedio por categoría
    category_salary = df_filtered.groupby('Categora')['salario_limpio'].agg(['mean', 'count']).round(0)
    category_salary = category_salary.sort_values('mean', ascending=False)
    
    # Crear gráfico de barras
    fig_category = go.Figure()
    
    fig_category.add_trace(go.Bar(
        x=category_salary.index,
        y=category_salary['mean'],
        text=[f"${x:,.0f}" for x in category_salary['mean']],
        textposition='outside',
        marker_color=px.colors.sequential.Viridis[:len(category_salary)],
        hovertemplate='<b>%{x}</b><br>' +
                     'Salario promedio: $%{y:,.0f}<br>' +
                     'Empleos: %{customdata}<extra></extra>',
        customdata=category_salary['count']
    ))
    
    fig_category.update_layout(
        height=500,
        title="Salario Mensual Promedio por Categoría Laboral",
        xaxis_title="Categoría",
        yaxis_title="Salario Promedio Mensual (USD)",
        showlegend=False,
        xaxis_tickangle=-45,
        margin=dict(b=100)
    )
    
    st.plotly_chart(fig_category, use_container_width=True)

st.markdown("---")

# ============================================================
# SECCIÓN 3: DISTRIBUCIÓN SALARIAL POR CLUSTER
# ============================================================

if 'cluster_nombre' in df_filtered.columns:
    st.header("Salario Promedio por Cluster")
    st.write("Análisis de clusters salariales identificados por algoritmo de agrupamiento.")
    
    # Calcular estadísticas por cluster
    cluster_stats = df_filtered.groupby('cluster_nombre').agg({
        'salario_limpio': ['mean', 'min', 'max', 'count'],
        'Categora': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
    }).round(0)
    
    cluster_stats.columns = ['salario_promedio', 'salario_min', 'salario_max', 'n_empleos', 'categoria_principal']
    cluster_stats = cluster_stats.sort_values('salario_promedio', ascending=True)
    
    # Crear gráfico de barras horizontales
    fig_cluster = go.Figure()
    
    fig_cluster.add_trace(go.Bar(
        y=cluster_stats.index.tolist(),
        x=cluster_stats['salario_promedio'],
        orientation='h',
        marker_color=px.colors.sequential.Plasma[:len(cluster_stats)],
        text=[f"${x:,.0f}" for x in cluster_stats['salario_promedio']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                     'Salario promedio: $%{x:,.0f}<br>' +
                     'Empleos: %{customdata[0]}<br>' +
                     'Categoría: %{customdata[1]}<extra></extra>',
        customdata=np.column_stack((cluster_stats['n_empleos'], cluster_stats['categoria_principal']))
    ))
    
    # Añadir línea de promedio general
    promedio_general = df_filtered['salario_limpio'].mean()
    fig_cluster.add_vline(
        x=promedio_general,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio General: ${promedio_general:,.0f}",
        annotation_position="top right"
    )
    
    fig_cluster.update_layout(
        height=500,
        title="Distribución Salarial por Cluster",
        xaxis_title="Salario Promedio Mensual (USD)",
        yaxis_title="Cluster",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_cluster, use_container_width=True)

st.markdown("---")

# ============================================================
# SECCIÓN 4: TRAYECTORIAS PROFESIONALES
# ============================================================

st.header("Trayectorias Profesionales y Evolución Salarial")
st.write("Simulación de progresión salarial en diferentes trayectorias profesionales.")

# Definir trayectorias
trayectorias = {
    'Asistente → Director (Ciencias)': {
        'etapas': ['Asistente', 'Coordinador', 'Gerente', 'Director'],
        'salarios': [966, 3434, 6727, 10480],
        'color': '#2E86AB'
    },
    'Desarrollador Senior → Líder': {
        'etapas': ['Senior Dev', 'Tech Lead'],
        'salarios': [6955, 11127],
        'color': '#A23B72'
    },
    'Trayectoria Técnica': {
        'etapas': ['sin titulacion', 'nivel tecnico', 'tecnico senior'],
        'salarios': [432, 2727, 6364],
        'color': '#F18F01'
    }
}

# Crear gráfico de líneas
fig_trayectorias = go.Figure()

# Añadir cada trayectoria
for nombre, datos in trayectorias.items():
    fig_trayectorias.add_trace(go.Scatter(
        x=datos['etapas'],
        y=datos['salarios'],
        mode='lines+markers+text',
        name=nombre,
        line=dict(color=datos['color'], width=3),
        marker=dict(size=12),
        text=[f"${s:,.0f}" for s in datos['salarios']],
        textposition="top center",
        hoverinfo='text+name',
        hovertext=[f"{etapa}<br>Salario: ${salario:,.0f}" 
                  for etapa, salario in zip(datos['etapas'], datos['salarios'])]
    ))

# Configurar layout
fig_trayectorias.update_layout(
    height=500,
    title="Evolución Salarial en Trayectorias Profesionales",
    xaxis_title="Etapa Profesional",
    yaxis_title="Salario Mensual (USD)",
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
        if len(datos['salarios']) > 1:
            crecimiento = ((datos['salarios'][-1] - datos['salarios'][0]) / datos['salarios'][0]) * 100
            st.metric(
                label=nombre,
                value=f"+{crecimiento:.0f}%",
                delta=f"De ${datos['salarios'][0]:,.0f} a ${datos['salarios'][-1]:,.0f}"
            )

st.markdown("---")

# ============================================================
# SECCIÓN 5: DISTRIBUCIÓN DE SALARIOS
# ============================================================

st.header("Distribución de Salarios")
st.write("Histograma que muestra la frecuencia de diferentes rangos salariales.")

# Crear histograma
fig_hist = px.histogram(
    df_filtered,
    x='salario_limpio',
    nbins=30,
    title="Distribución de Salarios Mensuales",
    labels={'salario_limpio': 'Salario Mensual (USD)', 'count': 'Número de Empleos'},
    color_discrete_sequence=['#26d0ce']
)

# Añadir líneas de referencia
fig_hist.add_vline(
    x=df_filtered['salario_limpio'].mean(),
    line_dash="dash",
    line_color="red",
    annotation_text=f"Promedio: ${df_filtered['salario_limpio'].mean():,.0f}",
    annotation_position="top right"
)

fig_hist.add_vline(
    x=df_filtered['salario_limpio'].median(),
    line_dash="dash",
    line_color="green",
    annotation_text=f"Mediana: ${df_filtered['salario_limpio'].median():,.0f}",
    annotation_position="top left"
)

fig_hist.update_layout(
    height=500,
    showlegend=False,
    bargap=0.1
)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ============================================================
# SECCIÓN 6: ANÁLISIS DETALLADO POR CLUSTER
# ============================================================

if 'cluster_nombre' in df_filtered.columns:
    st.header("Análisis Detallado por Cluster")
    
    # Seleccionar cluster para análisis detallado
    clusters = df_filtered['cluster_nombre'].unique()
    selected_cluster = st.selectbox(
        "Selecciona un cluster para análisis detallado:",
        options=clusters
    )
    
    if selected_cluster:
        # Obtener datos del cluster seleccionado
        cluster_data = df_filtered[df_filtered['cluster_nombre'] == selected_cluster]
        
        # Mostrar métricas del cluster
        col_detail1, col_detail2, col_detail3 = st.columns(3)
        
        with col_detail1:
            avg_salary = cluster_data['salario_limpio'].mean()
            st.metric(
                "Salario Promedio",
                f"${avg_salary:,.0f}",
                f"Rango: ${cluster_data['salario_limpio'].min():,.0f}-${cluster_data['salario_limpio'].max():,.0f}"
            )
        
        with col_detail2:
            st.metric("Total Empleos", len(cluster_data))
        
        with col_detail3:
            if 'Categora' in cluster_data.columns:
                unique_cats = cluster_data['Categora'].nunique()
                st.metric("Categorías Únicas", unique_cats)
        
        # Mostrar distribución de categorías en el cluster
        if 'Categora' in cluster_data.columns:
            st.subheader("Distribución de Categorías en este Cluster")
            
            cat_dist = cluster_data['Categora'].value_counts().reset_index()
            cat_dist.columns = ['Categoría', 'Cantidad']
            
            fig_cat_dist = px.pie(
                cat_dist,
                values='Cantidad',
                names='Categoría',
                title=f"Categorías en Cluster {selected_cluster}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_cat_dist.update_layout(height=400)
            st.plotly_chart(fig_cat_dist, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(f"""
**Información del Análisis:**

**Empleos analizados:** {len(df_filtered)} de {len(df)} totales
**Rango salarial filtrado:** ${min_salary:,.0f} - ${max_salary:,.0f}
**Salario promedio filtrado:** ${df_filtered['salario_limpio'].mean():,.0f}

**Nota:** Este análisis está basado en datos de muestra y debe ser validado con información adicional.
""")
