
# ============================================================
# DASHBOARD DE ANÁLISIS DE CLUSTERS SALARIALES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard - Análisis de Clusters Salariales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS
# ============================================================

@st.cache_data
def load_data():
    """Cargar y preparar los datos"""
    try:
        df = pd.read_csv('empleos_analisis_final.csv')
        
        # Preparar datos para visualizaciones
        cluster_summary = df.groupby('cluster_nombre').agg({
            'salario_limpio': ['mean', 'min', 'max', 'count'],
            'Categora_refinada': lambda x: x.value_counts().index[0]
        }).round(0)
        
        cluster_summary.columns = ['salario_promedio', 'salario_min', 'salario_max', 'n_empleos', 'categoria_principal']
        cluster_summary = cluster_summary.sort_values('salario_promedio', ascending=False)
        
        return df, cluster_summary
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None, None

# ============================================================
# INICIALIZACIÓN
# ============================================================

# Título principal
st.title("📊 Dashboard de Análisis de Clusters Salariales")
st.markdown("---")

# Cargar datos
df, cluster_summary = load_data()

if df is None:
    st.error("No se pudieron cargar los datos. Verifica que el archivo 'empleos_analisis_final.csv' exista.")
    st.stop()

# ============================================================
# SIDEBAR - FILTROS Y CONFIGURACIÓN
# ============================================================

with st.sidebar:
    st.header("Configuración del Análisis")
    
    # Filtro por rango salarial
    st.subheader("Filtros Salariales")
    min_salary = st.slider(
        "Salario mínimo (USD)",
        min_value=int(df['salario_limpio'].min()),
        max_value=int(df['salario_limpio'].max()),
        value=int(df['salario_limpio'].min()),
        step=100
    )
    
    max_salary = st.slider(
        "Salario máximo (USD)",
        min_value=int(df['salario_limpio'].min()),
        max_value=int(df['salario_limpio'].max()),
        value=int(df['salario_limpio'].max()),
        step=100
    )
    
    # Filtro por clusters
    st.subheader("Selección de Clusters")
    all_clusters = cluster_summary.index.tolist()
    selected_clusters = st.multiselect(
        "Clusters a visualizar:",
        options=all_clusters,
        default=all_clusters[:min(5, len(all_clusters))]
    )
    
    # Opciones de visualización
    st.subheader("Opciones de Visualización")
    show_detailed_labels = st.checkbox("Mostrar etiquetas detalladas", value=True)
    color_scheme = st.selectbox(
        "Esquema de colores:",
        options=["viridis", "plasma", "inferno", "magma", "cividis", "Set3"]
    )
    
    st.markdown("---")
    
    # Métricas rápidas en sidebar
    st.subheader("Métricas Clave")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Empleos", len(df))
    with col2:
        st.metric("Clusters", len(cluster_summary))
    
    st.metric("Salario Promedio", f"${df['salario_limpio'].mean():,.0f}")
    st.metric("Salario Máximo", f"${df['salario_limpio'].max():,.0f}")

# ============================================================
# FILTRAR DATOS SEGÚN SELECCIONES
# ============================================================

df_filtered = df[
    (df['salario_limpio'] >= min_salary) & 
    (df['salario_limpio'] <= max_salary) &
    (df['cluster_nombre'].isin(selected_clusters))
]

cluster_summary_filtered = cluster_summary[cluster_summary.index.isin(selected_clusters)]

# ============================================================
# SECCIÓN 1: VISTA GENERAL - MÉTRICAS PRINCIPALES
# ============================================================

st.header("Resumen Ejecutivo")

# Crear columnas para métricas
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_salary = df_filtered['salario_limpio'].mean()
    st.metric(
        label="Salario Promedio",
        value=f"${avg_salary:,.0f}",
        delta=f"${avg_salary - df['salario_limpio'].mean():,.0f}" if len(selected_clusters) < len(all_clusters) else None
    )

with col2:
    st.metric("Empleos Analizados", len(df_filtered))

with col3:
    top_cluster = cluster_summary_filtered.iloc[0]
    st.metric(
        label="Cluster Mejor Pagado",
        value=top_cluster.name.replace('Cluster_', ''),
        delta=f"${top_cluster['salario_promedio']:,.0f}"
    )

with col4:
    best_value_cluster = cluster_summary_filtered.iloc[1] if len(cluster_summary_filtered) > 1 else cluster_summary_filtered.iloc[0]
    st.metric(
        label="Mejor Valor",
        value=best_value_cluster.name.replace('Cluster_', ''),
        delta=f"{best_value_cluster['n_empleos']} empleos"
    )

st.markdown("---")

# ============================================================
# SECCIÓN 2: PIRÁMIDE SALARIAL INTERACTIVA
# ============================================================

st.header("Pirámide Salarial por Cluster")

col1, col2 = st.columns([2, 1])

with col1:
    # Preparar datos para la pirámide
    clusters_ordered = cluster_summary_filtered.sort_values('salario_promedio', ascending=True).index
    salarios_avg = cluster_summary_filtered.loc[clusters_ordered]['salario_promedio'].values
    counts = cluster_summary_filtered.loc[clusters_ordered]['n_empleos'].values
    categorias = cluster_summary_filtered.loc[clusters_ordered]['categoria_principal'].values
    
    # Crear gráfico de barras horizontales con Plotly
    fig_pyramid = go.Figure()
    
    # Añadir barras
    fig_pyramid.add_trace(go.Bar(
        y=[c.replace('Cluster_', '') for c in clusters_ordered],
        x=salarios_avg,
        orientation='h',
        marker_color=px.colors.sequential.Viridis_r[:len(clusters_ordered)],
        text=[f"${s:,.0f}" for s in salarios_avg],
        textposition='outside',
        name='Salario Promedio',
        hovertemplate='<b>%{y}</b><br>' +
                     'Salario: $%{x:,.0f}<br>' +
                     'Empleos: %{customdata[0]}<br>' +
                     'Categoría: %{customdata[1]}<extra></extra>',
        customdata=np.column_stack((counts, categorias))
    ))
    
    # Añadir línea de promedio general
    promedio_general = df_filtered['salario_limpio'].mean()
    fig_pyramid.add_vline(
        x=promedio_general,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio General: ${promedio_general:,.0f}",
        annotation_position="top right"
    )
    
    # Configurar layout
    fig_pyramid.update_layout(
        height=500,
        title="Distribución Salarial por Cluster",
        xaxis_title="Salario Promedio Mensual (USD)",
        yaxis_title="Cluster",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_pyramid, use_container_width=True)

with col2:
    # Gráfico de donut para distribución
    fig_donut = go.Figure()
    
    fig_donut.add_trace(go.Pie(
        labels=[c.replace('Cluster_', '') for c in cluster_summary_filtered.index],
        values=cluster_summary_filtered['n_empleos'],
        hole=0.4,
        marker_colors=px.colors.sequential.Viridis[:len(cluster_summary_filtered)],
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>' +
                     'Empleos: %{value}<br>' +
                     'Porcentaje: %{percent}<extra></extra>'
    ))
    
    fig_donut.update_layout(
        height=400,
        title="Distribución de Empleos",
        showlegend=False,
        margin=dict(t=40, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig_donut, use_container_width=True)
    
    # Mostrar tabla resumen
    st.subheader("Resumen por Cluster")
    summary_table = cluster_summary_filtered.copy()
    summary_table.index = [c.replace('Cluster_', '') for c in summary_table.index]
    summary_table['salario_promedio'] = summary_table['salario_promedio'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(
        summary_table[['salario_promedio', 'n_empleos', 'categoria_principal']],
        use_container_width=True
    )

st.markdown("---")

# ============================================================
# SECCIÓN 3: ANÁLISIS DE BURBUJAS - SALARIO VS ESTABILIDAD
# ============================================================

st.header("Análisis de Salario vs Estabilidad")

# Calcular métricas de estabilidad
stability_data = []
for cluster, row in cluster_summary_filtered.iterrows():
    cluster_df = df_filtered[df_filtered['cluster_nombre'] == cluster]
    
    # Calcular estabilidad (menor rango = mayor estabilidad)
    salary_range = row['salario_max'] - row['salario_min']
    stability_score = 100 - (salary_range / row['salario_promedio']) * 20 if row['salario_promedio'] > 0 else 0
    
    stability_data.append({
        'cluster': cluster.replace('Cluster_', ''),
        'salario_promedio': row['salario_promedio'],
        'n_empleos': row['n_empleos'],
        'estabilidad': min(100, max(0, stability_score)),
        'rango_salarial': salary_range,
        'categoria': row['categoria_principal']
    })

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
    title="Relación Salario-Estabilidad por Cluster"
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
    xaxis_title="Salario Promedio (USD)",
    yaxis_title="Índice de Estabilidad (0-100)",
    hovermode='closest',
    showlegend=True
)

# Añadir etiquetas mejoradas
fig_bubbles.update_traces(
    textposition='top center',
    marker=dict(line=dict(width=2, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)

st.plotly_chart(fig_bubbles, use_container_width=True)

# Análisis de estabilidad
col1, col2 = st.columns(2)

with col1:
    st.subheader("Clusters más Estables")
    stable_clusters = stability_df.nlargest(3, 'estabilidad')
    for _, row in stable_clusters.iterrows():
        st.metric(
            label=row['cluster'],
            value=f"{row['estabilidad']:.1f} puntos",
            delta=f"${row['salario_promedio']:,.0f}"
        )

with col2:
    st.subheader("Clusters con Mayor Salario")
    high_salary_clusters = stability_df.nlargest(3, 'salario_promedio')
    for _, row in high_salary_clusters.iterrows():
        st.metric(
            label=row['cluster'],
            value=f"${row['salario_promedio']:,.0f}",
            delta=f"{row['estabilidad']:.1f} estabilidad"
        )

st.markdown("---")

# ============================================================
# SECCIÓN 4: TRAYECTORIAS PROFESIONALES
# ============================================================

st.header("Trayectorias Profesionales y Evolución Salarial")

# Definir trayectorias basadas en clusters
trayectorias = {
    'Técnico a Ejecutivo': {
        'etapas': ['Entry-Level', 'Mid-Level', 'Executive'],
        'salarios': [1417, 5512, 10949],
        'clusters': ['Cluster_6/Cluster_7', 'Cluster_4/Cluster_5', 'Cluster_1'],
        'color': '#2E86AB'
    },
    'Desarrollador a Líder Técnico': {
        'etapas': ['Junior Dev', 'Senior Dev', 'Tech Lead'],
        'salarios': [1853, 5610, 10949],
        'clusters': ['Cluster_6', 'Cluster_4', 'Cluster_1'],
        'color': '#A23B72'
    },
    'Especialista a Gerente': {
        'etapas': ['Especialista', 'Senior Spec', 'Manager'],
        'salarios': [1069, 6854, 10949],
        'clusters': ['Cluster_7', 'Cluster_2/Cluster_3', 'Cluster_1'],
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
        hovertext=[f"{etapa}<br>Salario: ${salario:,.0f}<br>Cluster: {cluster}" 
                  for etapa, salario, cluster in zip(datos['etapas'], datos['salarios'], datos['clusters'])]
    ))

# Añadir áreas sombreadas para niveles
fig_trayectorias.add_hrect(
    y0=0, y1=3000,
    fillcolor="green", opacity=0.1,
    layer="below", line_width=0,
    annotation_text="Entry Level"
)

fig_trayectorias.add_hrect(
    y0=3000, y1=7000,
    fillcolor="orange", opacity=0.1,
    layer="below", line_width=0,
    annotation_text="Mid Level"
)

fig_trayectorias.add_hrect(
    y0=7000, y1=12000,
    fillcolor="red", opacity=0.1,
    layer="below", line_width=0,
    annotation_text="Senior/Executive"
)

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
cols = st.columns(len(trayectorias))

for idx, (nombre, datos) in enumerate(trayectorias.items()):
    with cols[idx]:
        crecimiento = ((datos['salarios'][-1] - datos['salarios'][0]) / datos['salarios'][0]) * 100
        st.metric(
            label=nombre,
            value=f"+{crecimiento:.0f}%",
            delta=f"De ${datos['salarios'][0]:,.0f} a ${datos['salarios'][-1]:,.0f}"
        )

st.markdown("---")

# ============================================================
# SECCIÓN 5: DISTRIBUCIÓN DE CATEGORÍAS
# ============================================================

st.header("Distribución de Categorías por Cluster")

# Preparar datos para heatmap
top_categories = df_filtered['Categora_refinada'].value_counts().head(10).index.tolist()
heatmap_data = []

for cluster in cluster_summary_filtered.index:
    cluster_df = df_filtered[df_filtered['cluster_nombre'] == cluster]
    row = []
    for category in top_categories:
        count = len(cluster_df[cluster_df['Categora_refinada'] == category])
        row.append(count)
    heatmap_data.append(row)

# Crear heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=top_categories,
    y=[c.replace('Cluster_', '') for c in cluster_summary_filtered.index],
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
for cluster, row in cluster_summary_filtered.iterrows():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"**{cluster.replace('Cluster_', '')}**")
    with col2:
        st.write(f"{row['categoria_principal']} (${row['salario_promedio']:,.0f} promedio)")

st.markdown("---")

# ============================================================
# SECCIÓN 6: PERFIL DE CLUSTERS TOP
# ============================================================

# ============================================================
# SECCIÓN 6: PERFIL DE CLUSTERS - SOLO MÉTRICAS
# ============================================================

st.header("Perfil Comparativo de Clusters")

# Seleccionar número de clusters para comparar
top_n = st.slider("Número de clusters a comparar:", 2, len(cluster_summary_filtered), min(3, len(cluster_summary_filtered)))
top_clusters = cluster_summary_filtered.head(top_n).index.tolist()

st.subheader("Métricas Clave por Cluster")

# Calcular métricas avanzadas para cada cluster
cluster_metrics = []

for cluster in top_clusters:
    cluster_df = df_filtered[df_filtered['cluster_nombre'] == cluster]
    row = cluster_summary_filtered.loc[cluster]
    
    # Calcular diversidad de categorías
    categorias_unicas = cluster_df['Categora_refinada'].nunique()
    diversidad = categorias_unicas / row['n_empleos'] if row['n_empleos'] > 0 else 0
    
    # Calcular rango salarial relativo
    rango_salarial = (row['salario_max'] - row['salario_min']) / row['salario_promedio'] if row['salario_promedio'] > 0 else 0
    
    # Calcular estabilidad (inversa del rango)
    estabilidad = max(0, 100 - (rango_salarial * 30))
    
    cluster_metrics.append({
        'Cluster': cluster.replace('Cluster_', ''),
        'Salario Promedio': row['salario_promedio'],
        'Empleos': row['n_empleos'],
        'Rango Salarial': row['salario_max'] - row['salario_min'],
        'Estabilidad (%)': estabilidad,
        'Diversidad (%)': diversidad * 100,
        'Categoría Principal': row['categoria_principal'],
        'Categorías Únicas': categorias_unicas,
        'Salario Mínimo': row['salario_min'],
        'Salario Máximo': row['salario_max']
    })

metrics_df = pd.DataFrame(cluster_metrics)

# Mostrar tabla principal de métricas
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("Tabla de Métricas")
    
    # Formatear tabla para visualización
    display_df = metrics_df.copy()
    display_df['Salario Promedio'] = display_df['Salario Promedio'].apply(lambda x: f"${x:,.0f}")
    display_df['Rango Salarial'] = display_df['Rango Salarial'].apply(lambda x: f"${x:,.0f}")
    display_df['Salario Mínimo'] = display_df['Salario Mínimo'].apply(lambda x: f"${x:,.0f}")
    display_df['Salario Máximo'] = display_df['Salario Máximo'].apply(lambda x: f"${x:,.0f}")
    display_df['Estabilidad (%)'] = display_df['Estabilidad (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['Diversidad (%)'] = display_df['Diversidad (%)'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        display_df[['Cluster', 'Salario Promedio', 'Empleos', 'Estabilidad (%)', 'Diversidad (%)', 'Categoría Principal']],
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("Ranking Salarial")
    ranking_df = metrics_df.sort_values('Salario Promedio', ascending=False)
    
    for i, row in ranking_df.iterrows():
        st.metric(
            label=f"{row['Cluster']}",
            value=f"${row['Salario Promedio']:,.0f}",
            delta=f"{row['Empleos']} empleos"
        )

with col3:
    st.subheader("Mejor Estabilidad")
    estabilidad_df = metrics_df.sort_values('Estabilidad (%)', ascending=False)
    
    for i, row in estabilidad_df.head(3).iterrows():
        st.metric(
            label=f"{row['Cluster']}",
            value=f"{row['Estabilidad (%)']:.1f}%",
            delta=f"${row['Salario Promedio']:,.0f}"
        )

st.markdown("---")

# Gráficos simples de comparación
st.subheader("Comparativa Visual de Métricas")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    # Gráfico de barras para salarios
    fig_salarios = px.bar(
        metrics_df,
        x='Cluster',
        y='Salario Promedio',
        title='Salario Promedio por Cluster',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Set3[:len(metrics_df)],
        text_auto='.0f'
    )
    fig_salarios.update_traces(
        texttemplate='$%{text:,.0f}',
        textposition='outside'
    )
    fig_salarios.update_layout(
        height=400,
        showlegend=False,
        yaxis_title="Salario (USD)",
        xaxis_title=""
    )
    st.plotly_chart(fig_salarios, use_container_width=True)

with col_chart2:
    # Gráfico de burbujas simplificado
    fig_burbujas = px.scatter(
        metrics_df,
        x='Salario Promedio',
        y='Estabilidad (%)',
        size='Empleos',
        color='Cluster',
        hover_name='Cluster',
        hover_data=['Categoría Principal', 'Diversidad (%)'],
        title='Relación Salario-Estabilidad',
        color_discrete_sequence=px.colors.qualitative.Set3[:len(metrics_df)],
        size_max=50
    )
    fig_burbujas.update_layout(
        height=400,
        xaxis_title="Salario Promedio (USD)",
        yaxis_title="Estabilidad (%)"
    )
    st.plotly_chart(fig_burbujas, use_container_width=True)

st.markdown("---")

# Análisis detallado por cluster
st.subheader("Análisis Detallado por Cluster")

selected_cluster = st.selectbox(
    "Selecciona un cluster para análisis detallado:",
    options=metrics_df['Cluster'].tolist()
)

if selected_cluster:
    cluster_data = metrics_df[metrics_df['Cluster'] == selected_cluster].iloc[0]
    cluster_full_df = df_filtered[df_filtered['cluster_nombre'] == f'Cluster_{selected_cluster}']
    
    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
    
    with col_detail1:
        st.metric("Salario Promedio", f"${cluster_data['Salario Promedio']:,.0f}")
    
    with col_detail2:
        st.metric("Total Empleos", cluster_data['Empleos'])
    
    with col_detail3:
        st.metric("Estabilidad", f"{cluster_data['Estabilidad (%)']:.1f}%")
    
    with col_detail4:
        st.metric("Diversidad", f"{cluster_data['Diversidad (%)']:.1f}%")
    
    st.markdown("---")
    
    # Distribución de categorías en el cluster seleccionado
    if len(cluster_full_df) > 0:
        st.write(f"**Distribución de categorías en {selected_cluster}:**")
        
        categorias_dist = cluster_full_df['Categora_refinada'].value_counts().head(10)
        
        col_cat1, col_cat2 = st.columns(2)
        
        with col_cat1:
            fig_categorias = px.pie(
                values=categorias_dist.values,
                names=categorias_dist.index,
                title=f'Top Categorías en {selected_cluster}',
                hole=0.3
            )
            fig_categorias.update_layout(height=400)
            st.plotly_chart(fig_categorias, use_container_width=True)
        
        with col_cat2:
            st.write("**Categorías principales:**")
            for categoria, count in categorias_dist.head(5).items():
                porcentaje = (count / len(cluster_full_df)) * 100
                st.write(f"• {categoria}: {count} empleos ({porcentaje:.1f}%)")
            
            st.write(f"\n**Categoría principal:** {cluster_data['Categoría Principal']}")
            st.write(f"**Categorías únicas:** {cluster_data['Categorías Únicas']}")
    
    # Empleos representativos
    st.write(f"**Empleos representativos en {selected_cluster}:**")
    empleos_representativos = cluster_full_df.nlargest(5, 'salario_limpio')[['Categora_refinada', 'salario_limpio']]
    
    for _, empleo in empleos_representativos.iterrows():
        st.write(f"• {empleo['Categora_refinada']}: ${empleo['salario_limpio']:,.0f}")

# Resumen ejecutivo
st.markdown("---")
st.subheader("Resumen Ejecutivo")

col_res1, col_res2 = st.columns(2)

with col_res1:
    mejor_salario = metrics_df.loc[metrics_df['Salario Promedio'].idxmax()]
    mejor_estabilidad = metrics_df.loc[metrics_df['Estabilidad (%)'].idxmax()]
    
    st.info(f"""
    **Cluster con Mayor Salario: {mejor_salario['Cluster']}**
    - Salario promedio: ${mejor_salario['Salario Promedio']:,.0f}
    - {mejor_salario['Empleos']} empleos disponibles
    - Estabilidad: {mejor_salario['Estabilidad (%)']:.1f}%
    """)
    
    st.success(f"""
    **Cluster más Estable: {mejor_estabilidad['Cluster']}**
    - Estabilidad: {mejor_estabilidad['Estabilidad (%)']:.1f}%
    - Salario promedio: ${mejor_estabilidad['Salario Promedio']:,.0f}
    - {mejor_estabilidad['Empleos']} empleos disponibles
    """)

with col_res2:
    # Recomendaciones basadas en los datos
    st.write("**Recomendaciones:**")
    
    # Encontrar el mejor balance
    metrics_df['Score_Balance'] = (
        (metrics_df['Salario Promedio'] / metrics_df['Salario Promedio'].max()) * 0.5 +
        (metrics_df['Estabilidad (%)'] / 100) * 0.3 +
        (metrics_df['Diversidad (%)'] / 100) * 0.2
    )
    
    mejor_balance = metrics_df.loc[metrics_df['Score_Balance'].idxmax()]
    
    st.write(f"1. **Para mejor balance**: Considerar **{mejor_balance['Cluster']}**")
    st.write(f"   • Score de balance: {mejor_balance['Score_Balance']:.2f}/1.0")
    st.write(f"   • Salario: ${mejor_balance['Salario Promedio']:,.0f}")
    st.write(f"   • Estabilidad: {mejor_balance['Estabilidad (%)']:.1f}%")
    
    st.write(f"2. **Para máximo salario**: **{mejor_salario['Cluster']}**")
    st.write(f"3. **Para estabilidad**: **{mejor_estabilidad['Cluster']}**")

# Opción para descargar el análisis
st.markdown("---")
st.subheader("Exportar Análisis")

csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Métricas de Clusters",
    data=csv_metrics,
    file_name=f"metricas_clusters_top{top_n}.csv",
    mime="text/csv",
    help="Descarga las métricas detalladas de todos los clusters analizados"
)

# ============================================================
# SECCIÓN 8: EXPORTACIÓN DE DATOS
# ============================================================

st.markdown("---")
st.header("Exportación de Resultados")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    # Exportar datos completos
    csv_completo = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Datos Completos",
        data=csv_completo,
        file_name="datos_analisis_clusters.csv",
        mime="text/csv",
        help="Incluye todos los empleos con su cluster asignado"
    )

with col_exp2:
    # Exportar resumen por cluster
    resumen_export = cluster_summary_filtered.copy()
    resumen_export.index = [c.replace('Cluster_', '') for c in resumen_export.index]
    csv_resumen = resumen_export.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="📊 Descargar Resumen por Cluster",
        data=csv_resumen,
        file_name="resumen_clusters.csv",
        mime="text/csv",
        help="Resumen estadístico de cada cluster"
    )

with col_exp3:
    # Exportar análisis de trayectorias
    trayectorias_df = pd.DataFrame([
        {
            'Trayectoria': nombre,
            'Etapa 1': datos['etapas'][0],
            'Salario 1': datos['salarios'][0],
            'Etapa 2': datos['etapas'][1],
            'Salario 2': datos['salarios'][1],
            'Etapa 3': datos['etapas'][2],
            'Salario 3': datos['salarios'][2],
            'Crecimiento %': ((datos['salarios'][2] - datos['salarios'][0]) / datos['salarios'][0]) * 100
        }
        for nombre, datos in trayectorias.items()
    ])
    
    csv_trayectorias = trayectorias_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="🛤️ Descargar Trayectorias",
        data=csv_trayectorias,
        file_name="trayectorias_profesionales.csv",
        mime="text/csv",
        help="Análisis de trayectorias profesionales"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
**Notas Metodológicas:**
- Los datos han sido normalizados y limpiados para garantizar consistencia
- Los clusters se han definido mediante análisis de similitud semántica
- Todos los salarios se expresan en USD mensuales
- La estabilidad se calcula en función del rango salarial dentro de cada cluster

**Uso Recomendado:**
1. Utilice los filtros para personalizar el análisis según sus intereses
2. Explore las trayectorias profesionales para planificar su desarrollo
3. Consulte el perfil de clusters para identificar oportunidades alineadas con su perfil
""")

# ============================================================
# ESTILOS CSS ADICIONALES
# ============================================================

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
    }
    
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)
