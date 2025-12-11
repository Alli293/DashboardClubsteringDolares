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
# SECCIÓN 6: PERFIL DE CLUSTERS TOP - VERSIÓN SIMPLIFICADA
# ============================================================

st.header("Perfil Comparativo de Clusters")

# Seleccionar top clusters para análisis
top_n = st.slider("Número de clusters a comparar:", 2, 5, 3)
top_clusters = cluster_summary_filtered.head(top_n).index.tolist()

# Definir métricas para radar
metrics = ['Salario Promedio', 'Número Empleos', 'Estabilidad', 'Diversidad Categorías', 'Rango Salarial']

# Preparar datos para radar
radar_data = {}
for cluster in top_clusters:
    cluster_df = df_filtered[df_filtered['cluster_nombre'] == cluster]
    
    # Calcular métricas básicas
    salario_prom = cluster_df['salario_limpio'].mean()
    n_empleos = len(cluster_df)
    
    # Calcular rango salarial relativo
    if salario_prom > 0:
        rango_salarial = (cluster_df['salario_limpio'].max() - cluster_df['salario_limpio'].min()) / salario_prom
    else:
        rango_salarial = 0
    
    # Calcular estabilidad (inversa del rango)
    estabilidad = max(0, min(100, 100 - (rango_salarial * 20)))
    
    # Calcular diversidad de categorías
    if len(cluster_df) > 0:
        diversidad = cluster_df['Categora_refinada'].nunique() / len(cluster_df)
    else:
        diversidad = 0
    
    # Almacenar métricas (usar valores normalizados para el radar)
    radar_data[cluster] = {
        'salario': salario_prom,
        'empleos': n_empleos,
        'estabilidad': estabilidad,
        'diversidad': diversidad * 100,  # Convertir a porcentaje
        'rango': 100 - (rango_salarial * 30)  # Invertir para mejor visualización
    }

# Normalizar valores para el radar (0-1)
all_values = []
for cluster_data in radar_data.values():
    all_values.append(list(cluster_data.values()))

all_values_array = np.array(all_values)
normalized_data = {}
for i, cluster in enumerate(top_clusters):
    cluster_values = all_values_array[i]
    normalized = (cluster_values - cluster_values.min()) / (cluster_values.max() - cluster_values.min() + 1e-10)
    normalized_data[cluster] = normalized.tolist()

# Crear gráfico de radar
fig_radar = go.Figure()

# Paleta de colores profesional
colors_fill = [
    'rgba(255, 99, 132, 0.2)',   # Rojo suave
    'rgba(54, 162, 235, 0.2)',   # Azul suave
    'rgba(255, 206, 86, 0.2)',   # Amarillo suave
    'rgba(75, 192, 192, 0.2)',   # Verde suave
    'rgba(153, 102, 255, 0.2)'   # Púrpura suave
]

colors_line = [
    'rgb(255, 99, 132)',    # Rojo
    'rgb(54, 162, 235)',    # Azul
    'rgb(255, 206, 86)',    # Amarillo
    'rgb(75, 192, 192)',    # Verde
    'rgb(153, 102, 255)'    # Púrpura
]

for idx, cluster in enumerate(top_clusters):
    cluster_name = cluster.replace('Cluster_', '')
    values = normalized_data[cluster]
    values += values[:1]  # Cerrar el círculo
    
    # Crear texto para hover
    raw_data = radar_data[cluster]
    hover_text = []
    for i, metric in enumerate(metrics):
        if metric == 'Salario Promedio':
            hover_text.append(f"{metric}: ${raw_data['salario']:,.0f}")
        elif metric == 'Número Empleos':
            hover_text.append(f"{metric}: {int(raw_data['empleos'])}")
        elif metric == 'Estabilidad':
            hover_text.append(f"{metric}: {raw_data['estabilidad']:.1f}%")
        elif metric == 'Diversidad Categorías':
            hover_text.append(f"{metric}: {raw_data['diversidad']:.1f}%")
        elif metric == 'Rango Salarial':
            hover_text.append(f"{metric}: {raw_data['rango']:.1f}")
    
    hover_text += [hover_text[0]]  # Cerrar el círculo
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics + [metrics[0]],
        name=cluster_name,
        fill='toself',
        fillcolor=colors_fill[idx % len(colors_fill)],
        line=dict(color=colors_line[idx % len(colors_line)], width=2),
        hoverinfo='text',
        hovertext=hover_text
    ))

# Configurar layout del radar
fig_radar.update_layout(
    height=600,
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            showticklabels=False,
            showline=False
        ),
        angularaxis=dict(
            tickfont=dict(size=12),
            direction='clockwise'
        ),
        bgcolor='rgba(240, 240, 240, 0.1)'
    ),
    title=dict(
        text=f"Comparativa de Top {top_n} Clusters",
        font=dict(size=18, family="Arial, sans-serif")
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    ),
    margin=dict(l=50, r=50, t=80, b=50),
    paper_bgcolor='rgba(255, 255, 255, 0.9)',
    plot_bgcolor='rgba(255, 255, 255, 0.9)'
)

st.plotly_chart(fig_radar, use_container_width=True)

# Mostrar tabla con valores reales
st.subheader("Valores Reales de las Métricas")

# Crear tabla resumen
summary_data = []
for cluster in top_clusters:
    cluster_name = cluster.replace('Cluster_', '')
    data = radar_data[cluster]
    summary_data.append({
        'Cluster': cluster_name,
        'Salario Promedio': f"${data['salario']:,.0f}",
        'Número Empleos': data['empleos'],
        'Estabilidad': f"{data['estabilidad']:.1f}%",
        'Diversidad': f"{data['diversidad']:.1f}%",
        'Rango Salarial': f"{data['rango']:.1f}"
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)
# ============================================================
# SECCIÓN 7: RESUMEN Y RECOMENDACIONES
# ============================================================

st.markdown("---")
st.header("Conclusiones y Recomendaciones")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Principales Hallazgos")
    
    # Hallazgo 1: Cluster líder
    top_cluster = cluster_summary_filtered.iloc[0]
    st.info(f"""
    **Cluster Líder**: {top_cluster.name.replace('Cluster_', '')}
    - Salario promedio: ${top_cluster['salario_promedio']:,.0f}
    - {top_cluster['n_empleos']} empleos analizados
    - Categoría principal: {top_cluster['categoria_principal']}
    """)
    
    # Hallazgo 2: Mejor relación valor
    if len(cluster_summary_filtered) > 1:
        best_value = cluster_summary_filtered.iloc[1]
        st.success(f"""
        **Mejor Relación Valor**: {best_value.name.replace('Cluster_', '')}
        - Salario: ${best_value['salario_promedio']:,.0f}
        - {best_value['n_empleos']} oportunidades disponibles
        - Excelente equilibrio salario-estabilidad
        """)

with col2:
    st.subheader("Recomendaciones Estratégicas")
    
    st.markdown("""
    1. **Enfoque en Crecimiento**: Desarrollar habilidades hacia clusters de mayor salario
    2. **Diversificación**: Explorar múltiples categorías dentro del cluster objetivo
    3. **Planificación de Carrera**: Seguir trayectorias probadas para maximizar ingresos
    4. **Especialización**: Profundizar en categorías con alta demanda y remuneración
    """)

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
