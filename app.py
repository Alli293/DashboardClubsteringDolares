# ============================================================
# DASHBOARD DE ANÁLISIS DE CLUSTERS SALARIALES (USD)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard de Clusters Salariales (USD)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("empleos_analisis_final.csv")
    
    cluster_summary = df.groupby("cluster_nombre").agg({
        "salario_limpio": ["mean", "min", "max", "count"],
        "Categora_refinada": lambda x: x.value_counts().index[0]
    }).round(0)

    cluster_summary.columns = [
        "salario_promedio",
        "salario_min",
        "salario_max",
        "n_empleos",
        "categoria_principal"
    ]

    cluster_summary = cluster_summary.sort_values("salario_promedio", ascending=False)

    return df, cluster_summary

df, cluster_summary = load_data()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Configuración")

    min_salary, max_salary = st.slider(
        "Rango salarial (USD)",
        int(df["salario_limpio"].min()),
        int(df["salario_limpio"].max()),
        (int(df["salario_limpio"].min()), int(df["salario_limpio"].max()))
    )

    n_clusters = st.slider(
        "Cantidad de clusters a mostrar",
        min_value=3,
        max_value=15,
        value=7
    )

# ============================================================
# FILTROS
# ============================================================

df_filtered = df[
    (df["salario_limpio"] >= min_salary) &
    (df["salario_limpio"] <= max_salary)
]

cluster_filtered = cluster_summary.head(n_clusters)

# ============================================================
# SECCIÓN 1: PIRÁMIDE SALARIAL
# ============================================================

st.header("Pirámide Salarial por Cluster")
st.write("Muestra la distribución del salario promedio por cluster.")

clusters_ordered = cluster_filtered.sort_values("salario_promedio").index

fig_pyramid = go.Figure(go.Bar(
    y=[str(c) for c in clusters_ordered],
    x=cluster_filtered.loc[clusters_ordered]["salario_promedio"],
    orientation="h",
    text=[f"${x:,.0f}" for x in cluster_filtered.loc[clusters_ordered]["salario_promedio"]],
    textposition="outside"
))

fig_pyramid.update_layout(
    height=450,
    xaxis_title="Salario promedio mensual (USD)",
    yaxis_title="Cluster"
)

st.plotly_chart(fig_pyramid, use_container_width=True)

# ============================================================
# SECCIÓN 2: DISTRIBUCIÓN DE EMPLEOS
# ============================================================

st.header("Distribución de Empleos por Cluster")
st.write("Representa la proporción de empleos en cada cluster.")

fig_donut = go.Figure(go.Pie(
    labels=[str(c) for c in cluster_filtered.index],
    values=cluster_filtered["n_empleos"],
    hole=0.4
))

fig_donut.update_layout(height=400)

st.plotly_chart(fig_donut, use_container_width=True)

# ============================================================
# SECCIÓN 3: SALARIO VS ESTABILIDAD
# ============================================================

st.header("Salario vs Estabilidad")
st.write("Relaciona salario promedio con estabilidad salarial por cluster.")

stability_data = []

for cluster, row in cluster_filtered.iterrows():
    rango = row["salario_max"] - row["salario_min"]
    estabilidad = 100 - (rango / row["salario_promedio"]) * 20

    stability_data.append({
        "cluster": cluster,
        "salario": row["salario_promedio"],
        "estabilidad": max(0, min(100, estabilidad)),
        "empleos": row["n_empleos"]
    })

stability_df = pd.DataFrame(stability_data)

fig_bubble = px.scatter(
    stability_df,
    x="salario",
    y="estabilidad",
    size="empleos",
    color="cluster",
    hover_name="cluster",
    size_max=60
)

fig_bubble.update_layout(
    height=500,
    xaxis_title="Salario promedio (USD)",
    yaxis_title="Estabilidad"
)

st.plotly_chart(fig_bubble, use_container_width=True)

# ============================================================
# SECCIÓN 4: TRAYECTORIAS PROFESIONALES
# ============================================================

st.header("Trayectorias Profesionales y Evolución Salarial")
st.write("Simula la progresión salarial por nivel de formación.")

trayectorias = {
    "Ruta Técnica": {
        "etapas": ["Sin titulación", "Nivel técnico", "Técnico senior"],
        "salarios": [900, 1800, 3200]
    }
}

fig_trayectorias = go.Figure()

for nombre, datos in trayectorias.items():
    fig_trayectorias.add_trace(go.Scatter(
        x=datos["etapas"],
        y=datos["salarios"],
        mode="lines+markers+text",
        name=nombre,
        text=[f"${s:,.0f}" for s in datos["salarios"]],
        textposition="top center"
    ))

fig_trayectorias.update_layout(
    height=450,
    xaxis_title="Etapa profesional",
    yaxis_title="Salario mensual (USD)"
)

st.plotly_chart(fig_trayectorias, use_container_width=True)

# ============================================================
# SECCIÓN 5: DISTRIBUCIÓN DE CATEGORÍAS
# ============================================================

st.header("Distribución de Categorías por Cluster")
st.write("Muestra la concentración de categorías laborales por cluster.")

categoria_col = "Categora_refinada"

top_categories = df_filtered[categoria_col].value_counts().head(10).index

heatmap_data = []

for cluster in cluster_filtered.index:
    cluster_df = df_filtered[df_filtered["cluster_nombre"] == cluster]
    heatmap_data.append([
        len(cluster_df[cluster_df[categoria_col] == cat]) for cat in top_categories
    ])

fig_heatmap = go.Figure(go.Heatmap(
    z=heatmap_data,
    x=top_categories,
    y=[str(c) for c in cluster_filtered.index],
    colorscale="YlOrRd",
    text=heatmap_data,
    texttemplate="%{text}"
))

fig_heatmap.update_layout(
    height=500,
    xaxis_title="Categoría",
    yaxis_title="Cluster"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================
# SECCIÓN 6: PERFIL DE CLUSTER
# ============================================================

st.header("Perfil Detallado de Cluster")
st.write("Explora el detalle salarial y laboral de un cluster específico.")

selected_cluster = st.selectbox(
    "Selecciona un cluster",
    cluster_filtered.index.astype(str)
)

cluster_data = cluster_filtered.loc[selected_cluster]
cluster_jobs = df_filtered[df_filtered["cluster_nombre"] == selected_cluster]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Salario promedio", f"${cluster_data['salario_promedio']:,.0f}")

with col2:
    st.metric("Empleos", cluster_data["n_empleos"])

with col3:
    rango = cluster_data["salario_max"] - cluster_data["salario_min"]
    estabilidad = 100 - (rango / cluster_data["salario_promedio"]) * 20
    st.metric("Estabilidad", f"{estabilidad:.1f}%")

st.dataframe(
    cluster_jobs[["Título", "Empresa", categoria_col, "salario_limpio"]]
    .rename(columns={"salario_limpio": "Salario USD"})
    .head(10),
    use_container_width=True
)
