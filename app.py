# ============================================================
# DASHBOARD DE ANÁLISIS SALARIAL (USD) – LIMPIO Y ACADÉMICO
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
st.set_page_config(
    page_title="Análisis Salarial por Categoría",
    layout="wide"
)

st.title("Análisis Salarial por Categoría (USD)")

# ------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("empleos_analisis_final.csv")
    return df

df = load_data()

# ------------------------------------------------------------
# PREPARACIÓN: CLUSTERS → CATEGORÍA DOMINANTE
# ------------------------------------------------------------
categoria_dominante = (
    df.groupby("cluster_nombre")["Categora"]
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)

df["Cluster_Categoria"] = df["cluster_nombre"].map(categoria_dominante)

# Evitar nombres repetidos
counts = df["Cluster_Categoria"].value_counts()
repeated = counts[counts > 1].index.tolist()

if repeated:
    for cat in repeated:
        idxs = df[df["Cluster_Categoria"] == cat]["cluster_nombre"].unique()
        for i, cl in enumerate(idxs, start=1):
            df.loc[df["cluster_nombre"] == cl, "Cluster_Categoria"] = f"{cat} ({i})"

# ------------------------------------------------------------
# FILTRO SALARIAL
# ------------------------------------------------------------
min_sal, max_sal = int(df.salario_limpio.min()), int(df.salario_limpio.max())
rango = st.slider(
    "Rango salarial (USD)",
    min_sal, max_sal, (min_sal, max_sal)
)

df = df[(df.salario_limpio >= rango[0]) & (df.salario_limpio <= rango[1])]

# ============================================================
# GRÁFICO 1 – BOXPLOT SALARIAL POR CATEGORÍA
# ============================================================
st.markdown("**Distribución salarial mensual por categoría laboral.**")

fig_box = px.box(
    df,
    x="Categora",
    y="salario_limpio",
    points="outliers"
)
fig_box.update_layout(
    yaxis_title="Salario mensual (USD)",
    xaxis_title="Categoría"
)

st.plotly_chart(fig_box, use_container_width=True)

# ============================================================
# GRÁFICO 2 – SALARIO PROMEDIO POR CATEGORÍA
# ============================================================
st.markdown("**Comparación del salario promedio mensual entre categorías.**")

avg_cat = df.groupby("Categora")["salario_limpio"].mean().sort_values()

fig_avg_cat = px.bar(
    avg_cat,
    orientation="h",
    labels={"value": "Salario promedio (USD)", "index": "Categoría"}
)

st.plotly_chart(fig_avg_cat, use_container_width=True)

# ============================================================
# GRÁFICO 3 – SALARIO PROMEDIO POR CLUSTER SALARIAL
# ============================================================
st.markdown("**Salario promedio mensual según agrupación salarial.**")

avg_cluster = (
    df.groupby("Cluster_Categoria")["salario_limpio"]
    .mean()
    .sort_values()
)

fig_cluster = px.bar(
    avg_cluster,
    orientation="h",
    labels={"value": "Salario promedio (USD)", "index": "Cluster (categoría dominante)"}
)

st.plotly_chart(fig_cluster, use_container_width=True)

# ============================================================
# GRÁFICO 4 – PIRÁMIDE SALARIAL
# ============================================================
st.markdown("**Jerarquía salarial de los clusters según ingreso promedio.**")

fig_piramide = go.Figure(
    go.Bar(
        x=avg_cluster.values,
        y=avg_cluster.index,
        orientation="h",
        text=[f"${v:,.0f}" for v in avg_cluster.values],
        textposition="outside"
    )
)

fig_piramide.update_layout(
    xaxis_title="Salario promedio (USD)",
    yaxis_title="Cluster",
    height=500
)

st.plotly_chart(fig_piramide, use_container_width=True)

# ============================================================
# GRÁFICO 5 – TRAYECTORIAS PROFESIONALES
# ============================================================
st.markdown("**Evolución salarial estimada según trayectoria formativa y técnica.**")

trayectorias = {
    "Trayectoria técnica": {
        "etapas": ["Sin titulación", "Nivel técnico", "Técnico senior"],
        "salarios": [1200, 2200, 3800]
    }
}

fig_tray = go.Figure()

for nombre, t in trayectorias.items():
    fig_tray.add_trace(
        go.Scatter(
            x=t["etapas"],
            y=t["salarios"],
            mode="lines+markers+text",
            name=nombre,
            text=[f"${s:,.0f}" for s in t["salarios"]],
            textposition="top center"
        )
    )

fig_tray.update_layout(
    yaxis_title="Salario mensual (USD)",
    xaxis_title="Etapa profesional",
    height=450
)

st.plotly_chart(fig_tray, use_container_width=True)
