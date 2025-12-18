# ============================================================
# DASHBOARD DE ANÁLISIS DE CLUSTERS SALARIALES (USD)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

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

    categoria_col = "Categora"

    cluster_summary = df.groupby("cluster_nombre").agg(
        salario_promedio=("salario_limpio", "mean"),
        salario_min=("salario_limpio", "min"),
        salario_max=("salario_limpio", "max"),
        n_empleos=("salario_limpio", "count"),
        categoria_principal=(categoria_col, lambda x: x.value_counts().idxmax())
    ).round(0)

    cluster_summary = cluster_summary.sort_values(
        "salario_promedio", ascending=False
    )

    return df, cluster_summary, categoria_col

df, cluster_summary, categoria_col = load_data()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Configuración")

    min_salary, max_salary = st.slider(
        "Rango salarial (USD)",
        int(df["salario_limpio"].min()),
        int(df["salario_limpio"].max()),
        (
            int(df["salario_limpio"].min()),
            int(df["salario_limpio"].max())
        )
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
# SECCIÓN 1: PIRÁMIDE SALARIAL (SEMÁNTICA)
# ============================================================

st.header("Pirámide Salarial por Categoría")
st.write("Distribución del salario promedio usando la categoría dominante de cada cluster.")

fig_pyramid = go.Figure(go.Bar(
    y=cluster_filtered["categoria_principal"],
    x=cluster_filtered["salario_promedio"],
    orientation="h",
    text=[f"${x:,.0f}" for x in cluster_filtered["salario_promedio"]],
    textposition="outside"
))

fig_pyramid.update_layout(
    height=450,
    xaxis_title="Salario promedio mensual (USD)",
    yaxis_title="Categoría"
)

st.plotly_chart(fig_pyramid, use_container_width=True)

# ============================================================
# SECCIÓN 2: MÉTRICAS DE CALIDAD DEL CLUSTERING
# ============================================================

st.header("Métricas de Calidad del Clustering")

encoded_categories = LabelEncoder().fit_transform(
    df_filtered[categoria_col].astype(str)
)

silhouette = silhouette_score(
    df_filtered[["salario_limpio"]],
    df_filtered["cluster_nombre"]
)

ari = adjusted_rand_score(
    encoded_categories,
    df_filtered["cluster_nombre"]
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Silhouette Score", f"{silhouette:.3f}")

with col2:
    st.metric("ARI (Clusters vs Categoría)", f"{ari:.3f}")

# ============================================================
# SECCIÓN 3: TRAYECTORIAS PROFESIONALES (DESDE DATOS)
# ============================================================

st.header("Trayectorias Profesionales y Evolución Salarial")
st.write("Evolución salarial calculada a partir de las categorías laborales reales.")

niveles = {
    "Sin titulación": ["practica", "trainee", "junior", "entry"],
    "Técnico level": ["tecnico", "analista"],
    "Técnico senior": ["senior", "lead"],
    "Asistente": ["asistente"],
    "Coordinador": ["coordinador"],
    "Gerente": ["gerente"],
    "Director": ["director"]
}

trayectoria_data = []

for nivel, palabras in niveles.items():
    mask = df_filtered[categoria_col].str.lower().str.contains(
        "|".join(palabras), na=False
    )
    if mask.sum() >= 5:
        trayectoria_data.append({
            "nivel": nivel,
            "salario": df_filtered.loc[mask, "salario_limpio"].mean()
        })

trayectoria_df = pd.DataFrame(trayectoria_data)

fig_trayectorias = go.Figure(go.Scatter(
    x=trayectoria_df["nivel"],
    y=trayectoria_df["salario"],
    mode="lines+markers+text",
    text=[f"${x:,.0f}" for x in trayectoria_df["salario"]],
    textposition="top center"
))

fig_trayectorias.update_layout(
    height=450,
    xaxis_title="Nivel profesional",
    yaxis_title="Salario mensual (USD)"
)

st.plotly_chart(fig_trayectorias, use_container_width=True)

# ============================================================
# TEXTO ENRIQUECIDO
# ============================================================

st.markdown(
    f"""
    **Análisis de las trayectorias profesionales**

    Los resultados muestran una progresión salarial clara conforme aumenta el nivel profesional.
    Los perfiles iniciales asociados a **sin titulación** presentan los salarios más bajos,
    mientras que los niveles de **técnico senior** y roles de **coordinación y dirección**
    concentran los mayores ingresos.

    En promedio, el salario se incrementa de **${trayectoria_df.iloc[0]['salario']:,.0f}**
    en los niveles iniciales a **${trayectoria_df.iloc[-1]['salario']:,.0f}**
    en los niveles más altos, evidenciando el impacto directo de la experiencia,
    la especialización y la responsabilidad laboral.
    """
)

# ============================================================
# SECCIÓN 4: PERFIL DETALLADO POR CATEGORÍA
# ============================================================

st.header("Perfil Detallado por Categoría")

selected_category = st.selectbox(
    "Selecciona una categoría",
    cluster_filtered["categoria_principal"].unique()
)

category_clusters = cluster_filtered[
    cluster_filtered["categoria_principal"] == selected_category
]

avg_salary = category_clusters["salario_promedio"].mean()
total_jobs = category_clusters["n_empleos"].sum()

col1, col2 = st.columns(2)

with col1:
    st.metric("Salario promedio", f"${avg_salary:,.0f}")

with col2:
    st.metric("Total de empleos", int(total_jobs))

st.dataframe(
    df_filtered[df_filtered[categoria_col] == selected_category][
        ["Título", "Empresa", "salario_limpio"]
    ]
    .rename(columns={"salario_limpio": "Salario USD"})
    .head(10),
    use_container_width=True
)
