import pandas as pd
import numpy as np
import streamlit as st

def convert_columns_to_numeric(df, threshold=0.7):
    df = df.copy()
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() >= threshold:
            df[col] = coerced
    return df

def normalizar_valor(v):
    if pd.isna(v):
        return np.nan
    return str(v).strip().lower()


def clean_dataframe(df, remove_unknowns=True, custom_remove_list=None):
    df_clean = df.copy()
    initial_rows = len(df_clean)

    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates()

    df_clean = convert_columns_to_numeric(df_clean)

    if remove_unknowns:
        base_unknowns = ["UNKNOWN", "Unknown", "unknown", "N/A", "na", "NA"]
        df_clean = df_clean.replace(base_unknowns, np.nan)

    if custom_remove_list:
        df_clean = df_clean.replace(custom_remove_list, np.nan)

    df_clean = df_clean.dropna()

    total_removed = initial_rows - len(df_clean)
    return df_clean, total_removed, duplicate_count

def valores_eliminados(df_original, df_clean, remove_unknowns=True, custom_remove_list=None):

    # normalizamos
    df_o_norm = df_original.applymap(normalizar_valor)
    df_c_norm = df_clean.applymap(normalizar_valor)

    originales = set(df_o_norm.values.flatten())
    finales = set(df_c_norm.values.flatten())

    # lista unknowns
    base_unknowns = []
    if remove_unknowns:
        base_unknowns = ["unknown", "n/a", "na"]

    # lista custom
    custom_list = [normalizar_valor(v) for v in custom_remove_list] if custom_remove_list else []

    valores_a_buscar = set(base_unknowns + custom_list)

    # qué desapareció
    desaparecidos = originales - finales

    # devolver SOLO los valores que estaban en lo que el usuario quería eliminar
    eliminados = {v for v in desaparecidos if v in valores_a_buscar}

    return eliminados



def count_outliers(series):
    """Cuenta outliers usando el método del IQR."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((series < lower) | (series > upper)).sum()


def data_quality_report(df, removed_rows, removed_duplicates):
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "null_count": df.isnull().sum(),
        "null_%": df.isnull().mean() * 100,
        "unique_values": df.nunique(),
        "outliers": df.apply(count_outliers)
    })

    report.loc["TOTAL", "removed_duplicates"] = removed_duplicates
    report.loc["TOTAL", "total_removed_rows"] = removed_rows

    score = (
        100
        - report["null_%"].fillna(0)
        - report["outliers"].fillna(0) * 0.5
    )

    report["quality_score"] = score.clip(0, 100)
    return report


#streamlit

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode("utf-8")

st.set_page_config(
    page_title="Herramienta de Calidad de Datos",
    layout="wide"
)

st.title("Analizador de Calidad de Datos")
st.markdown("Sube tu archivo CSV para obtener un dataset limpio y un reporte de calidad completo.")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])


if uploaded_file is not None:

    df_original = pd.read_csv(uploaded_file)
    st.subheader(f"Datos Originales ({len(df_original)} filas)")
    st.dataframe(df_original.head())

    st.markdown("### Opciones de Limpieza")

    remove_unknowns = st.checkbox(
        "Eliminar valores como 'UNKNOWN', 'N/A', 'na', etc.", 
        value=True
    )

    custom_values = st.text_input(
        "Valores adicionales a eliminar (separados por coma)",
        value="",
        placeholder="Ej: Sin dato, --, vacío"
    )

    if custom_values.strip() != "":
        custom_values_list = [v.strip() for v in custom_values.split(",")]
    else:
        custom_values_list = []

    st.markdown("---")
    st.header("Proceso de Limpieza")

    df_clean, rows_removed, duplicates_removed = clean_dataframe(
        df_original,
        remove_unknowns=remove_unknowns,
        custom_remove_list=custom_values_list
    )
    deleted_values = valores_eliminados(
        df_original, 
        df_clean, 
        remove_unknowns=remove_unknowns,
        custom_remove_list=custom_values_list
    )

    report = data_quality_report(df_clean, rows_removed, duplicates_removed)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resumen de Limpieza")
        st.metric("Filas Originales", len(df_original))
        st.metric("Duplicados Eliminados", duplicates_removed)
        st.metric("Filas Eliminadas Totales", rows_removed)
        st.metric("Filas Finales Limpias", len(df_clean))

        csv_clean = convert_df_to_csv(df_clean)
        st.download_button(
            "Descargar Dataset Limpio",
            csv_clean,
            file_name="clean_data.csv",
            mime="text/csv"
        )

    with col2:
        st.subheader("Reporte de Calidad")
        st.dataframe(
            report.style.format({
                "null_%": "{:.2f}%",
                "outliers": "{:.0f}",
                "quality_score": "{:.2f}"
            })
        )

        csv_report = convert_df_to_csv(report)
        st.download_button(
            "Descargar Reporte",
            csv_report,
            file_name="quality_report.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.header("Dataset Limpio Listo para Tableau")
    st.dataframe(df_clean.head())
