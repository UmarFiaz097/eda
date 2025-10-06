import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Economic Data Analysis", layout="wide")
st.title("📊 Economic Data Analysis")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_file(file, sheet_name=None, encoding="utf-8", sep=","):
    name = getattr(file, "name", "")
    suffix = os.path.splitext(name)[1].lower()

    if suffix in [".csv", ".txt"]:
        try:
            return pd.read_csv(file, encoding=encoding, sep=sep)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1", sep=sep)
        except Exception as e:
            raise RuntimeError(f"CSV read error: {e}")

    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Excel read error: {e}")

    raise RuntimeError("Unsupported file type. Please upload CSV/XLSX/XLS.")

def is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    try:
        pd.to_datetime(s.dropna().head(10), errors="raise")
        return True
    except Exception:
        return False

# ---------------------------
# File uploader
# ---------------------------
st.sidebar.header("Upload & Options")
Ufile = st.sidebar.file_uploader(
    "Upload data file",
    type=["csv", "txt", "xlsx", "xls"],
    help="CSV or Excel supported"
)

csv_sep = st.sidebar.text_input("CSV delimiter (for CSV/TXT)", value=",")
csv_encoding = st.sidebar.text_input("CSV encoding", value="utf-8")

sheet_name = None
if Ufile and os.path.splitext(Ufile.name)[1].lower() in [".xlsx", ".xls"]:
    try:
        xls = pd.ExcelFile(Ufile)
        Ufile.seek(0)
        sheet_name = st.sidebar.selectbox("Excel sheet", options=xls.sheet_names, index=0)
    except Exception:
        sheet_name = None

if Ufile:
    try:
        df = load_file(Ufile, sheet_name=sheet_name, encoding=csv_encoding, sep=csv_sep)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("🔎 Preview")
    st.write(df.head(10))

    # ---------------------------
    # Data health & summary
    # ---------------------------
    st.subheader("🩺 Data Health")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    with st.expander("Missing values by column"):
        na_table = df.isna().sum().to_frame("missing_count")
        na_table["missing_%"] = (na_table["missing_count"] / len(df) * 100).round(2)
        st.dataframe(na_table)

    st.subheader("📈 Summary Statistics")
    if df.select_dtypes(include=np.number).shape[1] > 0:
        st.markdown("**Numeric columns**")
        st.dataframe(df.describe().T)
    if df.select_dtypes(exclude=np.number).shape[1] > 0:
        st.markdown("**Non-numeric columns (unique counts)**")
        nunique = df.select_dtypes(exclude=np.number).nunique().sort_values(ascending=False)
        st.dataframe(nunique.to_frame("unique_values"))

    # ---------------------------
    # Column-wise analysis
    # ---------------------------
    st.subheader("🧭 Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns, index=0)

    col = df[column]
    if is_datetime_series(col):
        if not pd.api.types.is_datetime64_any_dtype(col):
            with st.spinner("Parsing datetimes..."):
                df[column] = pd.to_datetime(col, errors="coerce")
        st.write(f"Summary of **{column}** (datetime):")
        st.caption("Note: Parsed with `pandas.to_datetime(..., errors='coerce')`.")
        agg_target = st.selectbox(
            "Pick a numeric column to aggregate (count if none)",
            options=["<count>"] + list(df.select_dtypes(include=np.number).columns),
            index=0
        )
        freq = st.selectbox(
            "Resample frequency",
            ["D - Daily", "W - Weekly", "M - Monthly", "Q - Quarterly", "Y - Yearly"],
            index=2
        )
        freq_map = {"D - Daily": "D", "W - Weekly": "W", "M - Monthly": "M", "Q - Quarterly": "Q", "Y - Yearly": "Y"}

        tmp = df[[column]].copy()
        if agg_target != "<count>":
            tmp[agg_target] = df[agg_target]
        tmp = tmp.dropna(subset=[column]).sort_values(column)
        tmp = tmp.set_index(column)

        if agg_target == "<count>":
            series = tmp.resample(freq_map[freq]).size().rename("count")
        else:
            series = tmp[agg_target].resample(freq_map[freq]).mean()

        st.line_chart(series)

    elif pd.api.types.is_numeric_dtype(col):
        st.write(f"Summary of **{column}** (numeric):")
        st.write(col.describe())

        bins = st.slider("Histogram bins", min_value=5, max_value=100, value=30)
        fig, ax = plt.subplots()
        col.dropna().hist(ax=ax, bins=bins)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        st.pyplot(fig)

        with st.expander("Boxplot"):
            fig2, ax2 = plt.subplots()
            ax2.boxplot(col.dropna().values, vert=True)
            ax2.set_title(f"Boxplot of {column}")
            ax2.set_ylabel(column)
            st.pyplot(fig2)

    else:
        st.write(f"Summary of **{column}** (categorical/text):")
        vc = col.astype("string").fillna("<NA>").value_counts().head(30)
        st.dataframe(vc.to_frame("count"))

        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Top 30 categories of {column}")
        ax.set_ylabel("Count")
        ax.set_xlabel(column)
        st.pyplot(fig)

    # ---------------------------
    # Multi-variable analysis
    # ---------------------------
    st.subheader("📊 Multi-variable Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) >= 2:
        selected_vars = st.multiselect(
            "Select two or more numeric variables to visualize together",
            num_cols,
            default=num_cols[:2]
        )

        chart_type = st.selectbox("Select chart type", ["Line Chart", "Scatter Plot", "Bar Chart"], index=0)

        if len(selected_vars) >= 2:
            fig, ax = plt.subplots(figsize=(10, 5))

            if chart_type == "Line Chart":
                df[selected_vars].plot(ax=ax)
                ax.set_title("Line Chart of Selected Variables")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")

            elif chart_type == "Scatter Plot":
                x_var = st.selectbox("X-axis variable", selected_vars, index=0)
                y_var = st.selectbox("Y-axis variable", selected_vars, index=1)
                ax.scatter(df[x_var], df[y_var], alpha=0.6)
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.set_title(f"Scatter Plot: {x_var} vs {y_var}")

            elif chart_type == "Bar Chart":
                df[selected_vars].plot(kind="bar", ax=ax)
                ax.set_title("Bar Chart of Selected Variables")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")

            st.pyplot(fig)
        else:
            st.info("Please select at least two variables to plot.")
    else:
        st.warning("No numeric columns available for multi-variable analysis.")

    # ---------------------------
    # Numeric vs Categorical comparison
    # ---------------------------
    st.subheader("🏷️ Numeric vs Categorical Comparison")

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if cat_cols and num_cols:
        cat_col = st.selectbox("Select categorical column", cat_cols)
        num_col = st.selectbox("Select numeric column", num_cols)

        agg_func = st.selectbox("Aggregation function", ["mean", "sum", "median", "count"], index=0)

        try:
            grouped = df.groupby(cat_col)[num_col].agg(agg_func).sort_values(ascending=False)
            st.dataframe(grouped.to_frame(f"{agg_func}({num_col})"))

            fig, ax = plt.subplots(figsize=(10, 5))
            grouped.plot(kind="bar", ax=ax)
            ax.set_title(f"{agg_func.capitalize()} of {num_col} by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(f"{agg_func}({num_col})")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Aggregation error: {e}")
    else:
        st.info("Need at least one categorical and one numeric column for this analysis.")

    # ---------------------------
    # Correlation (numeric)
    # ---------------------------
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        st.subheader("🔗 Correlation (numeric)")
        corr = num.corr(numeric_only=True)
        st.dataframe(corr.style.format("{:.2f}"))
        with st.expander("Correlation heatmap (matplotlib)"):
            fig, ax = plt.subplots(figsize=(min(12, 1 + 0.7 * corr.shape[1]), min(8, 1 + 0.5 * corr.shape[0])))
            cax = ax.imshow(corr.values, interpolation="nearest")
            ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.index)
            ax.set_title("Correlation Heatmap")
            fig.colorbar(cax)
            st.pyplot(fig)

    # ---------------------------
    # Quick filters & export
    # ---------------------------
    st.subheader("🧹 Quick Filter / Export")
    with st.expander("Filter rows (pandas query syntax)"):
        st.caption("Example: `price > 100 and region == 'East'`")
        q = st.text_input("Enter query")
        if q:
            try:
                filtered = df.query(q)
                st.write(filtered.head(20))
                st.success(f"Filtered rows: {len(filtered):,}")
                csv = filtered.to_csv(index=False).encode("utf-8")
                st.download_button("Download filtered CSV", csv, file_name="filtered.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Query error: {e}")

    csv_all = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download full data as CSV", csv_all, file_name="data_export.csv", mime="text/csv")

else:
    st.info("Upload a CSV or Excel file to begin.")
