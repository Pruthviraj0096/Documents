# app.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# ======================================================
# OPTIONAL OLLAMA SUPPORT
# ======================================================
OLLAMA_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except:
    pass

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="visualization",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
    }

    .stTextArea textarea {
        background-color: #262730;
        color: white;
    }

    .stFileUploader {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# HEADER
# ======================================================
st.title("Visualization")

st.markdown(
    """
    ### Upload Excel/CSV files and chat with your data

    Ask:
    - Business questions
    - Trends
    - Correlations
    - Insights
    - Visualizations
    - Pivot tables
    - Statistical summaries
    """
)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:

    st.header("⚙️ AI Settings")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    st.markdown("---")

    use_ollama = st.toggle(
        "Use Ollama Backup",
        value=False
    )

    ollama_model = st.text_input(
        "Ollama Model",
        value="llama3"
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.1
    )

    st.markdown("---")

    st.markdown("### 📌 Supported Charts")

    st.markdown(
        """
        - Bar Charts
        - Line Charts
        - Scatter Plots
        - Histograms
        - Heatmaps
        - Correlation Matrix
        - Pivot Tables
        - Pie Charts
        - Area Charts
        - Violin Plots
        - Pairplots
        - Countplots
        - Boxplots
        """
    )

# ======================================================
# LOAD MODEL
# ======================================================
def load_llm():

    # ======================================================
    # OPENAI
    # ======================================================
    if openai_key:

        return ChatOpenAI(
            api_key=openai_key,
            model="gpt-4o-mini",
            temperature=temperature
        )

    # ======================================================
    # OLLAMA
    # ======================================================
    elif use_ollama:

        if not OLLAMA_AVAILABLE:

            st.error(
                "Ollama package not installed."
            )

            st.stop()

        try:

            return Ollama(
                model=ollama_model,
                temperature=temperature
            )

        except Exception:

            st.error(
                "Ollama is not running."
            )

            st.stop()

    # ======================================================
    # NO MODEL
    # ======================================================
    else:

        st.warning(
            "Please provide OpenAI API key OR enable Ollama."
        )

        st.stop()

# ======================================================
# FILE UPLOADER
# ======================================================
uploaded_file = st.file_uploader(
    "📁 Upload File",
    type=["xlsx", "xls", "csv"]
)

# ======================================================
# MAIN APP
# ======================================================
if uploaded_file is not None:

    try:

        # ======================================================
        # LOAD DATA
        # ======================================================
        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            excel_file = pd.ExcelFile(uploaded_file)

            sheet_name = st.selectbox(
                "Select Sheet",
                excel_file.sheet_names
            )

            df = pd.read_excel(
                uploaded_file,
                sheet_name=sheet_name
            )

        # ======================================================
        # SAMPLE LARGE DATASETS
        # ======================================================
        if len(df) > 5000:

            df = df.sample(5000)

            st.warning(
                "Large dataset detected. Sampled 5000 rows."
            )

        st.success("✅ File loaded successfully")

        # ======================================================
        # DATA PREVIEW
        # ======================================================
        st.subheader("📄 Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # ======================================================
        # METRICS
        # ======================================================
        st.subheader("📌 Dataset Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:

            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:

            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        # ======================================================
        # COLUMN INFO
        # ======================================================
        with st.expander("📋 View Columns"):

            st.write(list(df.columns))
            st.write(df.dtypes)

        # ======================================================
        # QUICK STATS
        # ======================================================
        with st.expander("📈 Statistical Summary"):

            try:
                st.dataframe(
                    df.describe(),
                    use_container_width=True
                )
            except:
                st.warning("No numeric columns found.")

        # ======================================================
        # LOAD LLM
        # ======================================================
        llm = load_llm()

        # ======================================================
        # CREATE AGENT
        # ======================================================
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            max_iterations=20,
            max_execution_time=300,
            early_stopping_method="generate"
        )

        # ======================================================
        # CHAT HISTORY
        # ======================================================
        if "messages" not in st.session_state:

            st.session_state.messages = []

        # ======================================================
        # DISPLAY CHAT HISTORY
        # ======================================================
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        # ======================================================
        # CHAT INPUT
        # ======================================================
        user_query = st.chat_input(
            "Ask questions about your data..."
        )

        if user_query:

            # ======================================================
            # SAVE USER MESSAGE
            # ======================================================
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_query
                }
            )

            with st.chat_message("user"):

                st.markdown(user_query)

            # ======================================================
            # ASSISTANT RESPONSE
            # ======================================================
            with st.chat_message("assistant"):

                with st.spinner("🤖 Analyzing data..."):

                    try:

                        lower_query = user_query.lower()

                        # ======================================================
                        # VISUALIZATION KEYWORDS
                        # ======================================================
                        visualization_keywords = [
                            "chart",
                            "graph",
                            "plot",
                            "visualize",
                            "visualization",
                            "heatmap",
                            "matrix",
                            "correlation",
                            "pivot",
                            "table",
                            "bar",
                            "line",
                            "scatter",
                            "histogram",
                            "distribution",
                            "trend",
                            "boxplot",
                            "pie",
                            "area",
                            "violin",
                            "pairplot",
                            "countplot"
                        ]

                        # ======================================================
                        # VISUALIZATION MODE
                        # ======================================================
                        if any(
                            keyword in lower_query
                            for keyword in visualization_keywords
                        ):

                            viz_prompt = f"""
                            You are a data visualization planner.

                            Dataset columns:
                            {list(df.columns)}

                            Column types:
                            {df.dtypes.to_dict()}

                            User request:
                            {user_query}

                            Return ONLY in this format:

                            chart_type:<type>
                            x:<column>
                            y:<column>

                            Allowed chart types:
                            bar
                            line
                            scatter
                            histogram
                            box
                            heatmap
                            matrix
                            pivot
                            pie
                            area
                            violin
                            pairplot
                            countplot
                            """

                            # ======================================================
                            # DIRECT LLM CALL
                            # ======================================================
                            response = llm.invoke(viz_prompt)

                            if hasattr(response, "content"):

                                output = response.content

                            else:

                                output = str(response)

                            # ======================================================
                            # PARSE RESPONSE
                            # ======================================================
                            parsed = {}

                            lines = output.strip().split("\n")

                            for line in lines:

                                if ":" in line:

                                    key, value = line.split(
                                        ":",
                                        1
                                    )

                                    parsed[key.strip()] = value.strip()

                            chart_type = parsed.get(
                                "chart_type",
                                ""
                            ).lower()

                            x_col = parsed.get("x", "")
                            y_col = parsed.get("y", "")

                            valid_columns = list(df.columns)

                            numeric_columns = list(
                                df.select_dtypes(
                                    include="number"
                                ).columns
                            )

                            # ======================================================
                            # VALIDATIONS
                            # ======================================================
                            if (
                                chart_type not in [
                                    "heatmap",
                                    "matrix",
                                    "pairplot"
                                ]
                                and x_col
                                and x_col not in valid_columns
                            ):

                                st.error(
                                    f"Invalid X column: {x_col}"
                                )

                                st.stop()

                            if (
                                chart_type not in [
                                    "histogram",
                                    "heatmap",
                                    "matrix",
                                    "pairplot",
                                    "pie",
                                    "countplot"
                                ]
                                and y_col
                                and y_col not in valid_columns
                            ):

                                st.error(
                                    f"Invalid Y column: {y_col}"
                                )

                                st.stop()

                            sns.set_style("darkgrid")

                            fig, ax = plt.subplots(
                                figsize=(12, 6)
                            )

                            # ======================================================
                            # BAR CHART
                            # ======================================================
                            if chart_type == "bar":

                                sns.barplot(
                                    data=df,
                                    x=x_col,
                                    y=y_col,
                                    palette="viridis",
                                    ax=ax
                                )

                            # ======================================================
                            # LINE CHART
                            # ======================================================
                            elif chart_type == "line":

                                sns.lineplot(
                                    data=df,
                                    x=x_col,
                                    y=y_col,
                                    marker="o",
                                    ax=ax
                                )

                            # ======================================================
                            # SCATTER PLOT
                            # ======================================================
                            elif chart_type == "scatter":

                                sns.scatterplot(
                                    data=df,
                                    x=x_col,
                                    y=y_col,
                                    s=100,
                                    ax=ax
                                )

                            # ======================================================
                            # HISTOGRAM
                            # ======================================================
                            elif chart_type == "histogram":

                                sns.histplot(
                                    df[x_col],
                                    kde=True,
                                    color="skyblue",
                                    ax=ax
                                )

                            # ======================================================
                            # BOXPLOT
                            # ======================================================
                            elif chart_type == "box":

                                sns.boxplot(
                                    data=df,
                                    x=x_col,
                                    y=y_col,
                                    palette="Set2",
                                    ax=ax
                                )

                            # ======================================================
                            # HEATMAP
                            # ======================================================
                            elif chart_type == "heatmap":

                                corr = df.corr(
                                    numeric_only=True
                                )

                                sns.heatmap(
                                    corr,
                                    annot=True,
                                    cmap="coolwarm",
                                    ax=ax
                                )

                            # ======================================================
                            # CORRELATION MATRIX
                            # ======================================================
                            elif chart_type == "matrix":

                                corr = df.corr(
                                    numeric_only=True
                                )

                                sns.heatmap(
                                    corr,
                                    annot=True,
                                    cmap="viridis",
                                    fmt=".2f",
                                    linewidths=0.5,
                                    ax=ax
                                )

                                ax.set_title(
                                    "Correlation Matrix"
                                )

                            # ======================================================
                            # PIVOT TABLE
                            # ======================================================
                            elif chart_type == "pivot":

                                pivot_table = pd.pivot_table(
                                    df,
                                    index=x_col,
                                    values=y_col,
                                    aggfunc="mean"
                                )

                                st.subheader("📊 Pivot Table")

                                st.dataframe(
                                    pivot_table,
                                    use_container_width=True
                                )

                                csv = pivot_table.to_csv().encode(
                                    "utf-8"
                                )

                                st.download_button(
                                    "⬇ Download Pivot CSV",
                                    csv,
                                    "pivot_table.csv",
                                    "text/csv"
                                )

                            # ======================================================
                            # PIE CHART
                            # ======================================================
                            elif chart_type == "pie":

                                pie_data = df[x_col].value_counts().head(10)

                                ax.pie(
                                    pie_data.values,
                                    labels=pie_data.index,
                                    autopct="%1.1f%%"
                                )

                                ax.set_title("Pie Chart")

                            # ======================================================
                            # AREA CHART
                            # ======================================================
                            elif chart_type == "area":

                                df_sorted = df.sort_values(
                                    by=x_col
                                )

                                ax.fill_between(
                                    df_sorted[x_col],
                                    df_sorted[y_col],
                                    alpha=0.5
                                )

                                ax.plot(
                                    df_sorted[x_col],
                                    df_sorted[y_col]
                                )

                                ax.set_title("Area Chart")

                            # ======================================================
                            # VIOLIN PLOT
                            # ======================================================
                            elif chart_type == "violin":

                                sns.violinplot(
                                    data=df,
                                    x=x_col,
                                    y=y_col,
                                    palette="muted",
                                    ax=ax
                                )

                                ax.set_title("Violin Plot")

                            # ======================================================
                            # PAIRPLOT
                            # ======================================================
                            elif chart_type == "pairplot":

                                numeric_df = df.select_dtypes(
                                    include=["number"]
                                )

                                pair_fig = sns.pairplot(
                                    numeric_df
                                )

                                st.pyplot(pair_fig.fig)

                            # ======================================================
                            # COUNTPLOT
                            # ======================================================
                            elif chart_type == "countplot":

                                sns.countplot(
                                    data=df,
                                    x=x_col,
                                    palette="coolwarm",
                                    ax=ax
                                )

                                plt.xticks(rotation=45)

                                ax.set_title("Count Plot")

                            else:

                                st.warning(
                                    "Unsupported chart type."
                                )

                            if chart_type not in [
                                "pivot",
                                "pairplot"
                            ]:

                                plt.xticks(rotation=45)

                                ax.set_title(
                                    f"{chart_type.upper()} Visualization",
                                    fontsize=16,
                                    fontweight="bold"
                                )

                                st.pyplot(fig)

                            # ======================================================
                            # QUICK INSIGHTS
                            # ======================================================
                            st.subheader("📈 Quick Insights")

                            try:

                                st.dataframe(
                                    df.describe(),
                                    use_container_width=True
                                )

                            except:
                                pass

                            assistant_response = (
                                f"✅ Generated {chart_type} visualization"
                            )

                            st.markdown(
                                assistant_response
                            )

                        # ======================================================
                        # QA MODE
                        # ======================================================
                        else:

                            qa_prompt = f"""
                            Answer using the dataframe only.

                            User Question:
                            {user_query}

                            Give concise insights.
                            """

                            response = agent.invoke(
                                qa_prompt
                            )

                            assistant_response = response[
                                "output"
                            ]

                            st.markdown(
                                assistant_response
                            )

                        # ======================================================
                        # SAVE ASSISTANT MESSAGE
                        # ======================================================
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": assistant_response
                            }
                        )

                    except Exception as e:

                        st.error(
                            f"Error: {str(e)}"
                        )

    except Exception as e:

        st.error(
            f"Failed to process file: {str(e)}"
        )

# ======================================================
# EMPTY STATE
# ======================================================
else:

    st.info(
        "📁 Upload an Excel or CSV file to begin."
    )