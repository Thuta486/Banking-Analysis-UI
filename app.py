import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)

st.set_page_config(page_title="Bank Term Deposit Subscription", layout="wide")

# ----------------------------------------------------------------------
# Load trained artifacts (produced by the notebook - see README)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("artifacts.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("bank_cleaned.csv")


artifacts = load_artifacts()
df = load_data()

pipelines_a = artifacts["pipelines_a"]
X_test_a = artifacts["X_test_a"]
y_test_a = artifacts["y_test_a"]
best_model_name = artifacts["best_model_name"]
FEATURES_REALISTIC = artifacts["features_realistic"]
numeric_features = artifacts["numeric_features"]
categorical_features = artifacts["categorical_features"]
cluster_scaler = artifacts["cluster_scaler"]
cluster_kmeans = artifacts["cluster_kmeans"]
cluster_numeric = artifacts["cluster_numeric"]
cluster_categorical = artifacts["cluster_categorical"]
cluster_encoded_columns = artifacts["cluster_encoded_columns"]
job_group_map = artifacts["job_group_map"]
cluster_profile = artifacts["cluster_profile"]
results_a = artifacts["results_a"]
results_b = artifacts["results_b"]
cv_results = artifacts["cv_results"]
findings = artifacts["findings"]
duration_effect = artifacts["duration_effect"]
# rules_to_no / rules_to_yes replace the old single "top_rules" table - falls back
# to top_rules if the notebook bundle hasn't been re-exported yet.
rules_to_no = artifacts.get("rules_to_no", artifacts.get("top_rules"))
rules_to_yes = artifacts.get("rules_to_yes")
cluster_subscription = artifacts.get("cluster_subscription")
cluster_nature = artifacts.get("cluster_nature", {})
k_range = artifacts.get("k_range")
inertias = artifacts.get("inertias")
silhouette_scores = artifacts.get("silhouette_scores")
suggested_k = artifacts.get("suggested_k")

with st.sidebar:
    st.header("IS-212 Project")
    st.caption("Thuta Kyaw Lynn - YKPT-22387")
    st.markdown("**Customer Subscription Prediction**\n\nBank Marketing dataset - descriptive & predictive mining")

st.title("Customer Subscription Prediction - Banking Marketing Campaigns")
st.caption(
    "IS-212 Data and Knowledge Mining project. Model, clusters, and charts below are "
    "the exact artifacts trained in the accompanying notebook - nothing here is retrained."
)
st.info(
    "**Project Goal:** Help banks identify customers who are more likely to subscribe "
    "to a term deposit, allowing them to prioritize calls and reduce wasted contact effort."
)

tabs = st.tabs([
    "Overview", "Data Preparation", "Descriptive Mining",
    "Predictive Modeling & Evaluation", "Predict a Customer",
])

# ----------------------------------------------------------------------
# TAB 1 - OVERVIEW
# ----------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Subscription rate", f"{df['y'].mean() * 100:.1f}%")

    fig = px.histogram(
        df, x="y", color="y", category_orders={"y": [0, 1]},
        labels={"y": "Subscribed"}, title="Target Distribution",
    )
    fig.update_xaxes(tickvals=[0, 1], ticktext=["No", "Yes"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric column statistics")
    st.dataframe(df[numeric_features].describe().T, use_container_width=True)

    st.subheader("Categorical column statistics")
    st.dataframe(df[categorical_features].describe().T, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2 - DATA PREPARATION
# ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Unknown-value share by column")
    unk = (df[categorical_features] == "unknown").mean().mul(100)
    unk = unk[unk > 0].sort_values(ascending=False)
    fig = px.bar(unk, labels={"value": "% unknown", "index": "Column"},
                 title="Percentage of 'unknown' values by column")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "`poutcome` is unknown mostly because the client was never contacted in a "
        "previous campaign - not random missingness."
    )

    st.divider()
    st.subheader("Customer Profiling")
    
    profile_cols = ["job", "education", "marital", "housing", "loan",
                     "contact", "poutcome", "age_group", "campaign_group"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    chosen_col = st.selectbox("Group subscription rate by:", profile_cols)

    rate = df.groupby(chosen_col, observed=True)["y"].mean().mul(100).sort_values(ascending=False)
    fig = px.bar(rate, labels={"value": "Subscription rate (%)", "index": chosen_col},
                 title=f"Subscription Rate by {chosen_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Data Visualization")
    v1, v2 = st.columns(2)
    v1.plotly_chart(px.histogram(df, x="job", title="Job Distribution").update_xaxes(categoryorder="total descending"),
                     use_container_width=True)
    v2.plotly_chart(px.histogram(df, x="education", title="Education Level").update_xaxes(categoryorder="total descending"),
                     use_container_width=True)
    v3, v4 = st.columns(2)
    v3.plotly_chart(px.histogram(df, x="age", title="Age Distribution", nbins=30), use_container_width=True)
    corr = df[numeric_features].corr()
    v4.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                               title="Correlation Heatmap"), use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 - DESCRIPTIVE MINING
# ----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Customer Segmentation (K-Means)")
    st.caption(
        "Segmentation and association rules below are the actual descriptive "
        "mining techniques used in this project - both discover structure directly from the data "
        "instead of grouping by a single known column."
    )

    if k_range and inertias and silhouette_scores:
        e1, e2 = st.columns(2)
        fig = px.line(x=k_range, y=inertias, markers=True,
                       labels={"x": "k", "y": "Inertia"}, title="Elbow Method")
        e1.plotly_chart(fig, use_container_width=True)
        fig = px.line(x=k_range, y=silhouette_scores, markers=True,
                       labels={"x": "k", "y": "Silhouette score"}, title="Silhouette Score by k")
        e2.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"k={suggested_k} chosen: highest silhouette score ({max(silhouette_scores):.3f}) "
            "and the smallest k within 0.01 of the best score, keeping clusters interpretable. "
            "`job` (12 categories) was grouped into 4 broader categories before clustering to avoid "
            "sparse one-hot columns destabilizing the distance metric."
        )

    st.dataframe(cluster_profile, use_container_width=True)

    cluster_rate = (
        cluster_subscription if cluster_subscription is not None
        else df.groupby("cluster")["y"].mean().mul(100)
    )
    fig = px.bar(cluster_rate, labels={"value": "Subscription rate (%)", "index": "Cluster"},
                 title="Subscription Rate by Cluster")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df, x="pca1", y="pca2", color=df["cluster"].astype(str),
        opacity=0.4, title="Customer Clusters (PCA 2D Projection)",
        labels={"color": "Cluster"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if cluster_nature:
        st.markdown("**What each cluster means**")
        st.caption(
            "Each cluster has a distinct nature, not just a different subscription number - "
            "based on prior contact history and the macroeconomic climate at the time of contact."
        )
        for cluster_id in sorted(cluster_nature.keys()):
            info = cluster_nature[cluster_id]
            rate_txt = ""
            if cluster_rate is not None and cluster_id in cluster_rate.index:
                rate_txt = f" ({cluster_rate.loc[cluster_id]:.2f}% subscription)"
            with st.expander(f"Cluster {cluster_id} - {info['label']}{rate_txt}"):
                st.write(info["desc"])

    st.divider()
    st.subheader("Association Rules")
    st.caption(
        "Goal: find out which combinations of customer/campaign attributes co-occur with a particular "
        "subscription outcome - which combinations are strongly linked to customers who do **not** "
        "subscribe (y=no), and which are linked to customers who **do** subscribe (y=yes). This is more "
        "directly useful for a marketing team than the raw statistics in Data Preparation alone, since "
        "it points to concrete profiles worth targeting or deprioritizing."
    )

    def _format_rules(rules_df, top_n):
        display = rules_df.sort_values("lift", ascending=False).head(top_n).copy()
        display["antecedents"] = display["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        display["consequents"] = display["consequents"].apply(lambda s: ", ".join(sorted(s)))
        return display[["antecedents", "consequents", "support", "confidence", "lift"]]

    top_n_rules = st.slider("Number of top rules to show (ranked by lift)", 5, 25, 10)

    st.markdown("**Rules pointing to y=no (not subscribing)**")
    st.caption(f"Showing the top {top_n_rules} of {len(rules_to_no):,} rules found, ranked by lift.")
    st.dataframe(_format_rules(rules_to_no, top_n_rules), use_container_width=True)
    st.caption(
        "The strongest patterns all point to the same group: technicians with a professional-course "
        "education, no personal loan, and no prior campaign contact are consistently linked to not "
        "subscribing (lift around 4.05-4.08) - about four times more common among non-subscribers than "
        "by chance. A useful negative pattern for deprioritizing unlikely-to-convert profiles."
    )

    if rules_to_yes is not None and len(rules_to_yes) > 0:
        st.markdown("**Rules pointing to y=yes (subscribing)**")
        st.caption(f"Showing the top {min(top_n_rules, len(rules_to_yes))} of {len(rules_to_yes):,} rules found, ranked by lift.")
        st.dataframe(_format_rules(rules_to_yes, top_n_rules), use_container_width=True)
        st.caption(
            "The strongest positive pattern: customers whose previous campaign outcome was a success "
            "(poutcome=success) are strongly linked to subscribing again, especially when contacted by "
            "cellular phone (lift up to 6.45) - about six times more common among subscribers than by "
            "chance. This is the clearest actionable pattern found: past success is the best predictor "
            "of future success."
        )
    else:
        st.info(
            "No rules pointing to y=yes passed the support/confidence/lift thresholds used here - "
            "on its own, that is a finding worth reporting, since it would mean no single categorical "
            "combination alone predicts subscription strongly."
        )

# ----------------------------------------------------------------------
# TAB 4 - PREDICTIVE MODELING & EVALUATION
# ----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Model Comparison")
    st.info(
    "**Data Leakage Check:** `duration` is excluded from the realistic model because "
    "call duration is only known after the call. Experiment B includes it for comparison "
    "to show how data leakage can artificially improve model performance."
)
    colA, colB = st.columns(2)
    colA.markdown("**Experiment A - without `duration`**")
    colA.dataframe(results_a, use_container_width=True)
    colB.markdown("**Experiment B - with `duration`**")
    colB.dataframe(results_b, use_container_width=True)

    st.markdown("**Effect of including `duration`**")
    st.dataframe(duration_effect, use_container_width=True)

    fig = px.bar(results_a, barmode="group", title="Model Comparison - Experiment A (no duration)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-Validation (5-fold, Experiment A)")
    st.dataframe(cv_results, use_container_width=True)

    st.caption(f"**Best model: {best_model_name}** - selected by highest ROC-AUC on Experiment A (realistic, no duration).")

    st.subheader("Findings")
    st.dataframe(findings, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Interactive Model Inspection")
    model_names = [m for m in pipelines_a if m != "Baseline (Majority)"]
    selected_models = st.multiselect("Models to plot on the ROC curve:", model_names, default=[best_model_name])

    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    for name in selected_models:
        proba = pipelines_a[name].predict_proba(X_test_a)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_a, proba)
        auc = roc_auc_score(y_test_a, proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})", mode="lines"))
    fig.update_layout(title="ROC Curve(s)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Threshold analysis** - drag the slider to see precision/recall/F1 and the confusion matrix update live.")
    inspect_model = st.selectbox("Model:", model_names, index=model_names.index(best_model_name))
    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05)

    proba = pipelines_a[inspect_model].predict_proba(X_test_a)[:, 1]
    pred_t = (proba >= threshold).astype(int)

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{precision_score(y_test_a, pred_t, zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(y_test_a, pred_t):.3f}")
    m3.metric("F1-score", f"{f1_score(y_test_a, pred_t):.3f}")

    cm = confusion_matrix(y_test_a, pred_t)
    fig = px.imshow(
        cm, text_auto=True, x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"],
        color_continuous_scale="Blues", title=f"Confusion Matrix - {inspect_model} @ threshold {threshold:.2f}",
    )
    st.plotly_chart(fig, use_container_width=True)

    if inspect_model in ("Random Forest", "Extra Trees", "Decision Tree"):
        st.markdown("**Feature importance**")
        pipe = pipelines_a[inspect_model]
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
        importances = importances.sort_values(ascending=False).head(12)
        fig = px.bar(importances[::-1], orientation="h", title=f"Top Feature Importances - {inspect_model}")
        st.plotly_chart(fig, use_container_width=True)
    elif inspect_model == "Logistic Regression":
        st.markdown("**Logistic regression coefficients**")
        pipe = pipelines_a[inspect_model]
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        coefs = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
        top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).head(12).index)
        fig = px.bar(top_coefs.sort_values(), orientation="h", title="Top Logistic Regression Coefficients")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 5 - PREDICT A CUSTOMER
# ----------------------------------------------------------------------
with tabs[4]:
    st.subheader("Enter customer and campaign details")

    def options(col):
        return sorted(df[col].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 95, 40)
        job = st.selectbox("Job", options("job"))
        marital = st.selectbox("Marital status", options("marital"))
        education = st.selectbox("Education", options("education"))
        default = st.selectbox("Credit in default?", options("default"))
        housing = st.selectbox("Housing loan", options("housing"))
    with c2:
        loan = st.selectbox("Personal loan", options("loan"))
        contact = st.selectbox("Contact method", options("contact"))
        month = st.selectbox("Last contact month", options("month"))
        day_of_week = st.selectbox("Last contact day of week", options("day_of_week"))
        campaign = st.slider("Campaign contacts (this campaign)", 1, 20, 1)
        pdays = st.slider("Days since last contact (999 = never)", -1, 999, 999)
    with c3:
        previous = st.slider("Previous contacts", 0, 10, 0)
        poutcome = st.selectbox("Previous campaign outcome", options("poutcome"))
        emp_var_rate = st.number_input("Employment variation rate", value=float(df["emp.var.rate"].median()))
        cons_price_idx = st.number_input("Consumer price index", value=float(df["cons.price.idx"].median()))
        cons_conf_idx = st.number_input("Consumer confidence index", value=float(df["cons.conf.idx"].median()))
        euribor3m = st.number_input("Euribor 3-month rate", value=float(df["euribor3m"].median()))
    nr_employed = st.number_input("Number of employees", value=float(df["nr.employed"].median()))

    if st.button("Predict", type="primary"):
        was_contacted = 0 if pdays == 999 else 1
        row = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day_of_week, "campaign": campaign,
            "pdays": pdays, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "nr.employed": nr_employed, "was_previously_contacted": was_contacted,
        }])[FEATURES_REALISTIC]

        final_pipeline = pipelines_a[best_model_name]
        pred = final_pipeline.predict(row)[0]
        proba = final_pipeline.predict_proba(row)[0, 1]

        r1, r2 = st.columns(2)
        r1.metric("Predicted subscription", "Yes" if pred == 1 else "No")
        r2.metric("Probability of subscription", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        # Automatically assign this hypothetical customer to a cluster.
        job_group = job_group_map.get(job, "unknown")
        cluster_row = pd.DataFrame([{
            "age": age, "campaign": campaign, "previous": previous,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "job_group": job_group, "marital": marital, "education": education,
            "housing": housing, "loan": loan, "was_previously_contacted": was_contacted,
        }])
        cluster_encoded_row = pd.get_dummies(cluster_row, columns=cluster_categorical)
        cluster_encoded_row = cluster_encoded_row.reindex(columns=cluster_encoded_columns, fill_value=0)
        cluster_scaled_row = cluster_scaler.transform(cluster_encoded_row)
        assigned_cluster = int(cluster_kmeans.predict(cluster_scaled_row)[0])

        st.markdown(f"**Automatically assigned segment: Cluster {assigned_cluster}**")
        st.dataframe(cluster_profile.loc[[assigned_cluster]], use_container_width=True)