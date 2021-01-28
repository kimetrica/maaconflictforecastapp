#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan 25 17:46:54 2021
@author: yaredhurisa """

import datetime
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from sklearn import base
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score,
    classification_report,
    auc,
    roc_curve,
    confusion_matrix,
)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import base64
import latex


url = "https://data.kimetrica.com/dataset/8c728bc7-7390-44c1-a99c-83c08b216d03/resource/262d427c-883a-4c8b-80e3-8fca5b3f97c5/download/myn_final_data_binary.csv"
df = pd.read_csv(url, index_col=0)


@st.cache
def load_data(df):

    return (
        df,
        df.shape[0],
        df.shape[1],
    )


data = df
rows = data.shape[0]
columns = data.shape[1]
data = data[
    [
        "admin1",
        "admin2",
        "month_year",
        "drought_index",
        "mean_rainfall",
        "pulses_price",
        "rice_price",
        "longitude",
        "latitude",
        "mining_area_log",
        "pop_density",
        "urban_pop",
        "lc",
        "youth_bulge",
        "years_schooling",
        "poverty",
        "tv",
        "stunting",
        "gender_index",
        "wasting",
        "road_density",
        "ethnicty_count",
        "actor_gf",
        "cc_frequency",
        "actor_rf",
        "cc_onset_x",
        "cellphone",
        "battles",
        "electricity",
        "infant_mortality",
        "patrilocal_index",
        "m_rebels",
        "remote_violence",
        "actor_c",
        "fatalities",
        "fatalities_per_event",
        "s_protesters",
        "protests",
        "violence",
        "actor_p",
        "m_civilians",
        "actor_pm",
        "sd",
        "pm_civilians",
        "r_civilians",
        "s_military",
        "m_p_militias",
        "r_rebels",
        "s_p_militias",
        "actor_r",
        "riots",
        "m_protesters",
        "cc_onset_y",
    ]
]

end_date = "2019-11"
mask = data["month_year"] < end_date
train1 = data.loc[mask]


start_date = "2019-10"
end_date = "2020-11"
mask = (data["month_year"] > start_date) & (data["month_year"] < end_date)
test1 = data.loc[mask]


end_date = "2020-11"
mask = data["month_year"] < end_date
re_train1 = data.loc[mask]


start_date = "2020-10"
end_date = "2021-11"
mask = (data["month_year"] > start_date) & (data["month_year"] < end_date)
current = data.loc[mask]

train = train1.drop(["admin1", "admin2", "month_year"], axis=1)
re_train = re_train1.drop(["admin1", "admin2", "month_year"], axis=1)
test = test1.drop(["admin1", "admin2", "month_year"], axis=1)
current1 = current.drop(["admin1", "admin2", "month_year"], axis=1)

X_train = train[train.columns[:-1]]
X_test = test[test.columns[:-1]]
X_re_train = re_train[train.columns[:-1]]
y_train = train.cc_onset_y
y_test = test.cc_onset_y
y_re_train = re_train.cc_onset_y

X_current = current1[current1.columns[:-1]]
y_current = current1.cc_onset_y
X_current.to_csv("new_data.csv")


def home_page_builder(df, data, rows, columns):
    st.title("Kimetrica Conflict Forecasting Model: Myanmar Analytical Activity (MAA)")
    st.write("")
    st.write("")
    st.subheader("INTRODUCTION")
    st.write("")
    st.write(
        "An early-warning system that can meaningfully forecast conflict in its various forms is necessary to respond to crises ahead of time. The ability to predict where and when conflict is more likely to occur will have a significant impact on reducing the devastating consequences of conflict. The goal of this conflict model is to forecast armed conflict over time and space in Myanmar at the second administrative level and on a monthly basis. This document will outline the model construction methodology and the model output.")
    st.write("")
    st.write("Most predictive models for conflict use country-level data in yearly time increments (Aas Rustad et al., 2011). One problem with this type of analysis is that it assumes that conflict is distributed uniformly throughout the country and uniformly throughout the year. This situation is rarely the case as conflict usually takes place on the borders of countries. For a model to be maximally useful, it must predict where in the country the conflict is likely to occur. Likewise, for a model to be useful for decision-makers, it must be able to predict when the conflict will occur (Brandt et al., 2011).")
    st.write("")
    st.write("To satisfy the requirements of the MAA project, we have built a model to predict conflict at the county (admin2) level at monthly time intervals one year into the future. This application presents the steps taken to build the model, visualize the data and result , run the model and model performance. ")
    st.write("")
    st.write("")
    st.subheader("INSTRUCTION")
    st.write("")
    st.write(
        "This website runs the conflict model and the associated pages that are useful for the users to understand the model outputs. The navigation buttons are provided in the drop down list under the main menu. The Home button represents the current page. You can navigate between pages by clicking a list of buttons including the page to run the model."
    )
    st.write("")
    st.write("")


def model_description_page_builder(df, data, rows, columns):
    st.title("Kimetrica Conflict Forecasting Model: Myanmar Analytical Activity (MAA)")
    st.write("")
    st.write("")
    st.subheader("INTRODUCTION")
    st.write("")
    st.write("The conflict data has two distinct features that require special care compared to conventional machine learning problems. These are class imbalance and recurrence.")
    st.write("")
    st.subheader("Class imbalance")
    st.write("")
    st.write("In reality, conflict occurs in a rare situation resulting in a significant class imbalance in the output data between conflict and non-conflict events. As can be seen from the following chart, overall, the percent of positive records for conflict ranges between 20 and 40 percent for most of the years. This requires a mechanism that can take into account for the less number of positive(conflict) records in the dataset.")
    st.write("")
    if st.checkbox("Show class imbalance"):
        source = df.groupby(["year", "cc_onset_y"])[
            "admin1"].count().reset_index()

        c_onset_chart = (
            alt.Chart(source, title="Number of conflict records by year")
            .mark_bar(size=20)
            .encode(
                alt.X("year:O", title="year"),
                alt.Y("admin1", title="percent of records"),
                alt.Color("cc_onset_y:O", legend=alt.Legend(
                    title="conflict Status")),
            )
            .properties(width=500)
        )
        st.altair_chart(c_onset_chart)
    st.write("")
    st.subheader("Recurrance")
    st.write("")
    st.write("The second aspect of the conflict event dataset is that, once conflict occurs, it has a tendency to last for an extended number of months and years. As such, the model needs to have the capacity to trace recurrence. CFM handles this issue by incorporating a threshold of probability of confidence in claiming the events. In this case, the model takes the current situation if the confidence level drops less than the average mean difference.")
    st.write("")
    st.subheader("EasyEnsemble classifier")
    st.write("")
    st.write("Undersampling is among the popular methods of handling class-imbalance. This method entails taking a subset of the major class to train the classifier. However, this method has a main deficiency as it ignores portions of the dataset in an attempt to balance the number of positive records.")
    st.write("")
    st.write("Xu-Ying, Jianxin, and Zhi-Hua (2080), proposed EasyEnsemble classifier to overcome the above problem of under sampling. EasyEnsemble forecast samples several subsets from the majority class and combines for a final decision. These independent samples ultimately take into account the different aspects of the entire dataset.")
    st.write("")
    st.subheader("Model Specification")
    st.write("")
    st.write("st.latex(Symbol'({x^1})^a')")
    st.write("")


def data_vis_page_builder(df, data, rows, columns):
    st.title("Data Visualization")
    st.write("")
    st.write("")
    st.subheader("INTRODUCTION")
    st.write("")
    st.write(
        "This page presents the exploratory analysis result of the target and input features used in the model. Check/uncheck the following list of tick boxes to view the result."
    )
    st.write("")
    st.write("")

    # show data visulization

    if st.checkbox("Number of conflict records"):
        source = df.groupby(["year", "cc_onset_y"])[
            "admin1"].count().reset_index()

        c_onset_chart = (
            alt.Chart(source, title="Number of conflict records by year")
            .mark_bar(size=20)
            .encode(
                alt.X("year:O", title="year"),
                alt.Y("admin1", title="percent of records"),
                alt.Color("cc_onset_y:O", legend=alt.Legend(
                    title="conflict Status")),
            )
            .properties(width=500)
        )
        st.altair_chart(c_onset_chart)


def logistic_train_metrics(df):
    """Return metrics and model for Logistic Regression."""

    # Fit model
    model_reg = Pipeline(
        [
            ("StandardScaller", StandardScaler()),
            (
                "RF",
                EasyEnsembleClassifier(
                    n_estimators=15,
                    n_jobs=84,
                    base_estimator=XGBClassifier(
                        max_delta_step=1,
                        base_estimator__gama=0.016,
                        base_estimator__alpha=100,
                        base_estimator__max_delta_step=1,
                        base_score=0.23,
                        max_depth=52,
                    ),
                    sampling_strategy="auto",
                ),
            ),
        ]
    )

    model_reg.fit(X_train, y_train)

    # Make predictions for test data

    # Evaluate predictions

    return model_reg


def logistic_page_builder(data):
    start_time = datetime.datetime.now()
    model_reg = logistic_train_metrics(data)
    y_pred = model_reg.predict_proba(X_test)

    prob_diff = []
    for prob in y_pred:
        prob_diff.append(abs(prob[1] - prob[0]))

    prob_pred = []
    for prob, t in zip(y_pred, y_test):
        if abs(prob[1] - prob[0]) > [pd.DataFrame(prob_diff).mean()]:
            prob_pred.append(np.argmax(prob))
        else:
            prob_pred.append(int(t))
    y_pred = prob_pred

    conf_ee = confusion_matrix(y_test, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in conf_ee.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in conf_ee.flatten() / np.sum(conf_ee)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    fig, ax = plt.subplots()
    ax = plt.axes()
    st.write(
        sns.heatmap(
            conf_ee,
            annot=labels,
            fmt="",
            cmap="Blues",
            xticklabels=["No Conflict", "Conflict"],
            yticklabels=["No Conflict", "Conflict"],
            ax=ax,
        )
    )
    ax.set_title("Final Model Error Matrix")
    st.subheader("EASYENSEMBLE CLASSIFIER")
    st.write("")
    st.subheader("Introduction")
    st.write(
        "The conflict data has to destinict features that requires special care compared to conventional machine learning problems. These are class imbalance and recurrence"
    )
    st.write("xx")

    st.write("Class imbalance")
    st.write("In reality, conflict occurs in a rare like situation resulting in  a significant class imbalance in the output data between  conflict and non-conflict events. As can be seen from the  following chart, overall, the percent of postive records for conflict is less than 10 percent for most of the years except for the last five years. This requires a mechanism that can take into account for the less number of postive(conflict) records in the dataset.")
    st.write("xx")

    st.write("")
    st.write(
        "Logistic Regression is a very popular Linear classification model, people usually use it as a baseline model and build the decision boundary."
    )
    st.write("See more from Wiki: https://en.wikipedia.org/wiki/Logistic_regression")
    st.write("")
    st.subheader("Logistic Regression metrics on testing dataset")
    st.write("")
    st.write("")
    st.write("")
    st.markdown(
        "We separated the dataset to training and testing dataset, using training data to train our model then do the prediction on testing dataset, here's Logistic Regression prediction performance: "
    )
    st.write("")
    st.write(
        f"Running time: {(datetime.datetime.now() - start_time).seconds} s")

    st.pyplot(fig)

    return model_reg


columns = X_train.shape[1]


def new_data_downloader(data):

    st.write("")
    st.subheader("Want to preview the raw data used for the model?")
    if st.checkbox("Raw Data"):

        st.write(
            f"Input dataset includes **{df.shape[0]}** rows and **{df.shape[1]}** columns")
        st.write(df.head())
    st.write("")
    st.subheader("Want to preview the processed data used for the model?")
    if st.checkbox("Processed data"):

        st.write(
            f"After Pre-processing the data for modeling, dataset includes **{data.shape[0]}** rows and **{data.shape[1]}** columns"
        )
        st.write(data.head())

    st.write("")
    st.subheader("Want to download the raw data used for the model?")
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

    st.write("")
    st.subheader("Want to download the new dataset to perform forecasting?")
    csv = X_current.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)


def logistic_predictor(model_reg, columns, X_train):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.text("This process probably takes few seconds...")
    st.write(
        "Note: Currently, the CSV file should have **exactly the same** format with **training dataset**:",
        X_test.head(2),
    )
    st.write(f"Training dataset includes **{columns}** columns")
    st.write("")

    st.write("")

    if uploaded_file:
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.write("-" * 80)
        st.write("Uploaded data:", data.head(30))
        st.write(
            f"Uploaded data includes **{data.shape[0]}** rows and **{data.shape[1]}** columns"
        )

        start_time = datetime.datetime.now()
        data = data.dropna(axis=0, how="all")

        X = data

        prediction = model_reg.predict(X)
        prediction_time = (datetime.datetime.now() - start_time).seconds
        data["conflict_forecast"] = [
            "No conflict" if i == 0 else "Conflict" for i in prediction
        ]
        st.write("")
        st.write("-" * 80)
        st.write("Prediction:")
        st.write(data.head(30))
        st.text(f"Running time: {prediction_time} s")
        st.write("")

        st.write("Metrics on uploaded data:")
        st.text("Note: This is only temporary since new data won't have labels")

        # Download the prediction as a CSV file
        prediction_downloader(data)


def prediction_downloader(data):
    st.write("")
    st.subheader("Want to download the prediction results?")
    csv = data.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)


def main():
    """Application of Kimetrica Conflict Forecasting Model: Myanmar Analytical Activity (MAA)"""

    st.sidebar.title("Menu")
    choose_model = st.sidebar.selectbox(
        "Choose the page or model", [
            "Home", "Model description", "Manage data", "Visualize data", "Run Conflict Model", "Visualize model output"]
    )

    # Home page building
    if choose_model == "Home":
        home_page_builder(df, data, rows, columns)
        # Home page building
    if choose_model == "Model description":
        model_description_page_builder(df, data, rows, columns)

        # Home page building
    if choose_model == "Manage data":
        new_data_downloader(X_current)
        # Home page building
    if choose_model == "Visualize data":
        data_vis_page_builder(df, data, rows, columns)

    # Page for Logistic Regression
    if choose_model == "Run Conflict Model":
        model_reg = logistic_page_builder(data)

        if st.checkbox("Want to Use this model to forecast using a new dataset?"):
            logistic_predictor(model_reg, columns, df)
        # Home page building
    if choose_model == "Visualize model output":
        model_vis_page_builder(df, data, rows, columns)


if __name__ == "__main__":
    main()
