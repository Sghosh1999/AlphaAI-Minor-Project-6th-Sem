import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


def main():

    def file_selector(folder_path='./dataset'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select A File", filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()  # Fetching the Dataset
    st.info("You Selected {}".format(filename))

    # Read Dataset
    df = pd.read_csv(filename)

    # Show Dataset

    if st.checkbox("Show Dataset"):
        number = st.number_input("Rows", 5, 100)
        st.dataframe(df.head(number))

        # Show Columns
    if st.button("Column Names"):
        st.write(df.columns)

    # Show Shape
    if st.checkbox("Shape Of dataset"):
        st.write(df.shape)


    # Select Columns and storing in list and making a new dataset
    if st.checkbox("Select Columns to show"):
        all_cols = df.columns.tolist()
        Selected_cols = st.multiselect("Select", all_cols)
        new_df = df[Selected_cols]
        st.dataframe(new_df)

    # Selecting the Target Column
    if st.checkbox("Select Target Column"):
        all_cols = df.columns.tolist()
        Selected_cols = st.multiselect("Select", all_cols, key='target')
        # tar_idx = df.columns.get_loc(Selected_cols)
        new_df = df[Selected_cols]
        st.dataframe(new_df)

    # Show DataTypes
    if st.checkbox("Data Types"):
        st.text("Data Types")
        st.write(df.dtypes)

    # Dataset Summary
    if st.checkbox("Dataset Summary"):
        st.write(df.describe().T)

    # Plot & Visualization

    st.subheader("Data Visualizations")
    # Corelation Plot
    # Seaborn Plot
    # Count Plot
    # Pie Chart

    #Scatter Plot Distribution
    if st.checkbox("Select Features  to explore Relation using Pair Plot"):
        all_cols = df.columns.tolist()
        Selected_cols = st.multiselect("Select", all_cols, key='col_corelation')
        fig = px.scatter_matrix(df, dimensions=Selected_cols,color = all_cols[1])
        st.plotly_chart(fig)

    if st.checkbox("Pie Chart"):
        all_columns_names = df.columns.tolist()
        if st.button("Genearte Plot", key="pie"):
            st.success("Generating Pie Plot")
            st.write(df.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # Customizable Plot

    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
    selected_columns = st.multiselect("Select Columns to Plot", all_columns_names)

   
    all_cols = df.columns.tolist()
    Selected_cols = st.multiselect("Select", all_cols, key='col_corelation1')
    fig = px.scatter_matrix(df, dimensions=Selected_cols,color = all_cols[1])
    st.plotly_chart(fig)


    if st.button("Genearte Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns))

        # Plot by Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_columns]
            st.area_chart(cust_data)

        elif type_of_plot == 'bar':
            cust_data = df[selected_columns]
            st.bar_chart(cust_data)

        elif type_of_plot == 'line':
            cust_data = df[selected_columns]
            st.line_chart(cust_data)


        # Custom Plot
        elif type_of_plot == 'box':
            cust_plot = df[selected_columns].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

        elif type_of_plot == 'hist':
            cust_plot = df[selected_columns].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

        elif type_of_plot == 'kde':
            cust_plot = df[selected_columns].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()