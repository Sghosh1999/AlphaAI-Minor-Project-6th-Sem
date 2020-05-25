#Authoe - Dipyaman


import os
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Header
    st.header('Regression')

    # File Selector
    def file_selector(folder_path='.\Regression_dataset'):
        file_list = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a Dataset', file_list)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.warning('Dataset Selected - {}'.format(filename))

    # Reading the Dataset
    dataset = pd.read_csv(filename)

    # Rows and Columns
    rows = dataset.shape[0]
    columns = dataset.shape[1]

    st.text('Row Count: {}\nColumn Count: {}'.format(rows, columns))

    # Show Dataset
    if st.checkbox('Show Dataset'):
        number_of_rows = st.number_input('Number of Rows to View: ', 5, rows)
        st.dataframe(dataset.head(number_of_rows))

    # Show Columns
    if st.checkbox('Show Column Details'):
        frame = {'Columns' : dataset.columns, 'Unique Values' : dataset.nunique().tolist(), 'Missing Values' : dataset.isnull().sum().tolist()}
        result = pd.DataFrame(frame)
        st.dataframe(result)

    X = None
    y = None
    dup_dataset = None

    # Select Dependent Variable
    if st.checkbox('Select Dependent Variable'):
        all_columns = dataset.columns.tolist()
        selected_column = st.selectbox('Select: ', all_columns)
        y = dataset[selected_column]
        X = dataset.drop([selected_column], axis=1)
        dup_dataset = X
        st.subheader('Dependent Variable')
        st.dataframe(y)
    
    # Preprocessing the Dataset
    if dup_dataset is not None:

        if st.checkbox('Show Independent Features'):
            st.subheader('Independent Variable')
            st.dataframe(X)

        cat_features = dup_dataset.select_dtypes(include=['object']).copy()
        other_features = dup_dataset.select_dtypes(exclude=['object']).copy()

        # Removing NaN values from Non-Categorical features
        for col in list(other_features.columns):
            if other_features[col].isnull().sum != 0:
                mean_val = other_features[col].mean()
                other_features[col] = other_features[col].fillna(mean_val)

        # Removing NaN values from Categorical features
        for col in list(cat_features.columns):
            if cat_features[col].isnull().sum() != 0:
                cat_features = cat_features.fillna(cat_features[col].value_counts().index[0])

        # Removing unnecessary Categorical features
        for col in list(cat_features.columns):
            if cat_features[col].nunique() > 10:
                cat_features = cat_features.drop(col, axis=1)

        # Converting the Categorical features to dtype = category
        for col in list(cat_features.columns):
            cat_features[col] = cat_features[col].astype('category')

        # Encoding Categorical features
        for col in list(cat_features.columns):
            temp_feature = pd.get_dummies(cat_features[col], prefix='encoded', drop_first=True)
            cat_features = pd.concat([cat_features, temp_feature], axis=1)
            cat_features = cat_features.drop(col, axis=1)
        
        # Encoded Dataset
        dup_dataset = pd.concat([other_features, cat_features], axis=1)
        if st.checkbox('Show Encoded Features'):
            st.subheader('Encoded Features')
            st.write(dup_dataset)
            st.write(dup_dataset.shape)

        # Splitting the Dataset into Training Set and Test Set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(dup_dataset, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        if st.checkbox('Show Scaled Features'):
            scaled_dataset = ['X_train', 'X_test', 'y_train', 'y_test']
            selected_dataset = st.selectbox('Select: ', scaled_dataset)
            if selected_dataset == 'X_train':
                st.subheader('X_train')
                st.write(X_train)
                st.write(X_train.shape)
            elif selected_dataset == 'X_test':
                st.subheader('X_test')
                st.write(X_test)
                st.write(X_test.shape)
            elif selected_dataset == 'y_train':
                st.subheader('y_train')
                st.write(y_train)
                st.write(y_train.shape)
            elif selected_dataset == 'y_test':
                st.subheader('y_test')
                st.write(y_test)
                st.write(y_test.shape)


        # Regression Models
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        models = [('Linear Regression', LinearRegression()),
                  ('Support Vector Regression', SVR()),
                  ('Decision Tree Regression', DecisionTreeRegressor()),
                  ('Random Forest Regression', RandomForestRegressor())]

        import time
        mae = []
        mse = []
        r2 = []
        time_data = []

        for name, model in models:
            regressor = model
            start_time = time.time()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            end_time = time.time()
            time_data.append(end_time-start_time)
            mae.append((name, mean_absolute_error(y_test, y_pred)))
            mse.append((name, mean_squared_error(y_test, y_pred)))
            r2.append((name, r2_score(y_test, y_pred)))

        a_mae = []
        b_mae = []
        a_mse = []
        b_mse = []
        a_r2 = []
        b_r2 = []

        # Model Accuracies
        st.subheader('Mean Absolute Error')
        for name, acc_predicted in mae:
            st.text('{} : MAE = {}'.format(name, acc_predicted))
            a_mae.append(name)
            b_mae.append(acc_predicted)

        st.subheader('Mean Squared Error')
        for name, acc_predicted in mse:
            st.text('{} : MAE = {}'.format(name, acc_predicted))
            a_mse.append(name)
            b_mse.append(acc_predicted)

        st.subheader('R2 Score')
        for name, acc_predicted in r2:
            st.text('{} : R2 = {}'.format(name, acc_predicted))
            a_r2.append(name)
            b_r2.append(acc_predicted)

        data_mae = {'Model' : a_mae, 'MAE' : b_mae}
        data_mae = pd.DataFrame(data_mae)

        data_mse = {'Model': a_mse, 'MSE': b_mse}
        data_mse = pd.DataFrame(data_mse)

        data_r2 = {'Model': a_r2, 'R2': b_r2}
        data_r2 = pd.DataFrame(data_r2)

        # Graphs
        import plotly.express as px
        fig = px.bar(data_mae, x='Model', y='MAE', color='MAE')
        st.plotly_chart(fig)

        import plotly.express as px
        fig = px.bar(data_mse, x='Model', y='MSE', color='MSE')
        st.plotly_chart(fig)

        

    

