import os
import streamlit as st
import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Header
    st.header('Classification')

    # File Selector
    def file_selector(folder_path='.\classification_dataset'):
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
        
        preprocessing.feature_reduction(dup_dataset,y,X)
        preprocessing.handling_missing_values(other_features)
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


        # Classification Models
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import cross_val_score

        models = [('Logistic Regression', LogisticRegression()),
                  ('K - Nearest Neighbours', KNeighborsClassifier()),
                  ('Support Vector Machine', SVC()),
                  ('Naive Bayes', GaussianNB()),
                  ('Decision Tree', DecisionTreeClassifier()),
                  ('Random Forest', RandomForestClassifier())]

        acc = []
        cm = []

        for name, model in models:
            classifier = model
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm.append(confusion_matrix(y_test, y_pred))
            accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
            acc.append((name, accuracies.mean()))

        a = []
        b = []

        if st.checkbox('Show Confusion Matrix'):
            available_models = ['Logistic Regression', 'K - Nearest Neighbours', 'Support Vector Machine',
            'Naive Bayes', 'Decision Tree', 'Random Forest']
            selected_model = st.selectbox('Select: ', available_models)
            if selected_model == 'Logistic Regression':
                st.subheader('Logistic Regression')
                st.write(cm[0])
            elif selected_model == 'K - Nearest Neighbours':
                st.subheader('K - Nearest Neighbours')
                st.write(cm[1])
            elif selected_model == 'Support Vector Machine':
                st.subheader('Support Vector Machine')
                st.write(cm[2])
            elif selected_model == 'Naive Bayes':
                st.subheader('Naive Bayes')
                st.write(cm[3])
            elif selected_model == 'Decision Tree':
                st.subheader('Decision Tree')
                st.write(cm[4])
            elif selected_model == 'Random Forest':
                st.subheader('Random Forest')
                st.write(cm[5])

        # Model Accuracies
        st.subheader('Models Analysis')
        for name, acc_predicted in acc:
            st.text('{} : {}'.format(name, acc_predicted))
            a.append(name)
            b.append(acc_predicted)

        data = {'Model' : a, 'Accuracy' : b}
        data = pd.DataFrame(data)

        # Graphs
        import plotly.express as px
        fig = px.bar(data, x='Model', y='Accuracy', color='Accuracy')
        st.plotly_chart(fig)

    


    #Filter Method: Spearman's Cross Corelation > 0.95
    
    #Make Correlation matrix
    # corr_matrix = dup_dataset.corr(method = "spearman").abs()

    # #Select upper triangle of matrix
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    

    # # Find index of feature columns with correlation greater than 0.95
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    

    
    # #Drop features
    # dup_dataset = dup_dataset.drop(to_drop,axis=1)

    # if st.checkbox('Show Preprocessed Datset(Removing highly Corelated Features)'):
    #     st.dataframe(dup_dataset)
    #     st.write(dup_dataset.shape)