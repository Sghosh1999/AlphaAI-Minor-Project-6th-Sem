import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from datetime import date, timedelta
pio.templates.default = "plotly_dark"


def main():
    #Title
    st.title("InDepth Analysis of Covid19 in India")
    #Description of the Coronavirus
    st.text("The Web App is based on Covid19 analysis , developed by us for making more awareness\namong people and to provide in depth analytics of Covid19 to people. It includes\nfuture prediction.prevention and cure methodologies and a symptoms analyzer for common\npeopleto ease the understandibility of the Covid19 pandemic.")
    image = Image.open('images/coronavirus.jpg')
    st.image(image, use_column_width=True, caption='Covid19')
    
    complete_data = pd.read_csv('./covid19_dataset/covid_19_clean_complete.csv',parse_dates=['Date'])

    if st.checkbox("Dataset Overview"):
        number = st.number_input("Rows", 5, 100)
        st.dataframe(complete_data.head(number))

    # cases 
    cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

    # Active Case = (confirmed - deaths - recovered)
    complete_data['Active'] = complete_data['Confirmed'] - complete_data['Deaths'] - complete_data['Recovered']

    # replacing Mainland china with just China
    complete_data['Country/Region'] = complete_data['Country/Region'].replace('Mainland China', 'China')
    
    st.markdown("Generating **Active Cases** using **Confirmed**,**Deaths**, and **Recovered**")
    if st.checkbox("Updated Dataset"):
        number = st.number_input("Rows", 5, 100,key='updated')
        st.dataframe(complete_data.head(number))

    #Missing Values Removal
    complete_data[['Province/State']] = complete_data[['Province/State']].fillna('')
    complete_data.rename(columns={'Date':'date'}, inplace=True)
    data = complete_data

   
    grouped = data.groupby('date')['date', 'Confirmed', 'Deaths','Active'].sum().reset_index()
    fig = px.line(grouped, x="date", y="Confirmed", title="Worldwide Confirmed Cases Over Time")
    st.plotly_chart(fig)

    #lINE plot representing Recovered Deaths and Active Cases
    temp = data.groupby('date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
    temp = temp.melt(id_vars="date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='case', value_name='count')
    fig = px.area(temp, x="date", y="count", color='case',
             title='Cases over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
    st.plotly_chart(fig)

    
    data['Province/State'] = data['Province/State'].fillna('')
    temp = data[[col for col in data.columns if col != 'Province/State']]

    latest = temp[temp['date'] == max(temp['date'])].reset_index()
    latest_grouped = latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()
    fig = px.bar(latest_grouped.sort_values('Confirmed', ascending=False)[:30][::-1], 
             x='Confirmed', y='Country/Region',
             title='Confirmed Cases Worldwide', text='Confirmed', height=1000, orientation='h')
    st.plotly_chart(fig)

    formated_gdf = data.groupby(['date', 'Country/Region'])['Confirmed', 'Deaths'].max()
    formated_gdf = formated_gdf.reset_index()
    formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
    formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
    formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

    st.title("Global Spread of Corona Virus")
    fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")
    # fig.update(layout_coloraxis_showscale=False)
    st.plotly_chart(fig)


    st.header("Symptom Analysis of Covid19")
    symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

    symptoms=pd.DataFrame(data=symptoms,index=range(14))
    fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 
             y="percentage", x="symptom", color='symptom', 
             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
    st.plotly_chart(fig)


if __name__ == 'main':
    main()





