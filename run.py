import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import math
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.sidebar.header('Parameter Setting')
with st.sidebar.form(key ='form1'):
    courseid = st.text_input('Course ID (e.g., CL10120192)')
    startdate = st.text_input('Course Start Date (e.g., 2020-02-23)')
    enddate = st.text_input('Course End Date (e.g., 2020-06-29)')
    studentid = st.text_input('Student ID (e.g., 3180300116)')
    submitted1 = st.form_submit_button(label = 'Submit Parameters')



def character_table():
    url = "https://raw.githubusercontent.com/kevinTHlin/Character_Persona/master/app/{0}.csv".format(courseid)
    df_uploaded = pd.read_csv(url, sep=",")
    class_start_date =  datetime.strptime(startdate , '%Y-%m-%d').date()
    class_end_date = datetime.strptime(enddate , '%Y-%m-%d').date()
    user = df_uploaded['get_user_ID'].unique()
    Table = {}     
    Learners = {}
    
    for i in user:
        Table[i] = df_uploaded.loc[df_uploaded['get_user_ID'] == i]
        Table[i]['get_user_ID'] = Table[i]['get_user_ID'].astype(str)
        Table[i] = Table[i].replace('不足1s', '00:00:01')
        Table[i]['time_spent'] = pd.to_timedelta(Table[i].time_spent)
        Table[i]['visit_time'] = pd.to_datetime(Table[i].visit_time, format='%Y.%m.%d %H:%M')
        Table[i]['visit_date'] = Table[i]['visit_time'].dt.date
        Table[i] = Table[i].sort_values('visit_time')
        Table[i]['time_spent'] = Table[i]['time_spent'].dt.total_seconds()
        Table[i] = Table[i].groupby(['visit_date']).agg(time_sum = 
                                                        ('time_spent', 'sum'),
                                                        visits_count = 
                                                        ('visit_time', 'count')).reset_index()
        Table[i]['day_difference'] = Table[i].diff(periods=1, axis=0)['visit_date'].fillna(pd.Timedelta(days=0))  
        Table[i]['day_difference'] = Table[i]['day_difference'].apply(lambda x: x.days)
        Table[i]['time_spent_pv'] = Table[i]['time_sum']/Table[i]['visits_count']        
        Learners["LEARNER{0}".format(i)] = []
    
        sum_Q1 = Table[i]['time_sum'].quantile(0.25)
        sum_Q3 = Table[i]['time_sum'].quantile(0.75)
        sum_IQR = sum_Q3 - sum_Q1
        
        before_class = Table[i][(Table[i]['visit_date'] < class_start_date)]['time_sum']
        in_semester = Table[i].loc[(Table[i]['visit_date'] >= class_start_date)
                                   & (Table[i]['visit_date'] <= class_end_date)]['time_sum']
        after_class = Table[i][(Table[i]['visit_date'] > class_end_date)]['time_sum']
        
        if before_class.sum(axis=0) > 0:
            meta = 1
            in_moti_before = before_class.median()
        else: 
            meta = 0
            in_moti_before = 0
        
        if after_class.sum(axis=0) > 0:
            in_moti_after = after_class.median()
        else:
            in_moti_after = 0
                    
        half = round(Table[i].shape[0]/2)
        first_half = Table[i].loc[:half, 'time_sum'] 
        second_half = Table[i].loc[half:, 'time_sum']  
        
        #Six Character Features
        Grit = Table[i]['time_sum'].median() / sum_IQR
        Self_control = Table[i]['time_spent_pv'].median()
        Meta_cog_Self_reg = (meta + (Table[i]['day_difference'].isin(Table[i]['day_difference'].mode()).count()/(Table[i]['day_difference'].nunique()))) / sum_IQR
        Motivation = in_moti_before + in_semester.median() + in_moti_after
        Engagement = Table[i]['time_sum'].median()
        Self_perception = -math.log(1 + abs(first_half.median() - second_half.median()))
                
        Learners["LEARNER{0}".format(i)].extend([str(i), Grit, Self_control, Meta_cog_Self_reg, Motivation, Engagement, Self_perception])    
        
    Learners_col = ['user_ID', 'Grit', 'Self_control', 'Meta_cog_Self_reg', 'Motivation', 'Engagement', 'Self_perception']
    df_Learners = pd.DataFrame(columns = Learners_col)
    for i in user:
        a_series = pd.Series(Learners['LEARNER' + str(i)], index = df_Learners.columns)
        df_Learners = df_Learners.append(a_series, ignore_index=True)
    df_Learners.set_index('user_ID', inplace=True)
    df_Learners = df_Learners.astype(np.float64)
        
    print(df_Learners.isnull().sum().sum())
    print(df_Learners.isin([np.inf, -np.inf]).sum().sum())        
    scaler = MinMaxScaler()    
    df_Learners_scaled = df_Learners[:]
    df_Learners_scaled = scaler.fit_transform(df_Learners_scaled)    
    table = pd.DataFrame(df_Learners_scaled, index = df_Learners.index, 
                                        columns = df_Learners.columns).round(4)

    return st.write(table.loc[studentid])



st.title('Character Persona')
if submitted1:
    if courseid and startdate and enddate and studentid:
        character_table()
    else:
        st.write('Please input all parameters required with the correct format.')