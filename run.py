import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.sidebar.header('Parameter Setting')
with st.sidebar.form(key ='form1'):
    courseid = st.text_input('*Course ID (e.g., CL10120192)')
    startdate = st.text_input('*Course Start Date (e.g., 2020-02-23)')
    enddate = st.text_input('*Course End Date (e.g., 2020-06-29)')
    studentid = st.text_input('Student ID (e.g., 3180300116)')
    cluster_n = st.selectbox('Number of Clusters Preferred', ('1', '2', '3', '4', '5', 
    '6', '7', '8', '9', '10'))

    st.write('*: required parameters')
    submitted1 = st.form_submit_button(label = 'Submit Parameters')


@st.cache
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

    return table


@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def character_PCA(table, PC_n=3):
    #PC_n=3 is default which is considered fixed.
    pca = PCA(n_components=PC_n, svd_solver = "full")
    pca_result = pca.fit_transform(table)
    df_Learners_scaled_PCA = table[:]
    for i in range(PC_n):
        df_Learners_scaled_PCA['PC' + str(i + 1)] = pca_result[:, i]

    (fig, ax) = plt.subplots(figsize=(11, 11))  #create a fig with 1 ax
    for i in range(0, len(table.columns)):    
        ax.arrow(0,
                 0,    
                 pca.components_[0, i],    
                 pca.components_[1, i],    
                 head_width=0.01,
                 head_length=0.01,
                color = 'slategray')
        plt.text(pca.components_[0, i] + 0.005,
                 pca.components_[1, i] + 0.005,
                 table.columns[i], 
                color = 'slategray')    
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(1 * np.cos(an), 1* np.sin(an), color = 'slategray') 
    plt.axis('equal')
    ax.set_title('Variable factor map')

    ax.spines['left'].set_position('zero')  
    ax.spines['bottom'].set_position('zero')  

    plt.axvline(0) 
    plt.axhline(0) 
    plt.close()

    (fig1, ax) = plt.subplots(figsize=(7, 5))  #create a fig with 1 ax  
    check_eigenValues_pca = pca.explained_variance_ratio_
    plt.bar(range(1, len(check_eigenValues_pca)+1), check_eigenValues_pca, color = 'slategray')
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(check_eigenValues_pca)+1),
             np.cumsum(check_eigenValues_pca),
             c='black',
             label="Cumulative Explained Variance")
    plt.legend(loc='center right')
    plt.close()
    return fig1


@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def character_kmeans(table, cluster_n=10, PC_n=2): 
    #PC_n=2 & cluster_n=10 are default which is considered fixed.
    pca = PCA(n_components=PC_n, svd_solver = "full")
    pca_result = pca.fit_transform(table)
    df_Learners_scaled_PCA = table[:]
    for i in range(PC_n):
        df_Learners_scaled_PCA['PC' + str(i + 1)] = pca_result[:, i]
    df_PC = df_Learners_scaled_PCA.iloc[:, -PC_n:]
    sse = {}
    for k in range(1, cluster_n):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_PC)    
        sse[k] = kmeans.inertia_       
    (fig2, ax) = plt.subplots(figsize=(7, 5))  #create a fig with 1 ax
    plt.plot(list(sse.keys()), list(sse.values()), color = 'slategray')
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.close()
    return fig2

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def character_persona(table, cluster_n, 
                        cluster_coldict_n = {0:'cornflowerblue', 
                        1:'mediumaquamarine', 2:'khaki', 3:'tomato',
                        4:'lightpink', 5:'cyan', 6:'blue',
                        7:'teal', 8:'yellowgreen', 9:'olive'}, 
                        PC_n=2): 
    #PC_n=2 is default which is considered fixed.
    pca = PCA(n_components=PC_n, svd_solver = "full")
    pca_result = pca.fit_transform(table)
    df_Learners_scaled_PCA = table[:]
    for i in range(PC_n):
        df_Learners_scaled_PCA['PC' + str(i + 1)] = pca_result[:, i]
    df_PC = df_Learners_scaled_PCA.iloc[:, -PC_n:]
    kmeans = KMeans(n_clusters=cluster_n)
    kmeans.fit(df_PC)
    y_kmeans = kmeans.predict(df_PC)
    df_Learners_scaled_PCA['cluster'] = y_kmeans.astype('float32')
    
    
    #fig3: k-means result
    fig3, ax = plt.subplots(figsize=(22, 21))  #create a fig with 1 ax
    colors =cluster_coldict_n 
    PSGR=df_Learners_scaled_PCA['cluster'].apply(lambda x: colors[x])   
    x = df_Learners_scaled_PCA['PC1'].astype('float32')
    y = df_Learners_scaled_PCA['PC2'].astype('float32')
    
    for i in range(0, len(table.columns)): 
        ax.arrow(0,
                 0,  
                 pca.components_[0, i],  
                 pca.components_[1, i],  
                 head_width=0.01,
                 head_length=0.01)
        plt.text(pca.components_[0, i] + 0.01,
                 pca.components_[1, i] + 0.01,
                 table.columns[i],
                 fontstyle = 'italic', 
                 fontsize = 'large', 
                 color = 'slategray') 
    ax.spines['left'].set_position('zero')  
    ax.spines['bottom'].set_position('zero')  
    plt.axvline(0)  
    plt.axhline(0)  
        
    plt.scatter(x, y, c=PSGR, s=150, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='dimgray', s=600, alpha=0.5)
    cluster_list = ["cluster {0}".format(i+1) for i in range(cluster_n)]
    for i, txt in enumerate(cluster_list):
        plt.annotate(txt, (centers[i, 0], centers[i, 1]), color='black', fontsize=20)
    plt.axis('equal')
    plt.close()

    #fig4: persona result
    cluster_df = {}
    for i in range(cluster_n):
        cluster_df[i] = df_Learners_scaled_PCA.loc[df_Learners_scaled_PCA['cluster'] == i]
    mean_cluster_list = {}
    for i in range(cluster_n):
       mean_cluster_list[i] = cluster_df[i].mean(axis=0)[:6].tolist() 
  
    categories = table.columns.tolist()        
    theta = np.linspace(0,2*np.pi,len(categories),endpoint=False)
    theta = np.concatenate((theta,[theta[0]]))
    categories = np.concatenate((categories,[categories[0]])) 
    fig4, axes = plt.subplots(1, cluster_n, subplot_kw=dict(polar=True), figsize=(22, 7))  #create a fig with n axes
    
    for i in range(cluster_n):     
        persona = mean_cluster_list[i]
        persona = np.concatenate((persona,[persona[0]]))        
        
        axes[i].plot(theta, persona, cluster_coldict_n[i], alpha = 0.25)
        axes[i].fill(theta, persona, cluster_coldict_n[i], alpha = 0.75)
        axes[i].set_thetagrids(theta*180/np.pi,categories)
        axes[i].set_ylim(0,1)
        axes[i].set_title(' - ', fontsize = 20)
    plt.close()
    return fig3, fig4


@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def character_persona_id(table, cluster_n, studentid,
                        cluster_coldict_n = {0:'cornflowerblue', 
                        1:'mediumaquamarine', 2:'khaki', 3:'tomato',
                        4:'lightpink', 5:'cyan', 6:'blue',
                        7:'teal', 8:'yellowgreen', 9:'olive'}, 
                        PC_n=2): 
    #PC_n=2 is default which is considered fixed.
    pca = PCA(n_components=PC_n, svd_solver = "full")
    pca_result = pca.fit_transform(table)
    df_Learners_scaled_PCA = table[:]
    for i in range(PC_n):
        df_Learners_scaled_PCA['PC' + str(i + 1)] = pca_result[:, i]
    df_PC = df_Learners_scaled_PCA.iloc[:, -PC_n:]
    kmeans = KMeans(n_clusters=cluster_n)
    kmeans.fit(df_PC)
    y_kmeans = kmeans.predict(df_PC)
    df_Learners_scaled_PCA['cluster'] = y_kmeans.astype('float32')
    
    
    #fig3: k-means result
    fig3, ax = plt.subplots(figsize=(22, 21))  #create a fig with 1 ax
    colors =cluster_coldict_n 
    PSGR=df_Learners_scaled_PCA['cluster'].apply(lambda x: colors[x])   
    x = df_Learners_scaled_PCA['PC1'].astype('float32')
    y = df_Learners_scaled_PCA['PC2'].astype('float32')
    
    for i in range(0, len(table.columns)): 
        ax.arrow(0,
                 0,  
                 pca.components_[0, i],  
                 pca.components_[1, i],  
                 head_width=0.01,
                 head_length=0.01)
        plt.text(pca.components_[0, i] + 0.01,
                 pca.components_[1, i] + 0.01,
                 table.columns[i],
                 fontstyle = 'italic', 
                 fontsize = 'large', 
                 color = 'slategray') 
    ax.spines['left'].set_position('zero')  
    ax.spines['bottom'].set_position('zero')  
    plt.axvline(0)  
    plt.axhline(0)  
        
    plt.scatter(x, y, c=PSGR, s=150, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='dimgray', s=600, alpha=0.5)
    cluster_list = ["cluster {0}".format(i+1) for i in range(cluster_n)]
    for i, txt in enumerate(cluster_list):
        plt.annotate(txt, (centers[i, 0], centers[i, 1]), color='black', fontsize=20)
    plt.axis('equal')
    plt.close()

    #fig4: persona result
    cluster_df = {}
    for i in range(cluster_n):
        cluster_df[i] = df_Learners_scaled_PCA.loc[df_Learners_scaled_PCA['cluster'] == i]
    mean_cluster_list = {}
    for i in range(cluster_n):
       mean_cluster_list[i] = cluster_df[i].mean(axis=0)[:6].tolist() 
  
    categories = table.columns.tolist()        
    theta = np.linspace(0,2*np.pi,len(categories),endpoint=False)
    theta = np.concatenate((theta,[theta[0]]))
    categories = np.concatenate((categories,[categories[0]])) 
    fig4, axes = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=(22, 7))  #create a fig with 1 axes
    
    target_cluster = df_Learners_scaled_PCA.loc[str(studentid), 'cluster']

    persona = mean_cluster_list[int(target_cluster)]
    persona = np.concatenate((persona,[persona[0]]))        
    
    axes.plot(theta, persona, cluster_coldict_n[int(target_cluster)], alpha = 0.25)
    axes.fill(theta, persona, cluster_coldict_n[int(target_cluster)], alpha = 0.75)
    axes.set_thetagrids(theta*180/np.pi,categories)
    axes.set_ylim(0,1)
    axes.set_title(' - ', fontsize = 20)
    plt.close()
    return fig3, fig4



if submitted1:
    result_table = character_table()
    if courseid and startdate and enddate and studentid:
        st.title('Character Features of Student {0}'.format(studentid))
        st.write(result_table.loc[studentid])
        if cluster_n:
            if int(cluster_n) <=10:
                fig_id1, fig_id2 = character_persona_id(table = result_table, cluster_n= int(cluster_n), 
                studentid = studentid)
                st.write(fig_id1)
                st.write(fig_id2)
        else: 
            st.write('Please enter a valid number of clusters')   
    elif courseid and startdate and enddate:
        st.title('Character Features of All Students')
        st.write(result_table)

        st.title('Explained Variance of Each PC')
        st.write(character_PCA(result_table))

        st.title('SSE of Different Number of Clusters')
        st.write(character_kmeans(result_table))
        if cluster_n:
            if int(cluster_n) <=10:
                fig_all1, fig_all2 = character_persona(table = result_table, cluster_n = int(cluster_n))
                st.write(fig_all1)
                st.write(fig_all2)
        else:
            st.write('Please enter a valid number of clusters')
    else:
        st.write('Please input all parameters required with the correct format.')   







