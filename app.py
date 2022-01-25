import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from st_aggrid import AgGrid
import tabula
#import pyodbc 
import re
import datetime
import time
#import feather
from tabula.io import read_pdf
import matplotlib.pyplot as plt
import os
import glob
import base64
from collections import defaultdict
import re
from requests import get
from st_aggrid import AgGrid
import os
import shutil
from collections import defaultdict
import cv2
import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import xml.etree.ElementTree as ET
#from selenium.webdriver.chrome.options import Options
#from selenium.webdriver.common.keys import Keys
import dataframe_image as dfi



@st.cache(allow_output_mutation=True)
def persistdata():
    d=detentions
    return d

st.set_page_config(layout='wide')


def save_image(df,pic_name):
    pic=f'{pic_name}.png'
    dfi.export(df,pic)
    img = cv2.imread(pic) 
    scale_percent = 100
    #calculate the 30 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return cv2.imwrite(pic,output)

def parse_work_container_list(file):
    plt.close("all")
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)


    df=pd.read_excel(fr'{file}.xlsx')


    ship=df.iloc[1,1]
    voyage=df.iloc[2,1]
    shift=df.iloc[2,3]
    crane=df.iloc[3,1]
    start=df.iloc[4,1]


    df=pd.read_excel(fr'{file}.xlsx',header=6)
    df.reset_index(drop=True, inplace=True)
    moves=df.copy()
    moves.rename(columns={'Completed Time':'Stamp'},inplace=True)
    moves=moves[moves['S']=='Completed']  #########################PAY ATTENTION
    gang=f"Crane - {moves['Equip'].value_counts().index[0]}"

    moves = moves[~moves.Stamp.duplicated(keep='first')]
    moves.sort_values(by='Stamp',ascending=True,inplace=True)
    moves['Lag']=moves['Stamp'].shift(-1)
    moves['Dur']=moves['Lag']-moves['Stamp']
    duration=[round(i.total_seconds()/60,1) for i in moves.Dur]
    Total_Moves=len(duration)
    moves.loc[:,'Dur']=duration
    moves.set_index('Stamp',drop=True,inplace=True)
    moves.sort_index(ascending=True,inplace=True)
    speed=[round(60/i,2) if i>0 else 25 for i in moves.Dur[:-1]]
    speed=np.array(speed)
    speed=np.where(speed>100,30,speed)
    moves.loc[:-1,'Speed']=speed
    moves.reset_index(inplace=True)
    hours=[datetime.datetime.strftime(i,'%H:%M') for i in moves.Stamp]


    moves.set_index('Stamp',drop=True,inplace=True)
    moves.sort_index(ascending=True,inplace=True)
    moves['Times']=hours

    moves.index=moves.index.tz_localize('America/Los_Angeles')
    #moves.index.tz_localize("US/Pacific")
    moves['Times_H']=moves.index.floor('H')
    moves['Naive_Times']=[datetime.datetime.strftime(i.to_pydatetime(),'%H:%M') for i in moves['Times_H']]
    moves['Naive_Times']= pd.Categorical(moves['Naive_Times'], ordered=True,categories=moves['Naive_Times'].unique())
    moves['Origin']=[i[:4] if not i[0].isdigit() else f'Bay-{j}' for i,j in zip(moves['Origination'],moves['Bay'])]

    moves['Target_Bay']=[i[:4] if not i[0].isdigit() else f'Bay-{j}' for i,j in zip(moves['Planned To'],moves['Bay'])]
    moves.drop(columns=["Status",'Is Twinned','UTR',"Active","Held",'Seal 1',"Seal 2","Cat","Crane S.No","Segments",
                   "Sort Code","Group Code","B.Seq.","Reason for Hold","Held by User","Prevent UTR Dispatch"],inplace=True)

    return moves,crane

def Yard_Bays(df):
    moves=df
    Yard_Bays=moves['Origin'].value_counts()
    Yard_Bays=pd.DataFrame(Yard_Bays)
    Yard_Bays.rename(columns={'Origin':'Count'},inplace=True)
    Yard_Bays.index=Yard_Bays.index.rename('Yard_Bays')
    return Yard_Bays

def Speed_By_Shipbay(df):
    moves=df
    Speed_By_Shipbay=pd.DataFrame(moves.groupby('Target_Bay')['Speed'].mean())
    Speed_By_Shipbay.rename(columns={'Speed':'Average Speed'},inplace=True)
    Speed_By_Shipbay['Average Speed']=[int(i) for i in Speed_By_Shipbay['Average Speed']]
    return Speed_By_Shipbay

def Speed_By_Yardbay(df):
    moves=df
    Speed_By_Yardbay=pd.DataFrame(moves.groupby('Origin')['Speed'].mean())
    Speed_By_Yardbay.rename(columns={'Speed':'Average Speed'},inplace=True)
    Speed_By_Yardbay['Average Speed']=[int(i) for i in Speed_By_Yardbay['Average Speed']]

def Origin_toTarget(df):
    moves=df
    Origin_toTarget=moves.groupby(['Origin','Action','Target_Bay'])['Seq'].count()
    Origin_toTarget=pd.DataFrame(Origin_toTarget)
    Origin_toTarget.rename(columns={'Seq':'Count'},inplace=True)
    return Origin_toTarget

def Speed(df):
    moves=df
    moves.reset_index(inplace=True)
    bins=pd.date_range('2022-01-12 18:00','2022-01-13-03:00',10).to_pydatetime()
    bins=pd.to_datetime(bins)

    Speed=pd.DataFrame(pd.cut(moves.Stamp,bins,labels=[i for i in range(9)]).value_counts().sort_index())
    Speed.rename(columns={'Stamp':'MPH'},inplace=True)
    Speed.index=range(1,10)
    Speed.index=Speed.index.rename('Hour')

    doves=moves.set_index('Stamp',drop=True).copy()
    doves.index=pd.to_datetime(doves.index)
    doves.index=doves.index.floor('H')
    MPH=pd.DataFrame(doves.groupby(doves.index)['Seq'].count()).rename(columns={'Seq':'Moves'})
    MPH.index=MPH.index.rename('Hour')
    return MPH,Speed
    
def Ship_Bays(df):
    moves=df
    try:
        moves.set_index('Stamp',drop=True,inplace=True)
    except:
        pass
    
    Counts_By_ShipBays=pd.DataFrame(moves.groupby(['Naive_Times','Action','Bay'])['Seq'].count())
    Counts_By_ShipBays.rename(columns={'Seq':'Count'},inplace=True)
    Counts_By_ShipBays.index=Counts_By_ShipBays.index.rename({'Naive_Times':'Hour'})
    Counts_By_ShipBays=Counts_By_ShipBays[Counts_By_ShipBays['Count']>0]
    #save_image(Counts_By_ShipBays,'Counts_By_ShipBays')
    # Counts_By_Bay=pd.DataFrame(moves.resample('1H')['Bay'].value_counts())
    # Counts_By_Bay.rename(columns={'Bay':'Count'},inplace=True)
    # Counts_By_Bay.index=Counts_By_Bay.index.rename(["Hour",'Ship_Bay'])
    return Counts_By_ShipBays

def Counts_By_Origin(df):
    moves=df
    Counts_By_Origin=pd.DataFrame(moves.groupby(['Naive_Times','Action','Origin'])['Seq'].count())
    Counts_By_Origin.rename(columns={'Seq':'Count'},inplace=True)
    Counts_By_Origin.index=Counts_By_Origin.index.rename(["Hour",'Action','From'])
    Counts_By_Origin=Counts_By_Origin[Counts_By_Origin['Count']>0]
    #save_image(Counts_By_Origin,'Counts_By_Origin')
    return Counts_By_Origin

def Origin_toTarget_Times(df):
      
    moves=df
    Origin_toTarget_Times=moves.groupby(['Naive_Times','Origin','Action','Target_Bay'])['Seq'].count()
    Origin_toTarget_Times=pd.DataFrame(Origin_toTarget_Times)
    Origin_toTarget_Times.rename(columns={'Seq':'Count'},inplace=True)
    Origin_toTarget_Times=Origin_toTarget_Times[Origin_toTarget_Times['Count']>0]
    Origin_toTarget_Times.index=Origin_toTarget_Times.index.rename({"Naive_Times":"Time"})
    #save_image(Origin_toTarget_Times,'Origin_toTarget_Times')
    return Origin_toTarget_Times

def OHLC(df):
    moves=df
    sample_rate='60T'
    ohlc_df=moves.resample(sample_rate)['Speed'].ohlc()
    line_df=moves.resample(sample_rate)['Speed'].mean()
    production_ohlcplot_fig = go.Figure(data=[go.Candlestick(x=ohlc_df.index,
                open=ohlc_df['open'], high=ohlc_df['high'],
                low=ohlc_df['low'], close=ohlc_df['close'])
                     ])

    production_ohlcplot_fig.update_layout(
             height=600,
             width=1000,
             title="PRODUCTION OHLC CHART",
             xaxis_title="Hour",
             yaxis_title="Moves Per Hour",
            legend_title="Legend Title",
            font=dict(
                     family="Courier New, monospace",
                     size=18,
                     color="RebeccaPurple"),xaxis_rangeslider_visible=False)
    return production_ohlcplot_fig.show()

def production_boxplot(df):
    moves=df
    production_boxplot_fig = px.box(moves, x=moves.Times_H, y=moves.Speed)
    production_boxplot_fig.update_layout(
                         height=600,
                         width=1000,
                         title="PRODUCTION BOX CHART",
                        xaxis_title="HOUR",
                        yaxis_title="MPH",
                        legend_title="Legend Title",
                         font=dict(
                            family="Courier New, monospace",
                             size=18,
                             color="RebeccaPurple"),xaxis_rangeslider_visible=False)
    return production_boxplot_fig.show()

def Times(df):
    moves=df
    first_half_first=moves['2022-01-12 18:00' : '2022-01-12-22:00'].index.min()
    first_half_last=moves['2022-01-12 18:00' : '2022-01-12-22:00'].index.max()
    second_half_first=moves['2022-01-12 23:00' : '2022-01-13-22:00'].index.min()
    second_half_last=moves['2022-01-12 23:00' : '2022-01-13-22:00'].index.max()
    time_worked_first_half=first_half_last-first_half_first
    time_worked_second_half=second_half_last-second_half_first
    First_Move_of_First_Half=datetime.datetime.strftime(first_half_first,'%H:%M')
    Last_Move_of_First_Half=datetime.datetime.strftime(first_half_last,'%H:%M')
    First_Move_of_Second_Half=datetime.datetime.strftime(second_half_first,'%H:%M')
    Last_Move_of_Second_Half=datetime.datetime.strftime(second_half_last,'%H:%M')
    #Total_Time_Worked=round(time_worked_first_half.total_seconds()/3600,1)+round(time_worked_second_half.total_seconds()/3600,1)


def downtime_plot(df,sample_minutes):
    plt.style.use('seaborn')

    sample=sample_minutes
    test=moves.resample(f'{sample}T').Bay.count()
    test=test.to_frame().rename(columns={'Bay':'Speed'})
    test['Speed']=test['Speed']*60/sample
    test.index=pd.DatetimeIndex.strftime(test.index,'%H:%M')
    fig,ax=plt.subplots(figsize=(15,5))
    plt.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.5)
    plt.xticks(rotation = 90,fontsize=15,color="purple")
    plt.xticks(np.arange(0, len(test.index)+1, 20/sample))
    return ax.plot(test.index,test.Speed)




def get_schedule():
    df=pd.read_excel('schedule.xlsx')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.set_index('Vessel',drop=True,inplace=True)
    return df




def get_weather_frame():


    weather=defaultdict(int)
    url ='https://api.weather.gov/gridpoints/SEW/117,51/forecast'
    #url='https://api.weather.gov/points/47.0379,-122.9007'   #### check for station info with lat/long
    durl='https://api.weather.gov/alerts?zone=WAC033'
    headers = { 
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
        'Accept-Language' : 'en-US,en;q=0.5', 
        'Accept-Encoding' : 'gzip', 
        'DNT' : '1', # Do Not Track Request Header 
        'Connection' : 'close' }

    response = get(url,headers=headers).json()

    #print(response)
    for period in response['properties']['periods']:
        date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00')
        date_f=datetime.datetime.strftime(datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00'),"%b-%d")
        date=f'{date_f} {period["name"]}' 
        #print(date)
        #print(period)
        weather[date]={'Temp':period['temperature'],'Wind':f'{period["windDirection"]}-{period["windSpeed"]}',
                        'Forecast':period['shortForecast'],'Detail':period['detailedForecast']}


    weather_forecast=pd.DataFrame.from_dict(weather,orient='index')
    weather_forecast.drop('Detail',axis=1,inplace=True)
    return weather_forecast







c=pd.read_excel('current_payroll.xlsx')

shift=c['SHIFT'].unique()[0]
crane_operators=pd.DataFrame(c[c['JOB']=='Crane Operator'][['NAME','GANG']].values,columns=['NAME','GANG'])
crane_operators.GANG=[f'Gang-{i}' for i in crane_operators.GANG]

gangs=[re.findall('(\d+)',i)[0] for i in c.GANG.value_counts().keys() if len(re.findall('(\d+)',i))>0]   #### OR
gangs=[i for i in c.GANG.value_counts().keys() if i.isdigit()]
gangs=[f'Gang-{g}' for g in gangs]

detentions=defaultdict()

if shift=='NIGHT':
    nhours=['18:00-19:00','19:00-20:00','20:00-21:00','21:00-22:00','22:00-23:00','23:00-00:00','00:00-01:00','01:00-02:00','02:00-03:00']
else:
    nhours=['08:00-09:00','09:00-10:00','10:00-11:00','11:00-12:00','12:00-13:00','13:00-14:00','14:00-15:00','15:00-16:00','16:00-17:00']






for gang in gangs:
    detentions[gang]={}
    for n in nhours:
        detentions[gang][n]={'Production':0,'Crane':0,'Lids':0,'Yard':0,'LaborFill':0,'Cones':0,'Prod_Sum':0} 



if "detentions" not in st.session_state:
    st.session_state.detentions = detentions

if 'crane_operators' not in st.session_state:
    st.session_state.crane_operators=crane_operators

detentions_df=pd.DataFrame.from_dict(st.session_state.detentions[gang],orient='index',columns=['Production',"Crane", "Lids",'Yard','LaborFill','Cones'])
detentions_df=detentions_df.astype('int')
detentions_df['Prod_Sum']=detentions_df.Production.cumsum()
detentions_df['Production']=detentions_df['Production'].replace(0,np.nan)
detentions_df.reset_index(inplace=True)
detentions_df.index+=1
a=[i for i in detentions_df.Prod_Sum]
b=[i for i in detentions_df.index ]
b[4:]=[i-1 for i in b[4:]]
detentions_df['Run_Avg']=[round(i/j,0) for i,j in zip(a,b)]


if "detentions_df" not in st.session_state:
    st.session_state.detentions_df = detentions_df





def parse_gang(file,data):

    # df=pd.read_excel(fr'C:\Users\afsin\Desktop\{file}.xlsx')

    # ship=df.iloc[1,1]
    # voyage=df.iloc[2,1]
    # shift=df.iloc[2,3]
    # crane=df.iloc[3,1]
    # start=df.iloc[4,1]

    # df=pd.read_excel(fr'C:\Users\afsin\Desktop\{file}.xlsx',header=6)
    # df.reset_index(drop=True, inplace=True)
    # moves=df.copy()
    # moves.rename(columns={'Completed Time':'Stamp'},inplace=True)
    # moves=moves[moves['S']=='Completed']
    # moves = moves[~moves.Stamp.duplicated(keep='first')]
    # moves.sort_values(by='Stamp',ascending=True,inplace=True)
    # moves['Lag']=moves['Stamp'].shift(-1)
    # moves['Dur']=moves['Lag']-moves['Stamp']
    # duration=[round(i.total_seconds()/60,1) for i in moves.Dur]
    # moves.loc[:,'Dur']=duration
    # moves.set_index('Stamp',drop=True,inplace=True)
    # moves.sort_index(ascending=True,inplace=True)
    # speed=[round(60/i,2) if i>0 else 25 for i in moves.Dur[:-1]]
    # speed=np.array(speed)
    # speed=np.where(speed>100,30,speed)
    # moves.loc[:-1,'Speed']=speed
    # moves.reset_index(inplace=True)
    # hours=[datetime.datetime.strftime(i,'%H:%M') for i in moves.Stamp]
    # moves.set_index('Stamp',drop=True,inplace=True)
    # moves.sort_index(ascending=True,inplace=True)
    # moves['Times']=hours
    # first_half_first=moves['2022-01-12 18:00' : '2022-01-12-22:00'].index.min()
    # first_half_last=moves['2022-01-12 18:00' : '2022-01-12-22:00'].index.max()
    # second_half_first=moves['2022-01-12 23:00' : '2022-01-13-22:00'].index.min()
    # second_half_last=moves['2022-01-12 23:00' : '2022-01-13-22:00'].index.max()
    # time_worked_first_half=first_half_last-first_half_first
    # time_worked_second_half=second_half_last-second_half_first

    doves=moves.resample('1H').Bay.count()
    d=[datetime.datetime.strftime(i,'%H:%M') for i in doves.keys()]
    doves.index=d
    doves=doves.to_frame()
    doves['Speed']=doves['Bay']

    Moves=[i for i in doves.Bay]
    #Moves.insert(0,0)
    # Break=[0,0,0,15,25,0,0,0,0]
    # Crane=[i*25 for i in [0,  0.2,  0.2,  0,  0,  0,  0,  0,  0]]
    # Yard=[i*25 for i in  [0,  0,  0,  0,  0,  0,  0,  0,  0]]
    # Lids=[i*25 for i in  [0,  0,  0.3,  0,  0,  0.3,  0,  0,  0]]
    # LaborFill=[0,0,0,0,0,0,0,0,0]
    # Misc=[i*25 for i in  [0,  0,  0,  0,  0,  0,  0,  0,  0]]

    # data=np.array([Moves,Break,Crane,Yard,Lids,LaborFill,Misc])




def parse_entry(gang,crane_op,nhours):

    nhours=nhours
    
    
    doves=pd.DataFrame(index=nhours,columns=['Speed', 'Break', 'Crane', 'Yard', 'Lids', 'LaborFill', 'Cones'])
    
    doves['Speed']=gang_data[gang][0]    
    doves["Break"]=gang_data[gang][1]
    doves["Crane"]=gang_data[gang][2]
    doves["Yard"]=gang_data[gang][3]
    doves["Lids"]=gang_data[gang][4]
    doves["LaborFill"]=gang_data[gang][5]
    doves["Cones"]=gang_data[gang][6]

    wide_df = px.data.medals_long()
   
    fig = px.bar(doves, x=doves.index, y=['Speed','Break', 'Crane', 'Yard', 'Lids', 'LaborFill', 'Cones'],
               
                labels={'index':'HOUR','value':'PRODUCTION-Detentions'},
                text_auto=True,title=f'{gang} Production/Detentions')### title=f"Shift Production and Detentions -{ship} - {shift} - Crane {crane}",
    fig.update_traces(textposition='inside')
    fig.add_annotation(x=4, y=25,
                text="Meal",
                showarrow=True,
                arrowhead=0)
    fig.add_annotation(x=8, y=30,
                text=f"<b>TOTAL MOVES={int(st.session_state.detentions_df['Prod_Sum'].max())}</b>",
                bordercolor='black',
                font=dict(size=15),
                showarrow=False,
                arrowhead=0)
    for i,j in enumerate(st.session_state.detentions_df['Run_Avg']):
        fig.add_annotation(x=i, y=40,
                text=f"Run_Avg={j}",
                font=dict(size=13),
                showarrow=False,
                )
    
    fig.update_layout(height=700,width=1600,
        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0.0,
            dtick = 2),
            font=dict(
            family="Arial",
            size=20,
            color='#000000'),
            title=dict(
                text=f"{gang} PRODUCTION - DETENTIONS <br><br> {crane_op}",

                 x=0.5,
                y=0.95,
                font=dict(
                    family="Arial",
                    size=18,
                    color='#000000')),
            legend_title="Production /<br> Detentions"
            
            )
    
    return fig




gang_data=defaultdict()
for gang in gangs:

    Moves=[int(st.session_state.detentions[gang][i]['Production']) for i in nhours]
    Break=[0,0,0,0,25,0,0,0,0]
    Crane=[int(25*float(st.session_state.detentions[gang][i]['Crane'])) for i in nhours]
    Yard=[int(25*float(st.session_state.detentions[gang][i]['Yard'])) for i in nhours]
    Lids=[int(25*float(st.session_state.detentions[gang][i]['Lids'])) for i in nhours]
    LaborFill=[int(25*float(st.session_state.detentions[gang][i]['LaborFill'])) for i in nhours]
    Cones=[int(25*float(st.session_state.detentions[gang][i]['Cones']))for i in nhours]
    


    gang_data[gang]=[Moves,Break,Crane,Yard,Lids,LaborFill,Cones]
#print(gang_data) 
    



page_list=['Enter Detention']
for i in gangs:
    page_list.append(f'{i} Shift Production')
page_list.extend(["Payroll For This Shift","Weather Forecast",'Ship Schedule',
                    'Shift Analytics'])

page = st.selectbox("CHOOSE FROM DROPDOWN MENU", page_list) 






for gang in gangs:
    if page == f"{gang} Shift Production":
        cr=st.session_state.crane_operators
        crane_op=f"CRANE OPS - {[i for i in cr[cr['GANG']==gang]['NAME'].values][0]} - {[i for i in cr[cr['GANG']==gang]['NAME'].values][1]}"

        st.write(parse_entry(gang,crane_op,nhours))
        with st.expander("See explanation"):
            st.write("""
         Detentions are shown in units of moves. Based on an average of 
         25 moves an hour. E.g : A 0.6 detention in crane costs 0.6*25=15 Moves for that hour.
     """)
        
            st.write(pd.DataFrame.from_dict(st.session_state.detentions[gang]))
            st.write(detentions_df)





if page=="Payroll For This Shift":
    st.title('MARINE PAYROLL')
    vessels=list(c['VESSEL'].drop_duplicates().values)   ##### c was imported from payroll "current.xls"

    cols=['VESSEL','VOYAGE','JOBNUMBER','DATE','SHIFT','NAME','WORKID','JOB','JOBCODE','FROM','TO','HOURS_WORKED','COST']
    st_ms=st.multiselect("Columns: Choose From Dropdown:",c.columns.tolist(),default=['VESSEL','VOYAGE','DATE','NAME','JOB','GANG'])

    jobs = c['JOB'].drop_duplicates()
    job_choice = st.sidebar.selectbox('Select job:', jobs)

    
    selection=c.query('''JOB==@job_choice''')
    selection_col=selection.loc[:,st_ms]
    AgGrid(selection_col)
    with st.expander("See explanation"):
     st.write("""
         Choose columns to display as you see fit.
         Choose from sidebar dropdown menu for different jobs.
     """)
     



if page == "Weather Forecast":
    st.write(get_weather_frame())
    with st.expander("See explanation/Source"):
        st.write("""
         Data compiled from National Weather Service API instantly
     """)




if page == "Ship Schedule":
    st.write(get_schedule())
    with st.expander("See explanation/Source"):
        st.write("""
         Ship Schedule compiled from nwseaportalliance.com periodically.
         Tide Information is calculated from NOAA Tides Web API based on ETA date/time""")




if page=="Enter Detention":
    st.title("PRODUCTION/DETENTION ENTRY")
   

    @st.cache(allow_output_mutation=True)
    def persistdata():
        d=detentions
        return d

  
    with st.container():
        gang=st.selectbox("Gangs", gangs)
        hour=st.selectbox("hour", nhours)
        d = persistdata()
        col1, col2 = st.columns(2)
        with col1:
            k = st.selectbox("type", ['Production',"Crane", "Lids",'Yard','LaborFill','Cones'])
        with col2:
            v = st.text_input("Value")
        button = st.button("Add Production/Detention")
        if button:
            if k and v:
                d[gang][hour][k] = v
        st.session_state.detentions=d

        detentions_df=pd.DataFrame.from_dict(st.session_state.detentions[gang],orient='index',columns=['Production',"Crane", "Lids",'Yard','LaborFill','Cones'])
        detentions_df=detentions_df.astype('float')
        detentions_df['Prod_Sum']=detentions_df.Production.cumsum()
        detentions_df['Production']=detentions_df['Production'].replace(0,np.nan)
        detentions_df.reset_index(inplace=True)
        detentions_df.index+=1
        a=[i for i in detentions_df.Prod_Sum]
        b=[i for i in detentions_df.index ]
        b[4:]=[i-1 for i in b[4:]]
        detentions_df['Run_Avg']=[round(i/j,0) for i,j in zip(a,b)]
        st.session_state.detentions_df=detentions_df
        #detentions_df['Rolling']=detentions_df.Production.rolling(2).mean()
        with st.expander("See explanation"):
            st.write(""" Detentions are shown in units of moves. Based on an average of 25 moves an hour. E.g : A 0.6 detention in crane costs 0.6*25=15 Moves for that hour.     """)

            detensions_df=detentions_df.astype(str)
            c=pd.DataFrame.from_dict(st.session_state.detentions[gang]))
            st.write(c.astype(str))
            st.write(detentions_df)
if page=="Shift Analytics":
    gang_selection = st.sidebar.radio(
    "Choose Gang",
    ([gang for gang in gangs]),key=f'gang_choice')
    for k in gangs:
        if gang_selection== k:
            
            moves=parse_work_container_list(f'{k}')[0]
            st.write(parse_work_container_list(f'{k}')[1])
            selection = st.sidebar.radio(
                    "Choose Analytics",
                ("Count By Yard Bays", "Count By Ship Bays", "Count By Origin",
                    "Origin To Target By Time","Downtime Plot"),key='AnalyticChoice')
            if selection=="Count By Yard Bays":
                st.dataframe(Yard_Bays(moves))
            if selection=="Count By Ship Bays":
                st.dataframe(Ship_Bays(moves))
                #st.image('Counts_By_ShipBays.png')
            if selection=="Count By Origin":
                st.dataframe(Counts_By_Origin(moves))
                #st.image('Counts_By_Origin.png')
            if selection=="Origin To Target By Time":
                st.dataframe(Origin_toTarget_Times(moves))
                #st.image('Origin_toTarget_Times.png')
            if selection=="Downtime Plot":
                plt.style.use('seaborn')
            
                plo_type = st.sidebar.radio(
                    "Choose Plot Type",
                    ("LINE", "SCATTER" ))
                sample = st.slider(
                    'Select Sampling Time in Minutes - Reccommended : 6 Mins',
                        2, 30, 6,4)
            
                test=moves.resample(f'{sample}T').Bay.count()
                test=test.to_frame().rename(columns={'Bay':'Speed'})
                test['Speed']=test['Speed']*60/sample
                test.index=pd.DatetimeIndex.strftime(test.index,'%H:%M')
                fig,ax=plt.subplots(figsize=(15,6))
                plt.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.5)
                plt.xticks(rotation = 90,fontsize=15,color="purple")
                plt.xticks(np.arange(0, len(test.index)+1, 20/sample))
                font1 = {'family':'serif','color':'blue','size':20}
                font2 = {'family':'serif','color':'darkred','size':15}

                plt.title("Production Speed By Time", fontdict = font1)
                plt.xlabel("TIME", fontdict = font2)
                plt.ylabel("MPH", fontdict = font2)
                if plo_type=='LINE':
                    ax.plot(test.index,test.Speed)
                else:
                    ax.scatter(test.index,test.Speed)
                st.pyplot(fig)