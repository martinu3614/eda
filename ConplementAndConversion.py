import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
from pylab import rcParams
import seaborn as sns
import sys
import scipy.stats as sts

#データ補完、変換用WebAPP
#サイトタイトル
st.title('データ補完&変換を簡単に...')

#ファイルアップロード
uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='csv')


if uploaded_file is not None:
    #アップロードファイルをDataFrameに変換
    df = pd.read_csv(uploaded_file)
    #データ型
    data_type = df.dtypes.to_frame()
    data_type.columns = ['データ型']
    #カラムのリスト
    column = df.columns.values
    column_list = column.tolist()
    #数値データのカラムリスト
    numerical_df = data_type[data_type['データ型'] != object]
    numerical_column = numerical_df.index.values
    numerical_column_list = numerical_column.tolist()
    #二値データのカラムリスト
    data_unique = df.nunique().to_frame()
    data_unique.columns = ['ユニークな要素数']
    twovalues_df = data_unique[data_unique['ユニークな要素数'] == 2]
    twovalues_column = twovalues_df.index.values
    twovalues_column_list = twovalues_column.tolist()
    #カテゴリ変数のカラムリスト（昇順）
    data_type_unique = df.dtypes.to_frame()
    data_type_unique['ユニークな要素数'] = data_unique['ユニークな要素数']
    categorical_column = data_type_unique[data_type_unique['ユニークな要素数'] < 10].index.values
    categorical_column_list = categorical_column.tolist()
    if len(categorical_column_list) != 0:
        categorical_data = df[categorical_column]  
        categorical_data_unique = categorical_data.nunique().to_frame()
        categorical_data_unique.columns = ['値の種類']
        #カラムリストを要素の種類数でソート
        categorical_data_unique = categorical_data_unique.sort_values('値の種類')
        categorical_column = categorical_data_unique.index.values
        categorical_column_list = categorical_column.tolist()
    #連続変数のカラムリスト
    continuous_column = np.intersect1d(numerical_column, data_unique[data_unique['ユニークな要素数'] >= 10].index.values)
    continuous_column_list = continuous_column.tolist()
    #連続変数とカテゴリ変数を合わせたカラムリスト
    con_cate_column = np.union1d(categorical_column, continuous_column)
    con_cate_column_list = con_cate_column.tolist()
else:
    pass


#______________________________________________________________________________________
#アップロードデータの確認
if uploaded_file is not None:
    data_display = st.checkbox('アップロードしたデータを表示する')
    if data_display == True:
        st.header('読み込みデータ（100行目まで）')
        st.dataframe(df.head(100))
    else:
        pass
else:
    st.header('csvファイルを選択して下さい')

