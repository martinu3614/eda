import streamlit as st
import pandas as pd
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


#___________________________________________________________________________________________
# サイドバー
# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='csv') 
if uploaded_file is not None :
    #csvファイルをdataframeに変換
    df = pd.read_csv(uploaded_file)
    #チェックボックス用のカラムリスト
    column = df.columns.values
    column_list = column.tolist()
    #数値データのカラムリスト
    data_type = df.dtypes.to_frame()
    data_type.columns = ['データ型']
    numerical_df = data_type[data_type['データ型'] != object]
    numerical_column = numerical_df.index.values
    numerical_column_list = numerical_column.tolist()
    #二値データのカラムリスト
    data_unique = df.nunique().to_frame()
    data_unique.columns = ['ユニークな要素数']
    twovalues_df = data_unique[data_unique['ユニークな要素数'] == 2]
    twovalues_column = twovalues_df.index.values
    twovalues_column_list = twovalues_column.tolist()

    #データの概要
    st.sidebar.title('データの概要')

    #データ型の確認
    check_datatype = st.sidebar.checkbox('データ型')
    if check_datatype == True :
        type_data = df.dtypes.to_frame()
    #要約統計量の確認
    check_summary = st.sidebar.checkbox('要約統計量')
    if check_summary == True :
        summary_data = df.describe()
        summary_data.index = ['データ数', '平均値', '標準偏差', '最小値', '第一四分位数', '中央値', '第二四分位数', '最大値']
    
    #欠損値数の確認
    check_null = st.sidebar.checkbox('欠損値数')
    if check_null == True :
        null_count = df.isnull().sum().to_frame()



    #グラフ作成
    st.sidebar.title('グラフ作成')

    #『ヒストグラムデータ』設定
    check_hist = st.sidebar.checkbox('ヒストグラム')
    if check_hist == True :
        hist_data = st.sidebar.selectbox('データ選択', column)

    #『散布図』『相関係数』設定
    check_corr = st.sidebar.checkbox('散布図（相関係数）')
    if check_corr == True :
        x_axis_corr = st.sidebar.selectbox('x軸項目', column)
        y_axis_corr = st.sidebar.selectbox('y軸項目', column)
    
    #『ヒートマップ』設定
    check_heat = st.sidebar.checkbox('ヒートマップ')
    if check_heat == True :
        heat_data = st.sidebar.multiselect('使用データ', numerical_column_list, numerical_column_list[0])

    
    #データ分析

    #『線形回帰分析』設定
    st.sidebar.title('線形回帰分析')
    check_regression = st.sidebar.checkbox('線形回帰分析を実施する')
    if check_regression == True :
        y_regression = st.sidebar.selectbox('目的変数', numerical_column)
        x_regression = st.sidebar.multiselect('説明変数', numerical_column_list, numerical_column_list[1])

    #『ロジスティック回帰分析』設定
    st.sidebar.title('ロジスティック回帰分析')
    check_logistic = st.sidebar.checkbox('ロジスティック回帰分析を実施する')
    if check_logistic == True:
        if len(twovalues_column) != 0:
            y_logistic = st.sidebar.selectbox('目的変数', twovalues_column)
            x_logistic = st.sidebar.multiselect('説明変数', column_list)
        else:
            st.sidebar.write('二値変数がありません')


#___________________________________________________________________________________________
# 以下、メイン画面
st.title('集計結果一覧')

#データフレーム
if uploaded_file is not None :
    data_display = st.checkbox('アップロードしたデータを表示する')
    if data_display == True :
        st.header('読み込みデータ（100行目まで）')
        st.dataframe(df.head(100))
else:
    st.header('csvファイルを選択して下さい')

if uploaded_file is not None :
    #データの概要
    if check_datatype == True :
        st.header('データ型')
        type_data.columns = ['データ型']
        st.table(type_data)

    #要約統計量の確認
    if check_summary == True :
        st.header('要約統計量')
        st.table(summary_data)

    #欠損数
    if check_null == True :
        st.header('欠損数')
        null_count.columns = ['欠損数']
        null_count['欠損割合'] = null_count['欠損数'] / len(df.index)
        st.table(null_count)
    
    #ヒストグラム
    if check_hist == True :
        st.header('ヒストグラム')
        #ヒストグラムの編集
        hist_edit = st.checkbox('グラフを編集する')
        #グラフのデフォルト
        opa = 0.6
        bins = int(math.log(len(df[hist_data]), 2)) + 1
        #詳細設定
        if hist_edit == True :
            opa = st.slider('透明度', min_value=0.0, max_value=1.0, step=0.01, value=opa)
            if df[hist_data].dtype != 'object' :
                bins = st.slider('ビンの数', min_value=2, max_value=100, step=1, value=bins)
        hist = px.histogram(data_frame=df[hist_data], x=hist_data, opacity=opa, nbins = bins)
        st.write(hist)
    
    #散布図
    if check_corr == True :
        st.header('散布図')
        check_color_corr = st.checkbox('第3変数を指定する')
        if x_axis_corr != y_axis_corr :
            scat_df = pd.concat([df[x_axis_corr],df[y_axis_corr]], axis=1, join='outer')
            if check_color_corr == True :
                color_column = st.selectbox('第3変数を指定して下さい', column)
                if x_axis_corr != color_column and y_axis_corr != color_column:
                    scat_df = pd.concat([scat_df, df[color_column]], axis=1, join='outer')
                    fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr, color=color_column)
                else:
                    fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr)
                    st.write('x,yと異なる変数を選択して下さい')
            else:
                fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
            if x_axis_corr in numerical_column_list and y_axis_corr in numerical_column_list :
                #相関係数の算出method
                method_corr = 'pearson'
                method_corr = st.radio('相関係数の算出方法（散布図）', ('pearson', 'spearman', 'kendall'))
                scat_corr = scat_df.corr(method=method_corr)
                correlation_coefficient = scat_corr.iat[0,1]
                st.write('相関係数 : ', correlation_coefficient)
            
        else:
            st.write('※2つの異なる列を選択して下さい')
    
    #ヒートマップ(相関行列)
    if check_heat == True :
        st.header('ヒートマップ（相関行列）')
        if len(heat_data) <= 1 :
            st.write('変数を複数選択して下さい')
        else:
            heat_df = df[heat_data]
            method_heat = 'pearson'
            method_heat = st.radio('相関係数の算出方法（ヒートマップ）', ('pearson', 'spearman', 'kendall'))
            heat_corr = heat_df.corr(method=method_heat)
            #ヒートマップ設定
            value_display = True
            value_display = st.radio('相関係数の表示有無', (True, False))
            fig_heat = plt.figure()
            ax1 = fig_heat.add_subplot(1, 1, 1)
            sns.heatmap(heat_corr, vmin=-1.0, vmax=1.0, center=0, annot=value_display, fmt='.1f', xticklabels=heat_corr.columns.values, yticklabels=heat_corr.columns.values)
            st.pyplot(fig_heat)

    

    #線形回帰分析
    if check_regression == True :
        st.header('線形回帰分析')
        if len(x_regression) == 0:
            st.write('説明変数を選択して下さい')
        elif y_regression in x_regression :
            st.write('説明変数に目的変数が含まれています')
        elif x_regression != y_regression :
            #説明変数の作成
            df_x_regression = df[x_regression]
            #目的変数の作成
            df_y_regression = df[y_regression]
            #モデルへの当てはめ
            linear_regression = linear_model.LinearRegression()
            linear_regression.fit(df_x_regression, df_y_regression)
            #偏回帰係数の算出
            result_regression = pd.DataFrame(linear_regression.coef_)
            result_regression = result_regression.T
            result_regression.columns = x_regression
            result_regression.index = ['偏回帰係数']
            st.table(pd.DataFrame(result_regression))
            #モデルの適合度
            r2 = linear_regression.score(df_x_regression, df_y_regression)
            st.write('モデルの適合度 : ', r2)

    #ロジスティック回帰分析
    if check_logistic == True and len(twovalues_column) != 0:
        st.header('ロジスティック回帰分析')
        if len(x_logistic) == 0:
            st.write('説明変数を選択して下さい')
        elif y_logistic in x_logistic:
            st.write('説明変数に目的変数が含まれています')
        elif x_logistic != y_logistic:
            #説明変数の作成
            df_x_logistic = df[x_logistic]
            #目的変数の作成
            df_y_logistic = df[y_logistic]
            #モデルへの当てはめ
            logistic_regression = linear_model.LogisticRegression()
            logistic_regression.fit(df_x_logistic, df_y_logistic)
            #精度
            logistic_regression.score(df_x_logistic, df_y_logistic)