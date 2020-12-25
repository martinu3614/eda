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

#EDA用WebApp

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
    data_type_unique = data_type
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

#サイトタイトル
st.title('EDA（探索的データ解析）を簡単に...')

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


#______________________________________________________________________________________
#データの概要
if uploaded_file is not None:
    st.sidebar.title('データの概要')
else:
    pass

#データ型の確認
if uploaded_file is not None:
    dtype_check = st.sidebar.checkbox('データ型の確認')
    if dtype_check == True:
        st.title('データの概要')
        st.header('データ型の確認')
        st.table(data_type)
        st.write('(int : 整数 , float : 小数 , object : 文字 , bool : 真偽)')
    else:
        pass
else:
    pass


#統計量の確認
if uploaded_file is not None:
    summary_check = st.sidebar.checkbox('統計量の確認')
    summary_index = ['データ数', '平均値', '標準偏差', '最小値', '第一四分位数', '中央値', '第二四分位数', '最大値']
    if summary_check == True:
        if dtype_check == False:
            st.title('データの概要')
        st.header('要約統計量の確認')
        if len(numerical_column_list) != 0:
            summary_data = df.describe()
            summary_data.index = summary_index
            st.table(summary_data)
        else:
            st.write('数値型の変数がありません')
    else:
        pass
else:
    pass

#カテゴリ変数の値の種類
if uploaded_file is not None:
    unique_check = st.sidebar.checkbox('カテゴリ変数の値の種類')
    if unique_check == True:
        if dtype_check == False and summary_check == False:
            st.title('データの概要')
        st.header('カテゴリ変数の値の種類')
        if len(categorical_column_list) != 0:
            st.table(categorical_data_unique)
        else:
            st.write('カテゴリ変数がありません')
    else:
        pass
else:
    pass

#欠損数の確認
if uploaded_file is not None:
    null_check = st.sidebar.checkbox('欠損数の確認')
    if null_check == True:
        if dtype_check == False and summary_check == False and unique_check == False:
            st.title('データの概要')
        st.header('欠損数の確認')
        null_count = df.isnull().sum().to_frame()
        null_count.columns = ['欠損数']
        null_count['欠損割合'] = null_count['欠損数'] / len(df.index)
        st.table(null_count)
    else:
        pass
else:
    pass

#変数の相関行列
if uploaded_file is not None:
    corr_matrix_check = st.sidebar.checkbox('相関行列の確認')
    if corr_matrix_check == True:
        if dtype_check == False and summary_check == False and unique_check == False and null_check == False:
            st.title('データの概要')
        st.header('相関行列の確認')
        corr_matrix = df.corr()
        st.table(corr_matrix)
    else:
        pass
else:
    pass

#______________________________________________________________________________________
#グラフの作成
if uploaded_file is not None:
    st.sidebar.title('グラフの作成')
    grid_check = st.sidebar.checkbox('グラフにグリッドをいれる')
else:
    pass

if grid_check == True:
    sns.set()
else:
    sns.set_style('white')

#棒グラフの作成
if uploaded_file is not None:
    bar_check = st.sidebar.checkbox('棒グラフ')
    if bar_check == True:
        st.header('棒グラフ')
        st.write('（※値の種類が多いとグラフの作成に時間がかかることがあります）')
        bar_x = st.sidebar.selectbox('棒グラフのx軸の変数を選択してください', categorical_column)
        bar_hue_check = st.checkbox('カテゴリ毎に棒グラフを分ける')
        if bar_hue_check == True:
            bar_hue = st.selectbox('棒グラフを分けるカテゴリ変数', categorical_column)
            bar_fig = plt.figure()
            sns.countplot(x=bar_x, data=df, hue=bar_hue, )
        else:
            bar_fig = plt.figure()
            sns.countplot(x=bar_x, data=df)
        st.pyplot(bar_fig)
    else:
        pass
else:
    pass


#ヒストグラムの作成
#スタージェス
sturges = lambda n: math.ceil(math.log2(n*2))
if uploaded_file is not None:
    hist_check = st.sidebar.checkbox('ヒストグラム')
    if hist_check == True:
        st.header('ヒストグラム')
        hist_x = st.sidebar.selectbox('ヒストグラムのx軸の変数を指定して下さい', continuous_column)
        hist_bin = sturges(len(df[hist_x]))
        hist_hue_check = st.checkbox('カテゴリ毎にヒストグラムを分ける')
        if hist_hue_check == True:
            hist_hue = st.selectbox('ヒストグラムを分けるカテゴリ変数', categorical_column)
            hist_fig = sns.FacetGrid(df, col=hist_hue)
            hist_fig.map_dataframe(sns.histplot, x=hist_x, ec='white')
            hist_fig.set_axis_labels(hist_x, "Count")
            hist_fig_stack = plt.figure()
            sns.histplot(data=df, x=hist_x, hue=hist_hue, edgecolor='white')
            st.pyplot(hist_fig_stack)
        else:
            hist_fig = plt.figure()
            sns.histplot(data=df, x=df[hist_x], bins=hist_bin, ec='white')
        st.pyplot(hist_fig)
    else:
        pass
else:
    pass

#箱ひげ図andバイオリンプロットの作成
if uploaded_file is not None:
    box_check = st.sidebar.checkbox('箱ひげ図とバイオリンプロット')
    if box_check == True:
        st.header('箱ひげ図とバイオリンプロット')
        box_x = st.sidebar.selectbox('箱ひげ図のx軸となる変数を指定して下さい', categorical_column)
        box_y = st.sidebar.selectbox('箱ひげ図のy軸となる変数を指定して下さい', continuous_column)
        box_df = pd.concat([df[box_x], df[box_y]], axis=1, join='outer')
        box_fig = plt.figure()
        box_fig.add_subplot(111)
        sns.boxplot(x=box_x, y=box_y, data=box_df)
        st.pyplot(box_fig)
        violin_fig = plt.figure()
        sns.violinplot(x=df[box_x], y=df[box_y])
        st.pyplot(violin_fig)
    else:
        pass
else:
    pass

#散布図の作成
if uploaded_file is not None:
    scatt_check = st.sidebar.checkbox('散布図')
    if scatt_check == True:
        st.header('散布図')
        scatt_x = st.sidebar.selectbox('散布図のx軸の変数を指定して下さい', numerical_column)
        scatt_y = st.sidebar.selectbox('散布図のy軸の変数を指定して下さい', numerical_column)
        scatt_df = pd.concat([(df[scatt_x]), (df[scatt_y])], axis=1, join='outer')
        if scatt_x == scatt_y:
            st.write('x軸とy軸には別の変数を選択して下さい')
        else:
            scatt_color_check = st.checkbox('第3変数を指定する')
            if scatt_color_check == False:
                scatt_hist_list = ['普通', '六角形', '等高線', '回帰直線']
                scatt_hist_type = st.selectbox('散布図のタイプを指定して下さい', scatt_hist_list)
                if scatt_hist_type == '普通':
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df)
                elif scatt_hist_type == '六角形':
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df, kind='hex')
                elif scatt_hist_type == '等高線':
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df, kind='kde')
                else:
                    scatt_hist_fig = sns.jointplot(scatt_x, scatt_y, data=df, kind='reg')
                st.pyplot(scatt_hist_fig)
            else:
                scatt_color = st.selectbox('第3変数を選択して下さい', con_cate_column)
                scatt_df = pd.concat([scatt_df, df[scatt_color]], axis=1, join='outer')
                scatt_fig = sns.lmplot(x=scatt_x, y=scatt_y, data=scatt_df, hue=scatt_color)
                st.pyplot(scatt_fig)
    else:
        pass
else:
    pass

#散布図行列の作成
if uploaded_file is not None:
    scatt_matrix_check = st.sidebar.checkbox('散布図行列')
    if scatt_matrix_check == True:
        scatt_matrix_list = st.sidebar.multiselect('散布図行列に使用する変数', continuous_column_list, (continuous_column_list[0], continuous_column_list[1]))
        st.header('散布図行列')
        scatt_matrix_df = df[scatt_matrix_list]
        scatt_matrix_fig = sns.pairplot(scatt_matrix_df)
        st.pyplot(scatt_matrix_fig)
    else:
        pass
else:
    pass

#ヒートマップ
if uploaded_file is not None:
    heat_check = st.sidebar.checkbox('ヒートマップ')
    if heat_check == True:
        st.header('ヒートマップ')
        heat_list = st.sidebar.multiselect('ヒートマップに使用する変数', numerical_column_list, (numerical_column_list[0], numerical_column_list[1]))
        if len(heat_list) <= 1:
            st.write('変数を複数選択して下さい')
        else:
            heat_df = df[heat_list]
            #相関行列の算出
            heat_method = 'pearson'
            heat_method = st.radio('相関係数の算出方法（ヒートマップ）', ('pearson', 'spearman', 'kendall'))
            heat_corr = heat_df.corr(method=heat_method)
            #ヒートマップの作成
            value_display = True
            value_display = st.radio('相関係数の表示有無', (True, False))
            heat_fig = plt.figure()
            sns.heatmap(heat_corr,  vmin=-1.0, vmax=1.0, center=0, annot=value_display, fmt='.3f', xticklabels=heat_corr.columns.values, yticklabels=heat_corr.columns.values)
            st.pyplot(heat_fig)
    else:
        pass
else:
    pass

#折れ線グラフ
if uploaded_file is not None:
    line_chart_check = st.sidebar.checkbox('折れ線グラフ')
    if line_chart_check == True:
        st.header('折れ線グラフ')
        line_chart_x = st.sidebar.selectbox('折れ線グラフのx軸の変数を指定して下さい', numerical_column)
        line_chart_y = st.sidebar.selectbox('折れ線グラフのy軸の変数を指定して下さい', numerical_column)
        line_chart_color_check = st.checkbox('カテゴリ毎に折れ線グラフを分ける')
        if line_chart_color_check == False:
            line_chart_fig = plt.figure()
            sns.lineplot(x=line_chart_x, y=line_chart_y, data=df)
        else:
            line_chart_color = st.selectbox('折れ線グラフを分けるカテゴリ変数', categorical_column)
            line_chart_fig = plt.figure()
            sns.lineplot(x=line_chart_x, y=line_chart_y, hue=line_chart_color, data=df)
        st.pyplot(line_chart_fig)
    else:
        pass
else:
    pass