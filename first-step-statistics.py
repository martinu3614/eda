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

#再起処理回数設定
sys.setrecursionlimit(100000)


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
else:
    pass


if uploaded_file is not None:
    #データの整形
    st.sidebar.title('データの整形')

    #欠損値データの除去
    check_removal = st.sidebar.checkbox('欠損値データの除去')
    if check_removal == True:
        st.title('欠損値データの除去')
        removal_method = st.radio('除去方法', ('ペアワイズ除去', 'リストワイズ除去'))
        if removal_method == 'ペアワイズ除去':
            df_after = df.dropna()
            #チェックボックス用のカラムリスト
            column_after = df_after.columns.values
            column_list_after = column_after.tolist()
            #数値データのカラムリスト
            data_type_after = df_after.dtypes.to_frame()
            data_type_after.columns = ['データ型']
            numerical_df_after = data_type_after[data_type_after['データ型'] != object]
            numerical_column_after = numerical_df_after.index.values
            numerical_column_list_after = numerical_column_after.tolist()
            #二値データのカラムリスト
            data_unique_after = df_after.nunique().to_frame()
            data_unique_after.columns = ['ユニークな要素数']
            twovalues_df_after = data_unique_after[data_unique_after['ユニークな要素数'] == 2]
            twovalues_column_after = twovalues_df_after.index.values
            twovalues_column_list_after = twovalues_column_after.tolist()
        elif removal_method == 'リストワイズ除去':
            listwize_list = st.multiselect('リストワイズ除去の対象変数', column_list, column_list[0])
            df_after = df[listwize_list].dropna()
            #チェックボックス用のカラムリスト
            column_after = df_after.columns.values
            column_list_after = column_after.tolist()
            #数値データのカラムリスト
            data_type_after = df_after.dtypes.to_frame()
            data_type_after.columns = ['データ型']
            numerical_df_after = data_type_after[data_type_after['データ型'] != object]
            numerical_column_after = numerical_df_after.index.values
            numerical_column_list_after = numerical_column_after.tolist()
            #二値データのカラムリスト
            data_unique_after = df_after.nunique().to_frame()
            data_unique_after.columns = ['ユニークな要素数']
            twovalues_df_after = data_unique_after[data_unique_after['ユニークな要素数'] == 2]
            twovalues_column_after = twovalues_df_after.index.values
            twovalues_column_list_after = twovalues_column_after.tolist()
        else:
            pass
        st.header('欠損値処理後データの出力（csvファイル）')
        file_name = st.text_input('ファイル名を入力して下さい（Enterで確定）')
        out_put = st.button('出力する')
        if out_put == True and file_name != '':
            if '.csv' in file_name:
                df_after.to_csv(file_name)
            else:
                df_after.to_csv(file_name + '.csv')
        elif out_put == True and file_name == '':
            st.write('※ファイル名を入力して下さい※')
        else:
            pass
    else:
        pass
else:
    pass



if uploaded_file is not None:
    #データの概要
    st.sidebar.title('データの概要')

    #データ型の確認
    check_datatype = st.sidebar.checkbox('データ型')
    if check_datatype == True and check_removal == True:
        type_data = df.dtypes.to_frame()
        type_data.columns = ['データ型']
        type_data_after = df_after.dtypes.to_frame()
        type_data_after.columns = ['データ型']
    elif check_datatype == True and check_removal == False:
        type_data = df.dtypes.to_frame()
        type_data.columns = ['データ型']
    else:
        pass

    #要約統計量の確認
    check_summary = st.sidebar.checkbox('要約統計量')
    summary_index = ['データ数', '平均値', '標準偏差', '最小値', '第一四分位数', '中央値', '第二四分位数', '最大値']
    if check_summary == True and check_removal == True:
        if len(numerical_column_list_after) != 0:
            summary_data = df.describe()
            summary_data.index = summary_index
            summary_data_after = df_after.describe()
            summary_data_after.index = summary_index
        elif len(numerical_column_list_after) == 0 and len(numerical_column_list) != 0: 
            summary_data = df.describe()
            summary_data.index = summary_index
        else:
            pass
    elif check_summary == True and check_removal == False:
        if len(numerical_column_list) != 0:
            summary_data = df.describe()
            summary_data.index = summary_index
        else:
            pass
    else:
        pass
    
    #欠損値数の確認
    check_null = st.sidebar.checkbox('欠損値数')
    if check_null == True and check_removal == True:
        null_count = df.isnull().sum().to_frame()
        null_count_after = df_after.isnull().sum().to_frame()
        null_count.columns = ['欠損数']
        null_count_after.columns = ['欠損数']
    elif check_null == True and check_removal == False:
        null_count = df.isnull().sum().to_frame()
        null_count.columns = ['欠損数']
    else:
        pass
else:
    pass




if uploaded_file is not None:
    #グラフ作成
    st.sidebar.title('グラフ作成')

    #『ヒストグラムデータ』設定
    check_hist = st.sidebar.checkbox('ヒストグラム')
    if check_hist == True and check_removal == True:
        hist_data = st.sidebar.selectbox('データ選択（除去前）', column)
        hist_data_after = st.sidebar.selectbox('データ選択（除去後）', column_after)
    elif check_hist == True and check_removal == False:
        hist_data = st.sidebar.selectbox('データ選択', column)
    else:
        pass

    #『散布図』『相関係数』設定
    check_corr = st.sidebar.checkbox('散布図（相関係数）')
    if check_corr == True and check_removal == True:
        st.write('除去前変数')
        x_axis_corr = st.sidebar.selectbox('x軸項目（除去前）', column)
        y_axis_corr = st.sidebar.selectbox('y軸項目（除去前）', column)
        st.write('除去後変数')
        x_axis_corr_after = st.sidebar.selectbox('x軸項目（除去後）', column_after)
        y_axis_corr_after = st.sidebar.selectbox('y軸項目（除去後）', column_after)
    elif check_corr == True and check_removal == False:
        x_axis_corr = st.sidebar.selectbox('x軸項目（除去前）', column)
        y_axis_corr = st.sidebar.selectbox('y軸項目（除去前）', column)
    else:
        pass
    
    #『ヒートマップ』設定
    check_heat = st.sidebar.checkbox('ヒートマップ')
    if check_heat == True :
        heat_data = st.sidebar.multiselect('使用データ', numerical_column_list, numerical_column_list[0])
    else:
        pass
else:
    pass



if uploaded_file is not None:
    #データ分析

    #『線形回帰分析』設定
    st.sidebar.title('線形回帰分析')
    check_regression = st.sidebar.checkbox('線形回帰分析を実施する')
    if check_regression == True :
        y_regression = st.sidebar.selectbox('目的変数', numerical_column)
        x_regression = st.sidebar.multiselect('説明変数', numerical_column_list, numerical_column_list[1])
    else:
        pass

    #『ロジスティック回帰分析』設定
    st.sidebar.title('ロジスティック回帰分析')
    check_logistic = st.sidebar.checkbox('ロジスティック回帰分析を実施する')
    if check_logistic == True:
        if len(twovalues_column) != 0:
            y_logistic = st.sidebar.selectbox('目的変数', twovalues_column)
            x_logistic = st.sidebar.multiselect('説明変数', column_list)
        else:
            st.sidebar.write('二値変数がありません')
    else:
        pass
else:
    pass


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
        pass
else:
    st.header('csvファイルを選択して下さい')



if uploaded_file is not None :
    #データの概要
    if check_datatype == True and check_removal == True:
        st.header('データ型')
        datatype_select = st.radio('表示するデータ概要の選択', ('除去前', '除去後'))
        if datatype_select == '除去前':
            st.table(type_data)
        else:
            st.table(type_data_after)
    elif check_datatype == True and check_removal == False:
        st.table(type_data)
    else:
        pass
else:
    pass
 

if uploaded_file is not None :
    #要約統計量の確認
    if check_summary == True and check_removal == True:
        st.header('要約統計量')
        summary_select = st.radio('表示する要約統計量の選択', ('除去前', '除去後'))
        if summary_select == '除去前':
            if len(numerical_column_list) == 0:
                st.write('数値型の変数がありません')
            else:
                st.table(summary_data)
        else:
            if len(numerical_column_list_after) == 0:
                st.write('数値型の変数がありません')
            else:
                st.table(summary_data_after)
    elif check_summary == True and check_removal == False:
        st.header('要約統計量')
        if len(numerical_column_list) == 0:
            st.write('数値型の変数がありません')
        else:
            st.table(summary_data)
    else:
        pass
else:
    pass

if uploaded_file is not None :
    #欠損数
    if check_null == True and check_removal == True:
        st.header('欠損値')
        null_select = st.radio('表示する欠損数の選択', ('除去前', '除去後'))
        if null_select == '除去前':
            null_count['欠損割合'] = null_count['欠損数'] / len(df.index)
            st.table(null_count)
        else:
            null_count_after['欠損割合'] = null_count_after['欠損数'] / len(df.index)
            st.table(null_count_after)
    elif check_null == True and check_removal == False:
        st.header('欠損値')
        null_count['欠損割合'] = null_count['欠損数'] / len(df.index)
        st.table(null_count)
    else:
        pass
else:
    pass
        


if uploaded_file is not None :
    #ヒストグラム
    if check_hist == True and check_removal == True:
        st.header('ヒストグラム')
        hist_select = st.radio('表示するヒストグラムの選択', ('除去前', '除去後'))
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
            else:
                pass
        else:
            pass
        if hist_select == '除去前':
            hist = px.histogram(data_frame=df[hist_data], x=hist_data, opacity=opa, nbins = bins)
            st.write(hist)
        else:
            hist_after = px.histogram(data_frame=df_after[hist_data_after], x=hist_data_after, opacity=opa, nbins = bins)
            st.write(hist_after)
    elif check_hist == True and check_removal == False:
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
            else:
                pass
        else:
            pass
        hist = px.histogram(data_frame=df[hist_data], x=hist_data, opacity=opa, nbins = bins)
        st.write(hist)
    else:
        pass
else:
    pass



if uploaded_file is not None :
    #散布図
    #欠損値処理済
    if check_corr == True and check_removal == True:
        st.header('散布図')
        corr_select = st.radio('表示する散布図の選択', ('除去前', '除去後'))
        check_color_corr = st.checkbox('第3変数を選択する')
        #処理前表示
        if corr_select == '除去前':
            if x_axis_corr != y_axis_corr and check_color_corr == True:
                scat_df = pd.concat([df[x_axis_corr],df[y_axis_corr]], axis=1, join='outer')
                color_column = st.selectbox('第3変数を選択して下さい', column)
                if x_axis_corr != color_column and y_axis_corr != color_column:
                    scat_df = pd.concat([scat_df, df[color_column]], axis=1, join='outer')
                    fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr, color=color_column)
                else:
                    st.write('第3変数にはx,yと異なる変数を選択して下さい')
                    fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            elif x_axis_corr != y_axis_corr and check_color_corr == False:
                scat_df = pd.concat([df[x_axis_corr],df[y_axis_corr]], axis=1, join='outer')
                fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.write('xとyは異なる変数を選択して下さい')
                fig_corr = None
        #処理後表示    
        else:
            if x_axis_corr_after != y_axis_corr_after and check_color_corr == True:
                scat_df = pd.concat([df[x_axis_corr_after],df[y_axis_corr_after]], axis=1, join='outer')
                color_column = st.selectbox('第3変数を選択して下さい', column_after)
                if x_axis_corr_after != color_column and y_axis_corr_after != color_column:
                    scat_df = pd.concat([scat_df, df[color_column]], axis=1, join='outer')
                    fig_corr = px.scatter(scat_df, x=x_axis_corr_after, y=y_axis_corr_after, color=color_column)
                else:
                    st.write('第3変数にはx,yと異なる変数を選択して下さい')
                    fig_corr = px.scatter(scat_df, x=x_axis_corr_after, y=y_axis_corr_after)

                st.plotly_chart(fig_corr, use_container_width=True)
            elif x_axis_corr_after != y_axis_corr_after and check_color_corr == False:
                scat_df = pd.concat([df[x_axis_corr_after],df[y_axis_corr_after]], axis=1, join='outer')
                fig_corr = px.scatter(scat_df, x=x_axis_corr_after, y=y_axis_corr_after)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.write('x,yは異なる変数を選択して下さい')
                fig_corr = None
    #欠損値処理なし
    elif check_corr == True and check_removal == False:
        st.header('散布図')
        check_color_corr = st.checkbox('第3変数を選択する')
        if x_axis_corr != y_axis_corr and check_color_corr == True:
            scat_df = pd.concat([df[x_axis_corr],df[y_axis_corr]], axis=1, join='outer')
            color_column = st.selectbox('第3変数を選択して下さい', column)
            if x_axis_corr != color_column and y_axis_corr != color_column:
                scat_df = pd.concat([scat_df, df[color_column]], axis=1, join='outer')
                fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr, color=color_column)
            else:
                st.write('第3変数にはx,yと異なる変数を選択して下さい')
                fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
        elif x_axis_corr != y_axis_corr and check_color_corr == False:
            scat_df = pd.concat([df[x_axis_corr],df[y_axis_corr]], axis=1, join='outer')
            fig_corr = px.scatter(scat_df, x=x_axis_corr, y=y_axis_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.write('xとyは異なる変数を選択して下さい')
            fig_corr = None
    else:
        fig_corr = None
else:
    pass

#相関係数の表示
if uploaded_file is not None:
    if fig_corr is not None:
        if check_removal == True and corr_select == '除去後':
            if x_axis_corr_after in numerical_column_list_after and y_axis_corr_after in numerical_column_list_after:
                #相関係数の算出method
                method_corr = 'pearson'
                method_corr = st.radio('相関係数の算出方法（散布図）', ('pearson', 'spearman', 'kendall'))
                scat_corr = scat_df.corr(method=method_corr)
                correlation_coefficient = scat_corr.iat[0,1]
            else:
                correlation_coefficient = 'カテゴリデータで相関係数は算出できません'
        else:
            if x_axis_corr in numerical_column_list and y_axis_corr in numerical_column_list:
                #相関係数の算出method
                method_corr = 'pearson'
                method_corr = st.radio('相関係数の算出方法（散布図）', ('pearson', 'spearman', 'kendall'))
                scat_corr = scat_df.corr(method=method_corr)
                correlation_coefficient = scat_corr.iat[0,1]
            else:
                correlation_coefficient = 'カテゴリデータで相関係数は算出できません'
        st.write('相関係数 : ', correlation_coefficient)
    else:
        pass
else:
    pass





if uploaded_file is not None : 
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
            sns.heatmap(heat_corr, vmin=-1.0, vmax=1.0, center=0, annot=value_display, fmt='.3f', xticklabels=heat_corr.columns.values, yticklabels=heat_corr.columns.values)
            st.pyplot(fig_heat)
    else:
        pass
else:
    pass

    
if uploaded_file is not None :
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
            #多重共線性の確認
            mul_corr_matrix = df_x_regression.corr(method='pearson')
            mul_corr_value = []
            for i in x_regression:
                mul_corr_matrix = mul_corr_matrix.sort_values(i, ascending=False)
                mul_corr_value.append(mul_corr_matrix[i].head(2).min())
            mul_corr_max = max(mul_corr_value)
            multicoll = st.checkbox('多重共線性の確認')
            if multicoll == True:
                multicoll_threshold = st.slider('多重共線性の閾値', max_value=1.0, min_value=0.5, step=0.1, value=0.7)
                if mul_corr_max >= multicoll_threshold:
                    st.write('多重共線性が認められます。説明変数間の相関を確認しましょう。')
                else:
                    st.write('多重共線性が認められませんでした。')
            else:
                pass
        else:
            pass
    else:
        pass
else:
    pass
            

if uploaded_file is not None :
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
        else:
            pass
    else:
        pass
else:
    pass