import os
import shutil
import re
import gzip
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sqlite3
import os
import pytz
import json
from alpha_generate_features import GenerateFeatures as AlphaGen
import requests

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')

from analogs.simple import NearestNeighborsPartitioned, date_filter, date_not_class_filter, more_dist_same_class_filter

import flask
from flask import Flask, send_file

PORT = 5000

default_path = ''

gen = AlphaGen()

with gzip.open(os.path.join('models', 'partition_searcher.pkl.gz'), 'rb') as f:
    partition_searcher = pickle.load(f)
partition_searcher.set_merge_filter(date_filter)
    
with gzip.open(os.path.join('models', 'not_analogs_searcher.pkl.gz'), 'rb') as f:
    not_analogs_searcher = pickle.load(f)
not_analogs_searcher.set_merge_filter(date_not_class_filter)
    
with gzip.open(os.path.join('models', 'dist_same_class_searcher.pkl.gz'), 'rb') as f:
    dist_same_class_searcher = pickle.load(f)
dist_same_class_searcher.set_merge_filter(more_dist_same_class_filter)

with open(os.path.join('models', 'price_model.pkl'), 'rb') as gfile:
    price_model = pickle.load(gfile)

df_gis_firms = pd.read_pickle('data/df_gis_firms.pkl', compression='gzip')
df_gis_house = pd.read_pickle('data/df_gis_house.pkl', compression='gzip')
df_gis_platforms = pd.read_pickle('data/df_gis_platforms.pkl', compression='gzip')
df_metro = pd.read_excel('data/metro_station.xlsx', engine='openpyxl')
with gzip.open('data/region_features.pkl', 'rb') as f:
    df_gis_city = pickle.load(f)

df_gis_city = df_gis_city[['Широта', 'Долгота']].rename(columns={'Долгота': 'longitude', 'Широта': 'latitude'})
df_gis_city['poi'] = 'Центр города'
df_gis = pd.concat([df_gis_firms, df_gis_house, df_gis_platforms, df_gis_city])

def construct_request(req):
    res = pd.DataFrame.from_dict(req['constructionCorps'])
    for key, value in req.items():
        if key == 'competitors':
            continue
        if key != 'constructionCorps':
            res[key] = value
            
    return res

def construct_current(req):
    res = pd.DataFrame.from_dict(req['constructionCorps'])
    for key, value in req.items():
        if key != 'constructionCorps':
            res[key] = value
    
    res['startDate'] = pd.to_datetime(res['startDate'])
    res['endDate'] = pd.to_datetime(res['endDate'])
    res['startDateSales'] = pd.to_datetime(res['startDateSales'])
    
    dates = []
    for i, row in res.iterrows():
        quarters = pd.date_range(row['startDateSales'], row['endDate'], freq='QS')
        ready = np.linspace(0, 1.07, len(quarters))
        quarters = [{
            'corpNumber': row['corpNumber'], 
            'date': q, 
            'readiness': r
        } for q, r in zip(quarters, ready)]
        dates.extend(quarters)
        
    dates = pd.DataFrame.from_dict(dates)
    
    res = res.merge(dates, on='corpNumber')
            
    return res

def rename_col(colname):
    colname = re.sub('^all\_', '', colname)
    colname = re.sub('\_analog', '', colname)
    
    return colname

def get_quarter_start(qstr):
    year, q = qstr.split('-')
    q = int(q)
    return '{}-{}-01'.format(year, str((q-1) * 3+1).zfill(2))

def my_logging(req, res, method):
    dt = datetime.now(pytz.timezone('Europe/Moscow')).strftime('%Y-%m-%d %H:%M:%S')
    id_req = datetime.now(pytz.timezone('Europe/Moscow')).strftime('%Y%m%d%H%M%S')
    req = str(req)
    res = str(res)

    log = {}
    log['id_req'] = id_req
    log['dt'] = dt
    log['req'] = req
    log['res'] = res
    log['method'] = method

    df = pd.DataFrame.from_records([log], index='id_req')

    conn = sqlite3.connect('db_logs.db')
    df.to_sql('requests', con=conn, if_exists='append', index='id_req')
    return id_req

def types_handler(x):
    """
    helper для функции generate_json
    :param x:
    :return:
    """
    if isinstance(x, np.datetime64) or isinstance(x, pd.Timestamp):
        return str(x)
    elif isinstance(x, np.int64) or isinstance(x, np.int32):
        return int(x)
    else:
        return x
    
def not_nan(x):
    """
    проверка на непустое значение
    :param x:
    :return:
    """
    return x == x

def generate_json(df, yes_no=True):
    """
    создание объекта json из таблицы
    :param df: таблица для трансформации
    :return:
    """
    df_json = []
    for row in range(len(df)):
        df_temp = {}
        for column in df.columns:
            if not_nan(df[column].iloc[row]):
                df_temp[column] = types_handler(df[column].iloc[row])
                if yes_no:
                    if df_temp[column] == 'Да':
                        df_temp[column] = True
                    elif df_temp[column] == 'Нет':
                        df_temp[column] = False
        df_json.append(df_temp)
    return df_json

def get_last10_logs():
    conn = sqlite3.connect('db_logs.db')
    df = pd.read_sql('SELECT * from requests', con=conn)
    df = df.tail(10)
    return df

app = Flask(__name__)

@app.route("/get_analogs", methods=["GET", "POST"])
def get_analogs():
    request = flask.request.json

    req = construct_request(request)
    req['all_city'] = req['city']
    req['all_classProperty_all_city'] = req['classProperty'] + '_' + req['all_city']

    now = pd.to_datetime(datetime.now())
    req['date'] = pd.to_datetime(get_quarter_start('{}-{}'.format(now.year, now.quarter)))
    
    analogs = partition_searcher.transform(req, 'latitude', 'longitude')
    not_analogs = not_analogs_searcher.transform(req, 'latitude', 'longitude')
    dist_same_class = dist_same_class_searcher.transform(req, 'latitude', 'longitude')
    
    analogs['is_competitors'] = 1
    not_analogs['is_competitors'] = 0
    dist_same_class['is_competitors'] = 0
    
    result = pd.concat([
        analogs, not_analogs, dist_same_class
    ])
    
    max_obj_date = result.groupby('all_ID_obj_analog')['all_endDate_analog'].max().reset_index()
    result = result.merge(max_obj_date, on=['all_ID_obj_analog', 'all_endDate_analog'])

    result = result.drop_duplicates(['all_corpNumber_analog'])

    analog_cols = [col for col in result.columns if col.endswith('_analog')]

    result = result[analog_cols + ['is_competitors']]
#     result['comment'] = None
#     result['oldReportID'] = None
    cols_replace = {col: rename_col(col) for col in result.columns}
    result = result.rename(columns=cols_replace)

    result = result.drop('corpNumber', 1)
    
    result['parking'] = result['parking'].map({0: 'Нет', 1: 'Есть'})

    int_columns = ['floorsMax', 'countApartments', 'squareCorp', 'is_competitors']
    for column in int_columns:
        result[column] = result[column].fillna(9999).astype('int')
    float_columns = ['readiness', 'percent_1_room', 'percent_2_room', 'percent_3_room', 'gis_dist_to_nearest_Metro',
                     'price_current', 'sales_average']
    for column in float_columns:
        result[column] = round(result[column], 1)
            
    result = {'competitors': generate_json(result)}

    id_req = my_logging(request, result, 'get_analogs')
    result['id_req'] = id_req
    
    return result

@app.route("/run_prediction", methods=["GET", "POST"])
def run_prediction():
    req = flask.request.json

    if ('current' not in req.keys()):# and ('competitors' not in input_json.keys()):
        return {'result': 'ERROR', 'message': 'wrong input format'}

    elif 'competitors' not in req:
        resp = requests.post('http://localhost:{}/get_analogs'.format(PORT), json=req['current']).json()
        competitors = resp['competitors']

    else:
        competitors = req['competitors']

    current = req['current']

    df = construct_current(current)
    pairs = pd.DataFrame.from_dict(competitors)
    
    competitors = pairs[pairs['is_competitors'] == 1].drop('is_competitors', 1)

    #TODO: FIX json format (add ans_num and dist)
    competitors['ans_num'] = list(range(len(competitors)))

    if len(competitors) == 0:
        return {'result': 'ERROR', 'message': 'zero competitors'}
    stop_cols = ['ans_num', 'dist']
    competitors.columns = [col + '_analog' if col not in stop_cols else col for col in competitors.columns]
    
    competitors['_merge_col'] = 1
    df['_merge_col'] = 1

    data = df.merge(competitors, on='_merge_col').drop('_merge_col', 1)
    
    data['ID_obj'] = data['projectName']
    stop_cols = ['date', 'ans_num', 'dist']
    data.columns = [col if col.startswith('all_') or col in stop_cols else 'all_' + col for col in data.columns]

    # ----------
    # move this chunk to price_model
    data['all_lat'] = data['all_latitude'].astype(float)
    data['all_lng'] = data['all_longitude'].astype(float)

    data['all_year'] = data['date'].dt.year
    data['all_quarter'] = data['date'].dt.quarter

    #TODO:FIX to date of competitor price
    data['all_year_analog'] = datetime.now().year
    data['all_quarter_analog'] = (datetime.now().month // 3) + 1

    #TODO:FIX to classProperty or use Andrey's model
    data['all_classCalc'] = data['all_classCalc_analog']

    #TODO:include corpNumber in analog info
    data['all_corpNumber_analog'] = data['all_name_corp_analog']

    #TODO: remove from model
    data['all_sumamount_livingapartmrnt'] = np.nan

    data['all_price_fact_analog'] = data['all_price_current_analog']

    data['all_percent_1_room'] = data['all_countOneRoom'] / data['all_countApartments']
    data['all_percent_2_room'] = data['all_countTwoRoom'] / data['all_countApartments']
    data['all_percent_3_room'] = data['all_countThreeRoom'] / data['all_countApartments']
    data['all_percent_4_room'] = data['all_countFourRoom'] / data['all_countApartments']

    data['all_flat_type'] = data['all_isApartments'].replace({True: 'апартамент', False: 'квартира'})

    data['all_rooms'] = np.dot(data[['all_countOneRoom', 'all_countTwoRoom', 'all_countThreeRoom', 'all_countFourRoom']].values, 
                                np.arange(1, 5).T) / data['all_countApartments']

    data = gen.generate_gis_features(df_gis, df_metro, data, radius=2)

    data['all_region'] = data['all_region_analog']

    # ------------

    analogs_prices = price_model.get(data)

    result = analogs_prices.groupby(['all_corpNumber', 'all_startDate', 'all_endDate', 'date'])['ans'].mean().reset_index()
    # result = result[result['date'].between(result['all_startDate'], result['all_endDate'])].reset_index(drop=True)
    result = result.rename(columns={'ans': 'price'})
    result['sales'] = np.nan
    
    max_date = result.groupby('all_corpNumber')['date'].max().reset_index()
    ugs = result.merge(max_date, on=['all_corpNumber', 'date'])
    ugs = ugs[['all_corpNumber', 'price']].\
            drop_duplicates(subset=['all_corpNumber'])
    ugs = ugs.rename(columns={'price': 'price_ugs'})

    result = result.merge(ugs, on=['all_corpNumber'])
    result['discount'] = (result['price_ugs'] - result['price']) / result['price_ugs']

    result['Квартал'] = result['date'].dt.year.astype(str) + 'Q' + result['date'].dt.quarter.astype(str)

    result = result.rename(columns={'price': 'Цена', 'discount': 'Дисконт к цене УГС', 'sales': 'Продажи'})

    archive_path = os.path.join(default_path , 'output', req['idReport'])
    path = os.path.join(archive_path, 'результаты прогноза')
    os.makedirs(path, exist_ok=True)
    # analogs_prices.to_excel(os.path.join(path, '{}.xlsx'.format(input_json['idReport'])))

    writer = pd.ExcelWriter(os.path.join(path, 'Расчет_{}.xlsx'.format(req['idReport'])), engine='xlsxwriter')

    for corp in result['all_corpNumber'].unique():
        data_corp = result.loc[result['all_corpNumber'] == corp, ['Квартал', 'Цена', 'Продажи', 'Дисконт к цене УГС']].sort_values('Квартал')
        data_corp.T.to_excel(writer, sheet_name='Корпус-'+str(corp))

    writer.save()

    df.to_excel(os.path.join(path, 'Объект_для_прогноза_{}.xlsx'.format(req['idReport'])), index=False)
    pairs.to_excel(os.path.join(path, 'Кандидаты_в_аналоги_{}.xlsx'.format(req['idReport'])), index=False)

    with open('data/mock.json', 'r') as f:
        resp = json.load(f)
    # os.system('sh copyReport.sh {}'.format(req['idReport']))

    id_req = my_logging(req, resp, 'run_prediction')
    resp['id_req'] = id_req
    
    return resp

@app.route("/")
def hello():
    import codecs
    #     f = codecs.open("readme.html",'r')
    #     return f.read()
    print('check API')
    return {'status': 'OK'}

@app.route("/logs", methods=["GET"])
def logs():
    result = get_last10_logs()
    result = result.sort_values('dt', ascending=False).to_json(orient='records', force_ascii=False)
    return result


@app.route('/download_logs', methods=["GET", "POST"])
def download_logs():
    return send_file('db_logs.db')


@app.route("/download_report", methods=["GET", "POST"])
def download_report():
    input_json = flask.request.json
    path = default_path + 'output/' + input_json['idReport']
    shutil.make_archive(path, 'zip', path)

    try:
        msg = 'Success'
        return send_file(path+'.zip')
    except Exception as ex:
        msg = str(ex)
    finally:
        id_req = my_logging({'path': path, 'msg': msg}, resp, 'download_report')
        os.remove(path+'.zip')


@app.route("/download_all_reports", methods=["GET", "POST"])
def download_all_reports():
    path = default_path + 'output'
    shutil.make_archive(path, 'zip', path)
    try:
        return send_file(path+'.zip')
    finally:
        os.remove(path+'.zip')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)