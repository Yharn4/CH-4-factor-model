import baostock as bs
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sqlalchemy import create_engine
from statsmodels.regression.rolling import RollingOLS

###### 获取因子数据 ######

def get_previous_trade_dates(date):
    rs = bs.query_trade_dates(start_date=date[:5]+('12' if int(date[5:7])==1 else str(int(date[5:7])-1))+'-20', end_date=date).get_data()
    return rs[rs['is_trading_day']=='1']['calendar_date'].tail(2).head(1).values[0]

def get_all_stocks_code(date):
    '''
    获取A股所有股票代码
    date必须为交易日
    '''
    
    #### 获取证券信息 ####
    rs = bs.query_all_stock(day=date)
    pre_date = get_previous_trade_dates(date)
    pre_rs = bs.query_all_stock(day=pre_date)
    # 剔除当日IPO的股票
    codes = pd.merge(rs.get_data(),pre_rs.get_data(),how='inner',on='code')
    
    # 过滤指数信息
    con = codes.code.str.contains('^bj|^sh.000|^sz.399')
    return codes[~con].reset_index(drop=True)

def get_stock_k_data(code,start_date,end_date):
    '''
    获取股票月频前复权数据
    '''
    
    rs_k = bs.query_history_k_data_plus(code,
        "date,code,close,volume,turn",
        start_date=start_date, end_date=end_date,
        frequency="m", adjustflag="2")
    k_data = rs_k.get_data()
        
    rs_peTTM = bs.query_history_k_data_plus(code,
        "date,code,peTTM",
        start_date=start_date, end_date=end_date, 
        frequency="d", adjustflag="2")
    peTTM_data = rs_peTTM.get_data()
     
    print(code,'OK')
    return pd.merge(k_data,peTTM_data,how='left',on=['date','code'])

def save_stocks_data(start_date,end_date):
    '''
    得到所有A股股票月频前复权数据并保存
    '''
    
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    # 获取股票数据
    codes = get_all_stocks_code(end_date)
    print('股票数据获取成功')
    stocks = codes['code'].apply(get_stock_k_data,start_date=start_date,end_date=end_date)
    data = pd.concat(stocks.values)
    
    #### 登出系统 ####
    bs.logout()
    
    # 保存股票数据
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    data.to_sql('stocks_data',connect,index=False,if_exists='append')
    print('保存成功')
    return data

###### 获取股票池数据 ######

def get_stocks_status(code,start_date,end_date):
    '''
    获取股票状态
    '''
    
    rs = bs.query_history_k_data_plus(code,
        "date,code,open,close,tradestatus,isST",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2")
    
    print(code,'OK')
    return rs.get_data()

def get_pool_stocks(date):
    '''
    获取沪深300和中证500成分股
    '''
    
    # 获取沪深300成分股
    hs300 = bs.query_hs300_stocks(date=date).get_data()
    zz500 = bs.query_zz500_stocks(date=date).get_data()
    
    pool = pd.concat([hs300,zz500])
    print(date,'成分股OK')
    return pool

def get_pool_change_date(start_date,end_date):
    # 获取每月最后一个交易日
    rs = bs.query_history_k_data_plus('sh.600000',
        "date",
        start_date=start_date, end_date=end_date,
        frequency="m", adjustflag="2")
    freq_m = rs.get_data()
    return freq_m

def save_pool_data(start_date,end_date):
    
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    # 获取每月最后一个交易日
    month_last_day = get_pool_change_date(start_date,end_date)
    pool_stock_update = month_last_day.apply(lambda x:get_pool_stocks(x['date']),axis=1)
    month_last_day['updateDate'] = pool_stock_update.apply(lambda x:x['updateDate'].unique()[0])
    total_stocks = pd.concat(pool_stock_update.values)
    include_stocks = Series(total_stocks['code'].unique())
    
    # pool_code = get_pool_stocks(end_date)
    stocks = include_stocks.apply(get_stocks_status,start_date=start_date, end_date=end_date)
    
    # 登出系统    
    bs.logout()
    
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    month_last_day.to_sql('pool_change_stocks_date',connect,index=False,if_exists='append')
    
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    total_stocks.to_sql('pool_all_stocks',connect,index=False,if_exists='append')
    
    pool = month_last_day.merge(pd.concat(stocks.values),how='left',on='date')
     # 保存股票数据
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    pool.to_sql('pool_data',connect,index=False,if_exists='append')
    print('保存成功')
    return pool


###### 因子数据处理 ######

def del_factor_data():
    
    # 从数据库获取股票信息
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    data = pd.read_sql('SELECT * FROM stocks_data',connect,parse_dates=['date'])
    
    # 增加因子序列
    data.replace('',np.NaN,inplace=True)
    data['close'] = data['close'].astype(np.float64)
    data['peTTM'] = data['peTTM'].astype(np.float64)
    data['MV'] = data['volume'].astype(np.float64) / data['turn'].astype(np.float64) 
    data['EP'] = data['peTTM']**(-1)
    data['TURN'] = data['turn'].astype(np.float64)
    data['profit'] = data.groupby(by='code')['close'].pct_change()
    
    # 剔除后30%股票
    data['keep'] = pd.cut(data['MV'],bins=data['MV'].quantile([0,0.3,1]),labels=list('DK'))
    data = data[data['keep']=='K']
    
    return data

def get_factor_data(data):
    '''
    得到因子预期收益率
    '''
    
    data.dropna(inplace=True)
    if data.size == 0:
        return [np.nan,np.nan,np.nan,np.nan]
    # 得到因子标签
    data['SMB'] = pd.cut(data['MV'],bins=data['MV'].quantile([0,0.5,1]),labels=list('SB'))
    data['VMG'] = pd.cut(data['EP'],bins=data['EP'].quantile([0,0.3,0.7,1]),labels=list('GMV'))
    data['PMO'] = pd.cut(data['TURN'],bins=data['TURN'].quantile([0,0.3,0.7,1]),labels=list('OMP'))
    
    # 计算组合预期收益率
    # data['profit'] = data['profit'].clip(data['profit'].mean()-3*data['profit'].std(),data['profit'].mean()+3*data['profit'].std())
    smb_vmg = data.groupby(by=['SMB','VMG']).apply(lambda x:np.average(x['profit'],weights=x['MV']))
    smb_pmo = data.groupby(by=['SMB','PMO']).apply(lambda x:np.average(x['profit'],weights=x['MV']))
    
    # 计算因子预期收益率
    mkt = np.average(data['profit'],weights=data['MV'])
    smb_value = (smb_vmg.loc['S'].sum() - smb_vmg.loc['B'].sum()) / 3
    smb_turnover = (smb_pmo.loc['S'].sum() - smb_pmo.loc['B'].sum()) / 3
    smb = (smb_value + smb_turnover) / 2
    vmg = (smb_vmg.loc[:,'V'].sum() - smb_vmg.loc[:,'G'].sum()) / 2
    pmo = (smb_pmo.loc[:,'P'].sum() - smb_pmo.loc[:,'O'].sum()) / 2
    
    return [mkt,smb,vmg,pmo]

def save_factor_data():
    data = del_factor_data()
    factor = data.groupby(by='date').apply(get_factor_data).apply(Series)
    factor.dropna(inplace=True)
    factor.columns=('mkt','smb','vmg','pmo')
    con = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    factor.reset_index().to_sql('factor_data',con,index=False,if_exists='append')
    
    return factor

###### 股票池股票清洗 ######

def get_pool_data():
    '''
    得到股票池数据
    '''
    
    # 从数据库获取股票信息
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    pool = pd.read_sql_query('SELECT * FROM pool_data',connect,parse_dates=['date'],dtype={'open':np.float64,'close':np.float64})
    pool
    
    # 剔除涨跌停、停牌、ST股的数据
    condition = pd.concat([pool.apply(lambda x:x['open']*0.9<x['close']<x['open']*1.1,axis=1),pool['tradestatus']=='1',pool['isST']=='0'],axis=1)
    pool = pool[condition.apply(all,axis=1)]
    
    # 计算收益率
    pool['profit'] = pool.groupby(by='code').close.pct_change()
    pool.dropna(inplace=True)
    pool.reset_index(drop=True,inplace=True)
    
    # 保存股票池数据
    con = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    pool.reset_index().to_sql('del_pool',con,index=False,if_exists='append')
    return pool


###### 多因子选股 ######

def rolling_reg(factors,factor):
    if factors.shape[0] >= 12 :
        mod = RollingOLS(np.array(factors['profit']), np.array(factors[factor]),window=12)
        regs = mod.fit()
        return regs.params
    else:
        return np.nan

def choice_stocks():
    # 从数据库获取因子信息
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    factor = pd.read_sql_query('SELECT * FROM factor_data',connect,parse_dates=['date'])
    # 从数据库获取股票池信息
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    pool = pd.read_sql_query('SELECT * FROM del_pool',connect,parse_dates=['date'])
    
    # 连接因子和股票池数据
    data = factor.merge(pool[['date','updateDate','code','profit','close']],how='left',on='date')
    data['conf'] = 1
    
    alpha = data.groupby(by='code').apply(lambda x:rolling_reg(x,['conf','mkt', 'smb', 'vmg', 'pmo']))
    alpha.dropna(inplace=True)
    data = pd.merge(data,alpha.reset_index(),how='inner',on='code')
    data['rank'] = data.groupby(by='code').apply(lambda x:x['date'].rank(method='first')).reset_index(level=0,drop=True).astype(np.int16)
    data['alpha'] = data.apply(lambda x:x[0][x['rank']-1],axis=1).str[0]
    alpha_data = data[['date','updateDate','code','alpha','profit','close']]
    alpha_data.dropna(inplace=True)
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    alpha_data.to_sql('regress_data',connect,index=False,if_exists='append')
    return alpha_data
    
# 分别获取alpha最大和最小的10条数据
def get_trade_sig(alpha_data):
    '''
    分别选出alpha最大和最小的前10只股票
    '''
    
    sig = alpha_data.sort_values(by='alpha')
    # sig = alpha[alpha.apply(lambda x:alpha.mean()-3*alpha.std()<x<alpha.mean()+3*alpha.std())].sort_values()
    buy = sig[sig['alpha']<0].head(10)
    sell = sig[sig['alpha']>0].tail(10)
    sig = pd.concat([buy,sell])
    # signal = pd.merge(sig,alpha_data,how='inner',left_index=True,right_index=True)
    return sig

def trade_sig_append_sell_info():
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    sig = pd.read_sql('SELECT * FROM regress_data',connect,parse_dates=['date'])
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    pool = pd.read_sql('SELECT * FROM pool_all_stocks',connect,parse_dates=['date'])
    connect = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    del_pool = pd.read_sql('SELECT * FROM del_pool',connect,parse_dates=['date'])
    print('从数据库取出成功')
    signal = pd.merge(sig,pool,on=['updateDate','code'])
    signal['code_rank'] = signal.groupby(by=['date','code'])['code'].rank(method='first')
    trade_sig = signal[signal['code_rank']==1].groupby(by='date').apply(get_trade_sig)
    trade_sig.reset_index(level=0,inplace=True,drop=True)
    print('连接交易信号与股票池成功')
    def get_sell_profit(detail):
        all_profit_info = del_pool[['date','code','profit']][detail['code']==del_pool['code']]
        try:
            detail['sell_profit'] = all_profit_info[all_profit_info['date']>=detail['date']].head(2)['profit'].values[-1]
            detail['sell_date'] = all_profit_info[all_profit_info['date']>=detail['date']].head(2)['date'].values[-1]
        except:
            detail['sell_profit'] = np.nan
            detail['sell_date'] = np.nan
            print(detail['code'].values,'异常')
        return detail
    trade_sig = trade_sig.apply(get_sell_profit,axis=1)
    print('OK')
    return trade_sig
  
###### 获得沪深300和中证500的收益数据 ######

def get_base_info():
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    rs = bs.query_history_k_data_plus("sh.000300",
        "date,code,close",start_date='2009-12-01', end_date='2022-08-01', frequency="m")
    hs300_info = rs.get_data()
    rs = bs.query_history_k_data_plus("sh.000905",
        "date,code,close",start_date='2009-12-01', end_date='2022-08-01', frequency="m")
    zz500_info = rs.get_data()

    # 登出系统
    bs.logout()
    hs300_info['profit'] = hs300_info['close'].astype(np.float64).pct_change()
    zz500_info['profit'] = zz500_info['close'].astype(np.float64).pct_change()
    base = pd.concat([hs300_info,zz500_info])
    # hs300_info.dropna(inplace=True)
    # zz500_info.dropna(inplace=True)
    base.dropna(inplace=True)
    con = create_engine("mysql+pymysql://root:981106@localhost:3306/stocks?charset=utf8")
    base.to_sql('base_data',con,index=False,if_exists='append')
    return base
