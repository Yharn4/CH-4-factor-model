from function import *

sig = choice_stocks()
sig


def get_trade_sig(alpha_data):
    '''
    分别选出alpha最大和最小的前10只股票
    '''

    alpha = alpha_data['alpha']
    sig = alpha[
        alpha.apply(lambda x: alpha.mean() - 3 * alpha.std() < x < alpha.mean() + 3 * alpha.std())].sort_values()
    buy = sig[sig < 0].head(10)
    sell = sig[sig > 0].tail(10)
    sig = pd.concat([buy, sell])
    signal = pd.merge(sig, alpha_data, how='inner', left_index=True, right_index=True)
    return signal

trade_sig = sig.groupby(by='date').apply(get_trade_sig)
trade_sig.reset_index(level=0,inplace=True,drop=True)
trade_sig.drop(columns='alpha_y',inplace=True)
trade_sig


def rebalancing(sig):
    global trade_pool
    try:
        pool = trade_pool.tail(15)[trade_pool['tag'] == 'B']
    except:
        pool = trade_pool.copy()
    # 将所有的股票平仓
    # sell = sig[sig['alpha_x']>0]
    selling = pd.merge(pool, sig, on='code')
    selling['cash_add'] = selling['amount'] * (selling['profit'] + 1)
    selling['cash'] = selling['cash'] + (selling['cash_add'].sum())
    selling['balancing_return'] = (selling['amount'] * selling['profit']).sum()
    # selling['balancing_return_rate'] = selling['balancing_return'] / selling['assets']
    selling['assets'] = selling['assets'] + selling['balancing_return']
    selling['amount'] = 0
    selling['tag'] = 'S'

    # 对alpha为负的股票按权重补仓
    buying = sig[sig['alpha_x'] < 0]
    buying['weight'] = buying['alpha_x'] / (buying['alpha_x'].sum())
    buying['amount'] = pool['assets'].iloc[-1] * buying['weight'] / buying['close'] // 100 * 100
    buying['cash_add'] = buying['amount'] * buying['close'] * (-1)
    buying['cash'] = pool['cash'].iloc[-1] + (buying['cash_add'].sum())
    buying['assets'] = pool['assets'].iloc[-1]
    buying['tag'] = 'B'

    total = pd.concat([selling, buying])
    trade_pool = pd.concat([trade_pool, total])
    print(trade_pool)
    return trade_pool

trade_pool = DataFrame([['',0,1000000,1000000]],columns=['code','amount','cash','assets'])
trade_sig.groupby(by='date').apply(rebalancing)

trade_pool