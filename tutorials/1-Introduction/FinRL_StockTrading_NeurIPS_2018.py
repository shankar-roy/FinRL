# matplotlib.use('Agg')
import datetime
import glob
import itertools
import os

import pandas as pd

from finrl import config
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_stats, backtest_plot, get_baseline


# sys.path.append("../FinRL-Library")
def generate_data():
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    quotes_filename = "." + os.sep + config.DATA_SAVE_DIR + os.sep + "quotes.csv"

    # from config.py start_date is a string, not provided now
    # config.START_DATE

    # from config.py end_date is a string, not provided now
    # config.END_DATE

    print(config_tickers.DOW_30_TICKER)

    try:
        df = pd.read_csv(quotes_filename)
    except FileNotFoundError as e:
        print("File not found, need to download data...")
        df = YahooDownloader(start_date='1970-01-01',
                             end_date='2022-07-31',
                             ticker_list=config_tickers.DOW_30_TICKER).fetch_data()
        df.sort_values(['date', 'tic'], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        # df.set_index('date')
        df.to_csv(quotes_filename)

    print(df.head())

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False)

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])

    processed_full = processed_full.fillna(0)
    processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)
    processed_full.set_index('date')

    train = data_split(processed_full, '1970-01-01', '2020-12-31')
    trade = data_split(processed_full, '2022-01-01', '2022-07-31')
    print(len(train))
    print(len(trade))
    # %%
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(config.INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        # initial_amount = 1000000, and hold no shares at beginning.
        # "initial_list": [1000000] + [0 for i in range(stock_dimension)],
        "initial_amount": 100_000,
        "num_stock_shares": [0] * stock_dimension,
        # buy and sell cost for each stock
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

#    trained_sac = agent.train_model(model=model_sac,
#                                    tb_log_name='sac',
#                                    total_timesteps=60000)

    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }

    model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=100_000)

    data_risk_indicator = processed_full[(processed_full.date < '2022-01-01') & (processed_full.date >= '1990-01-01')]
    insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])

    print(insample_risk_indicator.vix.describe())

    print(insample_risk_indicator.vix.quantile(0.996))

    print(insample_risk_indicator.turbulence.describe())

    print(insample_risk_indicator.turbulence.quantile(0.996))

    # trade = data_split(processed_full, '2020-07-01','2021-10-31')
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    # env_trade, obs_trade = e_trade_gym.get_sb_env()
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_ppo,
        environment=e_trade_gym)

    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    df_account_value.to_csv("." + os.sep + config.RESULTS_DIR + os.sep + "account_value_" + now + '.csv')
    df_actions.to_csv("." + os.sep + config.RESULTS_DIR + os.sep + "actions_" + now + '.csv')
    print("==============Get Backtest Results===========")

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("." + os.sep + config.RESULTS_DIR + os.sep + "perf_stats_all_" + now + '.csv')
    return df_account_value


def plot_account_value(df_account_value):
    # baseline stats
    print("==============Get Baseline Stats===========")
    print(df_account_value.tail())

    baseline_df = get_baseline(
        ticker="^DJI",
        start=df_account_value.loc[0, 'date'],
        end=df_account_value.loc[len(df_account_value)-1, 'date'])

    stats = backtest_stats(baseline_df, value_col_name='close')

    df_account_value.loc[0,'date']
    df_account_value.loc[len(df_account_value)-1,'date']
    print("==============Compare to DJIA===========")

    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                  baseline_ticker='^DJI',
                  baseline_start=df_account_value.loc[0, 'date'],
                  baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])


account_value_filename = "." + os.sep + config.RESULTS_DIR + os.sep + "account_value_*.csv"
for filename in glob.iglob(account_value_filename):
    account_value = pd.read_csv(filename)

try:
    account_value
except NameError:
    account_value = generate_data()
finally:
    plot_account_value(account_value)
