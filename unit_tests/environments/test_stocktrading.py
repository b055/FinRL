from __future__ import annotations

import numpy as np
import pytest

from finrl.meta.env_stock_trading.env_stocktrading import (
    StockTradingEnv,
)
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
import random
import pandas as pd


@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "GOOG"]


@pytest.fixture(scope="session")
def indicator_list():
    return ["open", "close", "high", "low", "volume"]


@pytest.fixture(scope="session")
def data(ticker_list):
    return YahooDownloader(
        start_date="2019-01-01", end_date="2019-02-01", ticker_list=ticker_list
    ).fetch_data()


def test_long_selling(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        previous_state=[2000, 36, 52, 2, 5, 10, 11, 10, 11],
        initial=False,
    )
    env._sell_stock(0, -2)
    np.testing.assert_almost_equal(
        env.state,
        [2073.799, 36.936, 53.73300, 0, 5, 10, 11, 10, 11],
        decimal=3,
    )


def test_long_buying_initial(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [2 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
    )
    env.step(np.array([0, 5 / hmax]))
    np.testing.assert_almost_equal(
        env.state,
        [731.066, 37.054, 53.516, 0, 5, 12, 13, 12, 13, 2, 2],
        decimal=3,
    )


def test_close_all_long_positions(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1.1
    hmax = 10
    action_space = hmax * 2 + 1
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i + 1 for i in range(len(data))]
    data["macd"] = [i + 2 for i in range(len(data))]
    data["turbulence"] = [2000 for _ in range(len(data))]
    turbulence_threshold = 1500

    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        initial_amount=1000,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        previous_state=[2000, 36, 52, 2, 5, 10, 11, 10, 11, 50, 2000],
        initial=False,
    )
    actions = np.array([-1 / hmax, 1 / hmax])
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [1983.113, 37.055, 53.516, 1, 6, 13, 14, 14, 15, 2000, 2000],
        decimal=3,
    )
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [2340.908, 36.691, 52.85950, 0, 0, 15, 16, 16, 17, 2000, 2000],
        decimal=3,
    )


def test_short_selling(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [5 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
    )
    actions = np.array([0, -2 / hmax])
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [1107.35854, 37.054, 53.516498, 0, -2, 12, 13, 12, 13, 5, 5],
        decimal=3,
    )


def test_short_selling_limit(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [5 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        initial=False,
        previous_state=[1070, 37, 53, -6, -5, 12, 13, 12, 13, 5, 5],
    )
    actions = np.array([-10 / hmax, -10 / hmax])
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [1975.787, 37.054, 53.5165, -16, -15, 12, 13, 12, 13, 5, 5],
        decimal=3,
    )


def test_short_sell_from_long_position(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [5 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        initial=False,
        previous_state=[1070, 37, 53, 1, -5, 12, 13, 12, 13, 5, 5],
    )
    actions = np.array([-10 / hmax, 0 / hmax])
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [1438.994, 37.054, 53.5165, -9, -5, 12, 13, 12, 13, 5, 5],
        decimal=3,
    )


def test_short_cover(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [5 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        previous_state=[1070, 37, 53, 1, -5, 12, 13, 12, 13, 5, 5],
        initial=False,
    )
    actions = np.array([0 / hmax, 2 / hmax])
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [962.427, 37.054, 53.51649, 1, -3, 12, 13, 12, 13, 5, 5],
        decimal=3,
    )


def test_long_buy_during_short(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [2 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        previous_state=[800, 36, 52, -7, -9, 10, 11, 10, 11, 50, 60],
        initial=False,
    )
    env.step(np.array([0, 10 / hmax]))
    # even though the action was to buy 10, it only bought 1 because of the existing short positions
    np.testing.assert_almost_equal(
        env.state,
        [262.133, 37.054, 53.516, -7.0, 1, 12.0, 13.0, 12.0, 13.0, 2.0, 2.0],
        decimal=3,
    )


def test_short_sell_during_short(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    hmax = 10
    action_space = stock_dim
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i for i in range(len(data))]
    data["macd"] = [i for i in range(len(data))]
    data["turbulence"] = [2 for _ in range(len(data))]

    turbulence_threshold = 100
    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=1000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        previous_state=[320, 36, 52, -7, 0, 10, 11, 10, 11, 50, 60],
        initial=False,
    )
    env.step(np.array([0, -10 / hmax]))
    # even though the action was to buy 10, it only bought 1 because of the existing short positions
    np.testing.assert_almost_equal(
        env.state,
        [373.679, 37.054, 53.516, -7, -1, 12, 13, 12, 13, 2, 2],
        decimal=3,
    )


def test_cover_all_short_positions(data):
    stock_dim = 2
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1.1
    hmax = 10
    action_space = hmax * 2 + 1
    initial_amount = 1000
    tech_indicator_list = ["rsi", "macd", "turbulence"]
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    data["rsi"] = [i + 1 for i in range(len(data))]
    data["macd"] = [i + 2 for i in range(len(data))]
    data["turbulence"] = [2000 for _ in range(len(data))]
    turbulence_threshold = 1500

    trade_data = data_split(
        data,
        start=data.iloc[len(data) // 4].date,
        end=data.iloc[(len(data) * 3) // 4].date,
    )
    env = StockTradingEnv(
        df=trade_data,
        stock_dim=stock_dim,
        hmax=hmax,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[buy_cost_pct] * stock_dim,
        sell_cost_pct=[sell_cost_pct] * stock_dim,
        reward_scaling=reward_scaling,
        initial_amount=1000,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        model_name="DRL",
        mode="trade",
        previous_state=[2000, 36, 52, -7, -9, 10, 11, 10, 11, 50, 2000],
        initial=False,
    )
    actions = np.array([-1 / hmax, 1 / hmax])
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [1983.113, 37.054, 53.5164, -8, -8, 13, 14, 14, 15, 2000, 2000],
        decimal=3,
    )
    env.step(actions)
    np.testing.assert_almost_equal(
        env.state,
        [1259.27, 36.691, 52.85950, 0, 0, 15, 16, 16, 17, 2000, 2000],
        decimal=3,
    )
