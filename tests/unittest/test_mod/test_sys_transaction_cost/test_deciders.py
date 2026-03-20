import pytest
import pandas as pd
from unittest.mock import MagicMock

from rqalpha.interface import TransactionCostArgs
from rqalpha.const import SIDE, POSITION_EFFECT, INSTRUMENT_TYPE
from rqalpha.mod.rqalpha_mod_sys_transaction_cost.deciders import (
    HKStockTransactionCostDecider,
    USStockTransactionCostDecider
)


@pytest.fixture
def mock_instrument():
    instrument = MagicMock()
    instrument.type = INSTRUMENT_TYPE.CS
    return instrument


def create_args(instrument, price, quantity, order_id=None, side=SIDE.BUY):
    return TransactionCostArgs(
        instrument=instrument,
        price=price,
        quantity=quantity,
        side=side,
        position_effect=POSITION_EFFECT.OPEN,
        order_id=order_id
    )


class TestHKStockTransactionCostDecider:
    def test_calc_single_trade_no_order_id(self, mock_instrument):
        decider = HKStockTransactionCostDecider()
        
        # 1. 小额交易，触发最低收费
        # 佣金 = max(10 * 100 * 0.0003 = 0.3, 3) = 3
        # 印花税 = max(10 * 100 * 0.001 = 1, 1) = 1
        # 平台使用费 = 15
        args = create_args(mock_instrument, price=10.0, quantity=100)
        cost = decider.calc(args)
        assert cost.commission == 3.0
        assert cost.tax == 1.0
        assert cost.other_fees == 15.0
        assert cost.total == 19.0

        # 2. 大额交易，按比例收费
        # 佣金 = max(100 * 10000 * 0.0003 = 300, 3) = 300
        # 印花税 = max(100 * 10000 * 0.001 = 1000, 1) = 1000
        # 平台使用费 = 15
        args2 = create_args(mock_instrument, price=100.0, quantity=10000)
        cost2 = decider.calc(args2)
        assert cost2.commission == 300.0
        assert cost2.tax == 1000.0
        assert cost2.other_fees == 15.0

    def test_calc_multiple_trades_same_order(self, mock_instrument):
        decider = HKStockTransactionCostDecider()
        order_id = 1001

        # 第一笔交易，金额很小，收取全部最低费用
        # 产生佣金0.3 < 3，印花税1.0 == 1，收取最低3和1，平台费15
        args1 = create_args(mock_instrument, price=10.0, quantity=100, order_id=order_id)
        cost1 = decider.calc(args1)
        assert cost1.commission == 3.0
        assert cost1.tax == 1.0
        assert cost1.other_fees == 15.0

        # 第二笔交易，金额很大
        # 佣金: 此前剩余2.7，新产生300，收 300 - 2.7 = 297.3?
        # 等等，如果第一次产生0.3，但是提前收了3，那么这里应该是 cost_commission(300) > commission(3), 收300 - 3 = 297
        # 根据逻辑，commission_map[order_id]初始化为3
        # 第一笔 cost = 0.3 <= 3, 收取3, map[order_id] -= 0.3 -> 2.7
        # 第二笔 cost = 300 > 2.7, map不等于3，收 300 - 2.7 = 297.3，map归0
        args2 = create_args(mock_instrument, price=100.0, quantity=10000, order_id=order_id)
        cost2 = decider.calc(args2)
        assert pytest.approx(cost2.commission) == 297.3
        
        # 第一笔印花税 cost=1.0 <= 1.0, 收取1.0, map[order_id] -= 1.0 -> 0.0
        # 第二笔印花税 cost=1000 > 0.0, map!=1.0, 收 1000 - 0 = 1000
        assert cost2.tax == 1000.0
        # 平台费不重复收
        assert cost2.other_fees == 0.0

    def test_batch_estimate(self):
        decider = HKStockTransactionCostDecider()
        delta_quantities = pd.Series([100, -10000])
        prices = pd.Series([10.0, 100.0])
        
        costs = decider.batch_estimate(delta_quantities, prices)
        
        # 1. qty=100, price=10 -> cost_money=1000
        # comm=max(1000*0.0003=0.3, 3)=3
        # tax=max(1000*0.001=1, 1)=1
        # platform=15
        # total=19
        assert costs.iloc[0] == 19.0
        
        # 2. qty=-10000, price=100 -> cost_money=1000000
        # comm=max(1000000*0.0003=300, 3)=300
        # tax=max(1000000*0.001=1000, 1)=1000
        # platform=15
        # total=1315
        assert costs.iloc[1] == 1315.0


class TestUSStockTransactionCostDecider:
    def test_calc_single_trade_no_order_id(self, mock_instrument):
        decider = USStockTransactionCostDecider()
        
        # 1. 小数量，触发最低收费
        # 佣金 = max(100 * 0.0049 = 0.49, 0.99) = 0.99
        # 平台使用费 = max(100 * 0.005 = 0.5, 1.0) = 1.0
        # 交收费 = 100 * 0.003 = 0.3
        # other_fees = 1.0 + 0.3 = 1.3
        # tax = 0
        args = create_args(mock_instrument, price=10.0, quantity=100)
        cost = decider.calc(args)
        assert cost.commission == 0.99
        assert cost.tax == 0.0
        assert cost.other_fees == 1.3

        # 2. 大数量，按比例收费
        # 佣金 = max(10000 * 0.0049 = 49.0, 0.99) = 49.0
        # 平台使用费 = max(10000 * 0.005 = 50.0, 1.0) = 50.0
        # 交收费 = 10000 * 0.003 = 30.0
        # other_fees = 80.0
        args2 = create_args(mock_instrument, price=100.0, quantity=10000)
        cost2 = decider.calc(args2)
        assert cost2.commission == 49.0
        assert cost2.tax == 0.0
        assert cost2.other_fees == 80.0

    def test_calc_multiple_trades_same_order(self, mock_instrument):
        decider = USStockTransactionCostDecider()
        order_id = 2001

        # 第一笔交易，小数量
        # 佣金：cost=0.49 <= 0.99，收0.99，剩余0.5
        # 平台费：cost=0.5 <= 1.0，收1.0，剩余0.5
        # 交收费：0.3
        args1 = create_args(mock_instrument, price=10.0, quantity=100, order_id=order_id)
        cost1 = decider.calc(args1)
        assert cost1.commission == 0.99
        assert cost1.other_fees == 1.3

        # 第二笔交易，大数量 (9900股)
        # 佣金：cost=9900*0.0049=48.51 > 0.5，收48.51-0.5=48.01
        # 平台费：cost=9900*0.005=49.5 > 0.5，收49.5-0.5=49.0
        # 交收费：9900*0.003=29.7
        # other_fees: 49.0 + 29.7 = 78.7
        args2 = create_args(mock_instrument, price=100.0, quantity=9900, order_id=order_id)
        cost2 = decider.calc(args2)
        assert pytest.approx(cost2.commission) == 48.01
        assert pytest.approx(cost2.other_fees) == 78.7

    def test_batch_estimate(self):
        decider = USStockTransactionCostDecider()
        delta_quantities = pd.Series([100, -10000])
        prices = pd.Series([10.0, 100.0])
        
        costs = decider.batch_estimate(delta_quantities, prices)
        
        # 1. qty=100
        # comm=max(100*0.0049=0.49, 0.99)=0.99
        # platform=max(100*0.005=0.5, 1.0)=1.0
        # settle=100*0.003=0.3
        # total=0.99+1.0+0.3=2.29
        assert pytest.approx(costs.iloc[0]) == 2.29
        
        # 2. qty=-10000
        # comm=max(10000*0.0049=49, 0.99)=49
        # platform=max(10000*0.005=50, 1.0)=50
        # settle=10000*0.003=30
        # total=49+50+30=129
        assert pytest.approx(costs.iloc[1]) == 129.0
