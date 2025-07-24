# -*- coding: utf-8 -*-
"""
High-Performance Parallel Backtesting Script

This script refactors the original notebook code into a modular, command-line-driven
tool for running multiple backtesting strategies in parallel, leveraging multi-core CPUs.

Key Features:
- Command-line interface for easy configuration.
- Parallel execution of strategies using concurrent.futures for significant speedup.
- Handles multiple strategies in a single run and consolidates results.
- Modular functions for clarity and maintainability.

How to Run from your terminal:
-------------------------------
python your_script_name.py \
    --holdings-file "E:/PBROE/ch6/PBROE_5.0_from_3.1_TS_10M_holdings_with_vol.csv" \
    --returns-file "E:/PBROE/data/TRDNEW_Mnth.csv" \
    --benchmark-file "E:/PBROE/data/benchmark_indices.csv" \
    --output-dir "E:/PBROE/ch6/backtest_results" \
    --volatility-cols roe_vol_2y roe_vol_3y roe_vol_5y \
    --start-date "2010-05-01" \
    --end-date "2025-04-30" \
    --benchmark-code "000300" \
    --risk-free-rate 0.03 \
    --quantile 0.5

"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- 核心回测逻辑函数 ---

def build_portfolio(holdings_df: pd.DataFrame, vol_col: str, quantile: float) -> dict:
    """
    为单个波动率因子构建投资组合。

    Args:
        holdings_df (pd.DataFrame): 包含所有持仓和波动率数据的DataFrame。
        vol_col (str): 用于筛选的单一波动率列名。
        quantile (float): 筛选的分位数阈值。

    Returns:
        dict: {调仓日期: {股票代码集合}}
    """
    # 移除特定波动率数据缺失的行
    df_filtered = holdings_df.dropna(subset=[vol_col]).copy()

    selections_dict = {}
    for date, group in df_filtered.groupby('调入日期'):
        try:
            threshold = group[vol_col].quantile(quantile)
            selected_stocks = set(group[group[vol_col] <= threshold]['stkcd'])
            if selected_stocks:
                selections_dict[date] = selected_stocks
        except Exception:
            # 如果分组数据为空或无法计算分位数，则跳过
            continue

    return selections_dict


def run_single_strategy_backtest(
        strategy_name: str,
        selections: dict,
        returns_df: pd.DataFrame,
        start_date: str,
        end_date: str
) -> pd.Series:
    """
    对单个策略执行向量化回测。

    Args:
        strategy_name (str): 策略的名称。
        selections (dict): 该策略的持仓字典。
        returns_df (pd.DataFrame): 所有股票的月度收益率数据。
        start_date (str): 回测开始日期。
        end_date (str): 回测结束日期。

    Returns:
        pd.Series: 包含该策略每月收益率的Series。
    """
    backtest_months = pd.to_datetime(pd.date_range(start_date, end_date, freq='MS'))

    # 将调仓日的持仓映射到每个交易月份
    portfolio_map = pd.Series(index=backtest_months, dtype='object')
    rebalance_dates = sorted(selections.keys())

    # 使用 pd.cut 实现高效的区间查找，比循环更快
    if rebalance_dates:
        # 创建一个Series，索引是调仓日，值是持仓集合
        selections_series = pd.Series(selections)
        # 使用 `reindex` 和 `ffill` 将每个回测月份的持仓填充为最近的一个调仓日持仓
        portfolio_map = selections_series.reindex(backtest_months, method='ffill')

    # 计算月度收益
    # 预处理收益率数据，将索引设置为时间和股票代码，便于快速查找
    returns_df_indexed = returns_df.set_index(['Trdmnt', 'Stkcd'])

    monthly_returns = []
    for month, stocks in portfolio_map.items():
        if not isinstance(stocks, set) or not stocks:
            monthly_returns.append(0.0)
            continue

        month_str = month.strftime('%Y-%m')
        try:
            # 使用 .loc 和 MultiIndex 进行高效切片
            # 我们只关心当月的收益，所以只选择 month_str
            # 使用 isin 筛选出当月持仓的股票
            relevant_returns = returns_df_indexed.loc[month_str]
            avg_return = relevant_returns[relevant_returns.index.isin(stocks)]['Mretwd'].mean()
            monthly_returns.append(avg_return if pd.notna(avg_return) else 0.0)
        except KeyError:
            # 如果当月没有任何收益数据
            monthly_returns.append(0.0)

    return pd.Series(monthly_returns, index=backtest_months, name=f"return_{strategy_name}")


def process_strategy_task(
        holdings_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        vol_col: str,
        quantile: float,
        start_date: str,
        end_date: str
) -> tuple[str, dict, pd.Series]:
    """
    一个完整的任务单元，用于并行处理单个策略。
    它包含：构建投资组合 -> 运行回测。
    """
    strategy_name = f"quality_{vol_col}"
    print(f"  -> [PID: {os.getpid()}] 开始处理策略: {strategy_name}...")

    # 1. 构建投资组合
    selections = build_portfolio(holdings_df, vol_col, quantile)

    # 2. 运行回测
    returns_series = run_single_strategy_backtest(strategy_name, selections, returns_df, start_date, end_date)

    print(f"  <- [PID: {os.getpid()}] 完成策略: {strategy_name}")
    return strategy_name, selections, returns_series


# --- 数据加载与主流程控制 ---

def load_common_data(returns_file: Path, benchmark_file: Path, benchmark_code: str) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    """加载所有策略共享的收益率和基准数据。"""
    print("--- 步骤 1: 加载公用数据 (股票收益率, 基准指数) ---")

    # 加载股票收益率
    # For GPU acceleration with RAPIDS, replace pd.read_csv with cudf.read_csv
    # import cudf
    # returns_df = cudf.read_csv(returns_file)
    returns_df = pd.read_csv(returns_file)
    returns_df['Stkcd'] = returns_df['Stkcd'].astype(str).str.zfill(6)
    # 预处理日期格式以匹配回测逻辑
    returns_df['Trdmnt'] = pd.to_datetime(returns_df['Trdmnt']).dt.strftime('%Y-%m')
    returns_df['Mretwd'] = pd.to_numeric(returns_df['Mretwd'], errors='coerce').fillna(0)
    print(f"股票收益率数据加载完成，共 {len(returns_df)} 条记录。")

    # 加载基准数据
    all_benchmarks_df = pd.read_csv(benchmark_file)
    benchmark_df = all_benchmarks_df[all_benchmarks_df['Indexcd'].astype(str).str.zfill(6) == benchmark_code].copy()
    benchmark_df['date'] = pd.to_datetime(benchmark_df['Month'], format='%Y-%m')
    benchmark_df.rename(columns={'Idxrtn': 'benchmark_return'}, inplace=True)
    benchmark_df = benchmark_df[['date', 'benchmark_return']]
    print(f"基准 '{benchmark_code}' 数据加载完成。")

    return returns_df, benchmark_df


def calculate_and_save_performance(
        all_returns_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        all_selections: dict,
        risk_free_rate: float,
        output_dir: Path,
):
    """
    计算并保存多个策略的详细绩效指标。
    """
    print("\n--- 步骤 3: 计算并保存各策略绩效指标 ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_output_file = output_dir / 'all_strategies_monthly_returns.csv'
    performance_output_file = output_dir / 'all_strategies_performance_summary.csv'

    # 1. 合并基准收益
    results = all_returns_df.join(benchmark_df.set_index('date'), how='left').fillna(0)

    all_metrics = []
    total_months = len(results)

    # 2. 计算基准的绩效
    annualized_benchmark_return = (1 + results['benchmark_return']).prod() ** (12 / total_months) - 1

    # 3. 循环计算每个策略的绩效
    for strategy_name, selections in all_selections.items():
        return_col = f'return_{strategy_name}'
        cum_col = f'cum_{strategy_name}'
        results[cum_col] = (1 + results[return_col]).cumprod()

        final_return = results[cum_col].iloc[-1]
        annualized_return = final_return ** (12 / total_months) - 1
        annualized_volatility = results[return_col].std() * np.sqrt(12)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0

        rolling_max = results[cum_col].expanding().max()
        drawdown = (results[cum_col] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        excess_return = results[return_col] - results['benchmark_return']
        annualized_excess_return = annualized_return - annualized_benchmark_return
        tracking_error = excess_return.std() * np.sqrt(12)
        information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else 0

        # 计算换手率
        turnover_list = []
        rebalance_dates = sorted(selections.keys())
        for i in range(1, len(rebalance_dates)):
            prev_portfolio = selections.get(rebalance_dates[i - 1], set())
            curr_portfolio = selections.get(rebalance_dates[i], set())
            if not prev_portfolio: continue
            stocks_sold = len(prev_portfolio - curr_portfolio)
            period_turnover = stocks_sold / len(prev_portfolio)
            turnover_list.append(period_turnover)
        # 换手率是基于调仓周期的，年化需要乘以每年的调仓次数
        num_rebalances_per_year = 12 / np.mean([(rebalance_dates[i] - rebalance_dates[i - 1]).days / 30.44 for i in
                                                range(1, len(rebalance_dates))]) if len(rebalance_dates) > 1 else 1
        annual_turnover = np.mean(turnover_list) * num_rebalances_per_year if turnover_list else 0.0

        metrics = {
            '策略名称': strategy_name,
            '年化收益率': annualized_return, '年化波动率': annualized_volatility, '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown, '年化换手率': annual_turnover, '累计收益率': final_return - 1,
            '年化超额收益率': annualized_excess_return, '信息比率': information_ratio, '跟踪误差': tracking_error,
        }
        all_metrics.append(metrics)

    # 4. 整理并保存结果
    performance_df = pd.DataFrame(all_metrics).set_index('策略名称')
    performance_df.loc['基准 (沪深300)', '年化收益率'] = annualized_benchmark_return

    results.to_csv(returns_output_file, encoding='utf-8-sig', float_format='%.6f')
    print(f"所有策略的月度收益数据已合并保存至: {returns_output_file}")

    formatted_df = performance_df.copy()
    percent_cols = ['年化收益率', '年化波动率', '最大回撤', '年化换手率', '累计收益率', '年化超额收益率', '跟踪误差']
    for col in percent_cols:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else '-')
    for col in ['夏普比率', '信息比率']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')

    formatted_df.to_csv(performance_output_file, encoding='utf-8-sig')
    print(f"所有策略的格式化绩效评估报告已保存至: {performance_output_file}")

    print("\n--- 各策略绩效对比简报 ---")
    print(formatted_df.to_string())


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="高性能并行化股票策略回测工具。")
    parser.add_argument('--holdings-file', type=Path, required=True, help='包含持仓和质量因子(如波动率)的CSV文件路径。')
    parser.add_argument('--returns-file', type=Path, required=True,
                        help='包含股票月度收益率的CSV文件路径 (TRDNEW_Mnth.csv)。')
    parser.add_argument('--benchmark-file', type=Path, required=True, help='包含基准指数月度收益率的CSV文件路径。')
    parser.add_argument('--output-dir', type=Path, required=True, help='存放输出结果 (收益率、绩效) 的文件夹。')
    parser.add_argument('--volatility-cols', nargs='+', required=True, help='持仓文件中用作筛选的一个或多个波动率列名。')
    parser.add_argument('--start-date', type=str, required=True, help='回测开始日期 (YYYY-MM-DD)。')
    parser.add_argument('--end-date', type=str, required=True, help='回测结束日期 (YYYY-MM-DD)。')
    parser.add_argument('--benchmark-code', type=str, default='000300', help='基准指数代码。')
    parser.add_argument('--risk-free-rate', type=float, default=0.03, help='年化无风险利率。')
    parser.add_argument('--quantile', type=float, default=0.5, help='用于筛选股票的分位数。')

    args = parser.parse_args()

    start_time = time.time()

    try:
        # 1. 加载所有策略都需要的公用数据
        returns_df, benchmark_df = load_common_data(args.returns_file, args.benchmark_file, args.benchmark_code)

        # 加载持仓文件，这个文件将被传递给所有并行进程
        print("--- 加载持仓数据 ---")
        holdings_df = pd.read_csv(args.holdings_file)
        holdings_df['调入日期'] = pd.to_datetime(holdings_df['调入日期'])
        holdings_df['stkcd'] = holdings_df['stkcd'].astype(str).str.zfill(6)
        print("持仓数据加载完成。")

        # 2. 使用 ProcessPoolExecutor 并行执行所有策略的回测
        print(f"\n--- 步骤 2: 并行回测 {len(args.volatility_cols)} 个策略 ---")
        all_selections = {}
        all_returns_list = []

        # 使用 with 语句确保进程池被正确关闭
        with ProcessPoolExecutor() as executor:
            # 创建未来任务列表
            futures = {
                executor.submit(
                    process_strategy_task,
                    holdings_df,
                    returns_df,
                    vol_col,
                    args.quantile,
                    args.start_date,
                    args.end_date
                ): vol_col
                for vol_col in args.volatility_cols
            }

            # 等待任务完成并收集结果
            for future in as_completed(futures):
                vol_col = futures[future]
                try:
                    strategy_name, selections, returns_series = future.result()
                    all_selections[strategy_name] = selections
                    all_returns_list.append(returns_series)
                except Exception as e:
                    print(f"!! 策略 {vol_col} 在执行中发生错误: {e}")
                    traceback.print_exc()

        if not all_returns_list:
            print("所有策略均未能成功执行，程序终止。")
            return

        # 3. 合并所有策略的收益率结果
        all_returns_df = pd.concat(all_returns_list, axis=1)

        # 4. 计算并保存所有策略的绩效
        calculate_and_save_performance(
            all_returns_df,
            benchmark_df,
            all_selections,
            args.risk_free_rate,
            args.output_dir
        )

        end_time = time.time()
        print(f"\n--- 所有任务完成！总耗时: {end_time - start_time:.2f} 秒 ---")

    except Exception as e:
        print(f"\n执行过程中出现严重错误: {e}")
        traceback.print_exc()


# 导入 os 模块以在日志中显示进程ID
import os

if __name__ == "__main__":
    # 为了在 Windows 上使用 multiprocessing，需要将主逻辑放在这个保护块内
    main()
