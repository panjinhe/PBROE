# backtest_engine.py
# 一个用于PB-ROE系列策略的通用、向量化回测引擎

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


# =================================================================== #
#                  【1. 投资组合构建模块】                          #
# =================================================================== #

def build_single_portfolio(file_path: Path, quantile: float, residual_col: str) -> pd.DataFrame:
    """
    【新增】从单个残差文件中，根据指定的分位数阈值构建投资组合。
    """
    print(f"--- 步骤 1: 构建单一策略投资组合 (来自: {file_path.name}) ---")
    try:
        df_all = pd.read_csv(file_path)
        df_all['调入日期'] = pd.to_datetime(df_all['调入日期'])
        df_all['stkcd'] = df_all['stkcd'].astype(str).str.zfill(6)
        print(f"成功加载 {len(df_all)} 条残差数据。")
    except FileNotFoundError:
        print(f"错误: 找不到残差文件 {file_path}。")
        return None

    portfolio_list = []
    for date, group in df_all.groupby('调入日期'):
        threshold = group[residual_col].quantile(quantile)
        selected_stocks = group[group[residual_col] <= threshold]
        portfolio_list.append(selected_stocks)

    strategy_df = pd.concat(portfolio_list)
    print(f"已根据残差最低 {quantile * 100:.0f}% 的标准，构建了包含 {len(strategy_df)} 条持仓记录的投资组合。\n")
    return strategy_df


def build_intersection_portfolio(file_p3: Path, file_p4: Path, quantile: float) -> pd.DataFrame:
    """
    构建两个策略最低残差组的交集投资组合。
    """
    print("--- 步骤 1: 构建交集投资组合 ---")
    try:
        df_p3 = pd.read_csv(file_p3)
        df_p3['stkcd'] = df_p3['stkcd'].astype(str).str.zfill(6)
        df_p3['调入日期'] = pd.to_datetime(df_p3['调入日期'])

        df_p4 = pd.read_csv(file_p4)
        df_p4['stkcd'] = df_p4['stkcd'].astype(str).str.zfill(6)
        df_p4['调入日期'] = pd.to_datetime(df_p4['调入日期'])
        print("策略 pbroe3.1 和 pbroe4.1 的残差文件加载成功。")
    except FileNotFoundError as e:
        print(f"错误: 找不到残差文件 {e.filename}。")
        return None

    common_dates = sorted(list(set(df_p3['调入日期']) & set(df_p4['调入日期'])))
    print(f"找到 {len(common_dates)} 个共同的调仓日期进行比较。")

    intersection_portfolio_list = []
    for date in common_dates:
        group3 = df_p3[df_p3['调入日期'] == date]
        if group3.empty: continue
        threshold3 = group3['residual_zscore'].quantile(quantile)
        portfolio3_stocks = set(group3[group3['residual_zscore'] <= threshold3]['stkcd'])

        group4 = df_p4[df_p4['调入日期'] == date]
        if group4.empty: continue
        threshold4 = group4['residual_zscore_adj'].quantile(quantile)
        portfolio4_stocks = set(group4[group4['residual_zscore_adj'] <= threshold4]['stkcd'])

        intersection_stocks = list(portfolio3_stocks.intersection(portfolio4_stocks))

        if intersection_stocks:
            temp_df = pd.DataFrame({'调入日期': date, 'stkcd': intersection_stocks})
            intersection_portfolio_list.append(temp_df)

    if not intersection_portfolio_list:
        print("错误：未能构建任何有效的交集投资组合。")
        return None

    strategy_df = pd.concat(intersection_portfolio_list)
    print(f"已成功构建交集投资组合，共包含 {len(strategy_df)} 条持仓记录。\n")
    return strategy_df


# =================================================================== #
#                  【后续模块与之前版本相同】                         #
# =================================================================== #

def load_and_preprocess_data(returns_path, benchmark_path, benchmark_code):
    print("--- 步骤 2: 加载并预处理收益与基准数据 ---")
    try:
        try:
            returns_df = pd.read_csv(returns_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            returns_df = pd.read_csv(returns_path, encoding='gbk')
        all_benchmarks_df = pd.read_csv(benchmark_path)
    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件 {e.filename}。")
        return None, None

    returns_df.rename(columns={'Stkcd': 'stkcd', 'Trdmnt': 'date', 'Mretwd': 'stock_return'}, inplace=True)
    returns_df['stkcd'] = returns_df['stkcd'].astype(str).str.zfill(6)
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    returns_df['stock_return'] = pd.to_numeric(returns_df['stock_return'], errors='coerce')

    all_benchmarks_df['Indexcd'] = all_benchmarks_df['Indexcd'].astype(str).str.zfill(6)
    benchmark_df = all_benchmarks_df[all_benchmarks_df['Indexcd'] == benchmark_code].copy()
    if benchmark_df.empty: return None, None
    benchmark_df['date'] = pd.to_datetime(benchmark_df['Month'], format='%Y-%m')
    benchmark_df.rename(columns={'Idxrtn': 'benchmark_return'}, inplace=True)
    benchmark_df = benchmark_df[['date', 'benchmark_return']]
    print("数据预处理完成。\n")
    return returns_df, benchmark_df


def run_backtest_vectorized(strategy_df, returns_df, start_date_str, end_date_str):
    print("--- 步骤 3: 执行回测 (向量化加速版) ---")
    strategy_df.rename(columns={'调入日期': 'date'}, inplace=True)
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    strategy_df = strategy_df[(strategy_df['date'] >= start_date) & (strategy_df['date'] <= end_date)]
    if strategy_df.empty: return pd.DataFrame()

    merged_returns = pd.merge(strategy_df, returns_df, on=['date', 'stkcd'], how='inner')
    portfolio_returns_df = merged_returns.groupby('date')['stock_return'].mean().reset_index()
    portfolio_returns_df.rename(columns={'stock_return': 'portfolio_return'}, inplace=True)
    print(f"回测完成，已生成 {len(portfolio_returns_df)} 条月度收益记录。\n")
    return portfolio_returns_df


def calculate_annual_turnover_vectorized(strategy_df, start_date_str, end_date_str):
    print("--- 步骤 4: 计算年化换手率 (向量化加速版) ---")
    strategy_df.rename(columns={'调入日期': 'date'}, inplace=True, errors='ignore')
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    strategy_df = strategy_df[(strategy_df['date'] >= start_date) & (strategy_df['date'] <= end_date)]
    if strategy_df.empty or len(strategy_df['date'].unique()) < 2: return 0.0

    holdings_matrix = strategy_df.pivot_table(index='date', columns='stkcd', aggfunc=lambda x: 1, fill_value=0)
    holdings_diff = holdings_matrix.diff().fillna(0)
    stocks_sold = (holdings_diff == -1).sum(axis=1)
    prev_holdings_count = holdings_matrix.shift(1).sum(axis=1)
    period_turnover = (stocks_sold / prev_holdings_count).replace([np.inf, -np.inf], 0).fillna(0)

    avg_period_turnover = period_turnover.mean()
    rebalances_per_year = 12  # 假设每月调仓
    annual_turnover = avg_period_turnover * rebalances_per_year
    print(f"计算完成。预估年化换手率: {annual_turnover:.2%}\n")
    return annual_turnover


def calculate_performance_and_save(portfolio_returns_df, benchmark_df, annual_turnover, config):
    print("--- 步骤 5: 计算绩效并保存结果 ---")
    merged_df = pd.merge(portfolio_returns_df, benchmark_df, on='date', how='left')
    merged_df['benchmark_return'].fillna(0.0, inplace=True)
    total_months = len(merged_df)
    if total_months == 0: return None, None

    merged_df['cumulative_return'] = (1 + merged_df['portfolio_return']).cumprod()
    final_cumulative_return = merged_df['cumulative_return'].iloc[-1]

    annualized_return = final_cumulative_return ** (12 / total_months) - 1
    annualized_volatility = merged_df['portfolio_return'].std() * np.sqrt(12)
    sharpe_ratio = (annualized_return - config[
        'RISK_FREE_RATE']) / annualized_volatility if annualized_volatility != 0 else 0

    rolling_max = merged_df['cumulative_return'].expanding().max()
    drawdown = (merged_df['cumulative_return'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    annualized_benchmark_return = (1 + merged_df['benchmark_return']).prod() ** (12 / total_months) - 1
    excess_return = merged_df['portfolio_return'] - merged_df['benchmark_return']
    annualized_excess_return = annualized_return - annualized_benchmark_return
    tracking_error = excess_return.std() * np.sqrt(12)
    information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else 0

    metrics = {'年化收益率': annualized_return, '年化波动率': annualized_volatility, '夏普比率': sharpe_ratio,
               '最大回撤': max_drawdown, '年化换手率': annual_turnover, '年化超额收益率': annualized_excess_return,
               '信息比率': information_ratio, '跟踪误差': tracking_error, '基准年化收益率': annualized_benchmark_return,
               '累计收益率': final_cumulative_return - 1}

    print(f"\n--- {config['STRATEGY_NAME']} 绩效简报 ---")
    for key, value in metrics.items():
        print(f"{key + ':':<12} {value:.2%}" if isinstance(value, float) and (
                    '率' in key or '回撤' in key) else f"{key + ':':<12} {value:.4f}")

    returns_filename = config['OUTPUT_DIR'] / f"{config['STRATEGY_NAME']}_returns.csv"
    metrics_filename = config['OUTPUT_DIR'] / f"{config['STRATEGY_NAME']}_performance.csv"

    merged_df.to_csv(returns_filename, index=False, encoding='utf-8-sig', float_format='%.6f')
    print(f"\n月度收益率详情已保存至: {returns_filename}")
    pd.DataFrame([metrics]).to_csv(metrics_filename, index=False, encoding='utf-8-sig', float_format='%.6f')
    print(f"绩效指标已保存至: {metrics_filename}\n")
    return merged_df, metrics


# =================================================================== #
#                          【主函数执行】                             #
# =================================================================== #

def main(config):
    """主执行函数"""
    try:
        # --- 核心修改：根据策略类型选择不同的组合构建函数 ---
        if config['STRATEGY_TYPE'] == 'single':
            strategy_df = build_single_portfolio(
                config['RESIDUAL_FILE'],
                config['RESIDUAL_QUANTILE'],
                config['RESIDUAL_COL']
            )
        elif config['STRATEGY_TYPE'] == 'intersection':
            strategy_df = build_intersection_portfolio(
                config['RESIDUAL_FILE_P3'],
                config['RESIDUAL_FILE_P4'],
                config['RESIDUAL_QUANTILE']
            )
        else:
            print(f"错误: 未知的策略类型 '{config['STRATEGY_TYPE']}'")
            return

        if strategy_df is None: return

        returns_df, benchmark_df = load_and_preprocess_data(
            config['RETURNS_FILE'],
            config['BENCHMARK_FILE'],
            config['BENCHMARK_CODE']
        )
        if returns_df is None: return

        portfolio_returns_df = run_backtest_vectorized(
            strategy_df.copy(),
            returns_df,
            config['BACKTEST_START_DATE'],
            config['BACKTEST_END_DATE']
        )
        if portfolio_returns_df.empty: return

        annual_turnover = calculate_annual_turnover_vectorized(
            strategy_df.copy(),
            config['BACKTEST_START_DATE'],
            config['BACKTEST_END_DATE']
        )

        calculate_performance_and_save(
            portfolio_returns_df,
            benchmark_df,
            annual_turnover,
            config
        )
        print("\n回测完成！")

    except Exception as e:
        print(f"\n执行过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()


# =================================================================== #
#                       【脚本执行入口】                              #
# =================================================================== #

if __name__ == "__main__":
    # --- 示例1：回测单一策略 (pbroe4.1) ---
    CONFIG_SINGLE = {
        "STRATEGY_TYPE": "single",  # <--- 指定策略类型

        # 文件路径
        "DATA_PATH": Path("E:/PBROE/data"),
        "CH4_PATH": Path("E:/PBROE/ch4"),

        # 输入文件
        "RESIDUAL_FILE": Path("E:/PBROE/ch4/pbroe4.1Residuals.csv"),  # <--- 单一残差文件
        "RESIDUAL_COL": "residual_zscore_adj",  # <--- 指定残差列名
        "RETURNS_FILE": Path("E:/PBROE/data/TRDNEW_Mnth.csv"),
        "BENCHMARK_FILE": Path("E:/PBROE/data/benchmark_indices.csv"),

        # 输出配置
        "OUTPUT_DIR": Path("E:/PBROE/ch4"),
        "STRATEGY_NAME": "pbroe4.1_single",

        # 策略参数
        "RESIDUAL_QUANTILE": 0.1,

        # 回测参数
        "BACKTEST_START_DATE": '2010-05-01',
        "BACKTEST_END_DATE": '2025-04-30',
        "BENCHMARK_CODE": '000300',
        "RISK_FREE_RATE": 0.03
    }

    # --- 示例2：回测交集策略 (pbroe4.2) ---
    CONFIG_INTERSECTION = {
        "STRATEGY_TYPE": "intersection",  # <--- 指定策略类型

        "DATA_PATH": Path("E:/PBROE/data"),
        "CH3_PATH": Path("E:/PBROE/ch3"),
        "CH4_PATH": Path("E:/PBROE/ch4"),

        "RESIDUAL_FILE_P3": Path("/ch3/pbroe3.1Residuals.csv"),
        "RESIDUAL_FILE_P4": Path("E:/PBROE/ch4/pbroe4.1Residuals.csv"),
        "RETURNS_FILE": Path("E:/PBROE/data/TRDNEW_Mnth.csv"),
        "BENCHMARK_FILE": Path("E:/PBROE/data/benchmark_indices.csv"),

        "OUTPUT_DIR": Path("E:/PBROE/ch4"),
        "STRATEGY_NAME": "pbroe4.2_intersection",

        "RESIDUAL_QUANTILE": 0.2,

        "BACKTEST_START_DATE": '2010-05-01',
        "BACKTEST_END_DATE": '2025-04-30',
        "BENCHMARK_CODE": '000300',
        "RISK_FREE_RATE": 0.03
    }

    # --- 选择要运行的配置 ---
    # 确保输出目录存在
    # CONFIG_SINGLE['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)
    # main(CONFIG_SINGLE) # <--- 运行单一策略回测

    CONFIG_INTERSECTION['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)
    main(CONFIG_INTERSECTION)  # <--- 运行交集策略回测
