# E:\PBROE\ch5\backtestch5.py
# 一个用于PB-ROE系列策略的通用、向量化回测引擎

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


# =================================================================== #
#                  【1. 投资组合构建模块】                          #
# =================================================================== #

def build_single_portfolio(file_path: Path, quantile: float, residual_col: str, start_date, end_date) -> pd.DataFrame:
    """
    从单个残差文件中，根据指定的分位数阈值构建投资组合。
    参数:
        file_path (Path): 残差数据文件的路径。
        quantile (float): 筛选残差的百分位数阈值（例如 0.1 代表最低 10%）。
        residual_col (str): 用于筛选的残差列名。
        start_date (pd.Timestamp): 回测开始日期。
        end_date (pd.Timestamp): 回测结束日期。
    返回:
        pd.DataFrame: 包含选定股票的投资组合DataFrame。
    """
    print(f"--- 步骤 1: 构建单一策略投资组合 (来自: {file_path.name}) ---")
    try:
        df_all = pd.read_csv(file_path)
        df_all['调入日期'] = pd.to_datetime(df_all['调入日期'])
        df_all['stkcd'] = df_all['stkcd'].astype(str).str.zfill(6)

        # 在加载后立即过滤到回测期间的因子数据
        df_all = df_all[(df_all['调入日期'] >= start_date) & (df_all['调入日期'] <= end_date)].copy()

        print(f"成功加载并过滤 {len(df_all)} 条残差数据至回测区间。")
    except FileNotFoundError:
        print(f"错误: 找不到残差文件 {file_path}。")
        return None
    except KeyError as e:
        print(f"错误: 残差文件中未找到列 '{e}'。请检查 'residual_col' 配置。")
        return None

    portfolio_list = []
    for date, group in df_all.groupby('调入日期'):
        if len(group) > 0 and residual_col in group.columns:
            threshold = group[residual_col].quantile(quantile)
            selected_stocks = group[group[residual_col] <= threshold]
            portfolio_list.append(selected_stocks)
        else:
            pass  # 避免过多输出

    if not portfolio_list:
        print("警告: 未能构建任何有效的投资组合，可能是数据不足或筛选条件过严。")
        return None

    strategy_df = pd.concat(portfolio_list)
    print(f"已根据残差最低 {quantile * 100:.0f}% 的标准，构建了包含 {len(strategy_df)} 条持仓记录的投资组合。\n")
    return strategy_df


def build_pbroe4_2_portfolio(file_p3: Path, file_p4: Path, quantile: float, start_date, end_date) -> pd.DataFrame:
    """
    构建 PBROE 4.2 策略（pbroe3.1 和 pbroe4.1 最低残差组的交集）投资组合。
    参数:
        file_p3 (Path): pbroe3.1 残差数据文件的路径。
        file_p4 (Path): pbroe4.1 残差数据文件的路径。
        quantile (float): 筛选残差的百分位数阈值（例如 0.1 代表最低 10%）。
        start_date (pd.Timestamp): 回测开始日期。
        end_date (pd.Timestamp): 回测结束日期。
    返回:
        pd.DataFrame: 包含选定股票的投资组合DataFrame。
    """
    print("--- 步骤 1: 构建 PBROE 4.2 交集投资组合 ---")
    try:
        df_p3 = pd.read_csv(file_p3)
        df_p3['stkcd'] = df_p3['stkcd'].astype(str).str.zfill(6)
        df_p3['调入日期'] = pd.to_datetime(df_p3['调入日期'])
        df_p3 = df_p3[(df_p3['调入日期'] >= start_date) & (df_p3['调入日期'] <= end_date)].copy()

        df_p4 = pd.read_csv(file_p4)
        df_p4['stkcd'] = df_p4['stkcd'].astype(str).str.zfill(6)
        df_p4['调入日期'] = pd.to_datetime(df_p4['调入日期'])
        df_p4 = df_p4[(df_p4['调入日期'] >= start_date) & (df_p4['调入日期'] <= end_date)].copy()

        print(f"策略 pbroe3.1 ({len(df_p3)}条) 和 pbroe4.1 ({len(df_p4)}条) 的残差文件加载并过滤成功。")
    except FileNotFoundError as e:
        print(f"错误: 找不到残差文件 {e.filename}。")
        return None

    common_dates = sorted(list(set(df_p3['调入日期']) & set(df_p4['调入日期'])))
    print(f"找到 {len(common_dates)} 个共同的调仓日期进行比较。")

    intersection_portfolio_list = []
    for date in common_dates:
        group3 = df_p3[df_p3['调入日期'] == date]
        if group3.empty or 'residual_zscore' not in group3.columns or len(group3) == 0: continue
        threshold3 = group3['residual_zscore'].quantile(quantile)
        portfolio3_stocks = set(group3[group3['residual_zscore'] <= threshold3]['stkcd'])

        group4 = df_p4[df_p4['调入日期'] == date]
        if group4.empty or 'residual_zscore_adj' not in group4.columns or len(group4) == 0: continue
        threshold4 = group4['residual_zscore_adj'].quantile(quantile)
        portfolio4_stocks = set(group4[group4['residual_zscore_adj'] <= threshold4]['stkcd'])

        intersection_stocks = list(portfolio3_stocks.intersection(portfolio4_stocks))

        if intersection_stocks:
            temp_df = pd.DataFrame({'调入日期': date, 'stkcd': intersection_stocks})
            intersection_portfolio_list.append(temp_df)

    if not intersection_portfolio_list:
        print("错误：未能构建任何有效的交集投资组合。可能是数据不足或共同日期太少。")
        return None

    strategy_df = pd.concat(intersection_portfolio_list)
    print(f"已成功构建 PBROE 4.2 交集投资组合，共包含 {len(strategy_df)} 条持仓记录。\n")
    return strategy_df


def build_pbroe5_0_portfolio(file_p3: Path, file_p4: Path, file_ts_quantile: Path,
                             ts_quantile_col: str, cross_sectional_quantile: float,
                             time_series_quantile_threshold: float, start_date, end_date) -> pd.DataFrame:
    """
    构建 PBROE 5.0 策略（pbroe3.1、pbroe4.1 和时序残差分位数最低组的交集）投资组合。
    参数:
        file_p3 (Path): pbroe3.1 原始残差数据文件的路径。
        file_p4 (Path): pbroe4.1 原始残差数据文件的路径。
        file_ts_quantile (Path): 包含时序残差分位数数据的文件路径。
        ts_quantile_col (str): 时序残差分位数列的名称（例如 'residual_quantile_10m'）。
        cross_sectional_quantile (float): 横截面残差（pbroe3.1和pbroe4.1）的筛选百分位数阈值。
        time_series_quantile_threshold (float): 时序残差分位数的筛选阈值（例如 0.2 代表低于 20%）。
        start_date (pd.Timestamp): 回测开始日期。
        end_date (pd.Timestamp): 回测结束日期。
    返回:
        pd.DataFrame: 包含选定股票的投资组合DataFrame。
    """
    print("--- 步骤 1: 构建 PBROE 5.0 融合投资组合 ---")
    try:
        df_p3 = pd.read_csv(file_p3)
        df_p3['stkcd'] = df_p3['stkcd'].astype(str).str.zfill(6)
        df_p3['调入日期'] = pd.to_datetime(df_p3['调入日期'])
        df_p3 = df_p3[(df_p3['调入日期'] >= start_date) & (df_p3['调入日期'] <= end_date)].copy()

        df_p4 = pd.read_csv(file_p4)
        df_p4['stkcd'] = df_p4['stkcd'].astype(str).str.zfill(6)
        df_p4['调入日期'] = pd.to_datetime(df_p4['调入日期'])
        df_p4 = df_p4[(df_p4['调入日期'] >= start_date) & (df_p4['调入日期'] <= end_date)].copy()

        df_ts = pd.read_csv(file_ts_quantile)
        df_ts['stkcd'] = df_ts['stkcd'].astype(str).str.zfill(6)
        df_ts['调入日期'] = pd.to_datetime(df_ts['调入日期'])
        df_ts = df_ts[(df_ts['调入日期'] >= start_date) & (df_ts['调入日期'] <= end_date)].copy()  # 过滤时序分位数文件

        print(f"所有残差和时序分位数文件加载并过滤成功 (p3:{len(df_p3)}, p4:{len(df_p4)}, ts:{len(df_ts)}条)。")
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。")
        return None
    except KeyError as e:
        print(f"错误: 时序分位数文件中未找到列 '{e}'。请检查 'ts_quantile_col' 配置或文件内容。")
        return None

    # 找到所有文件的共同日期
    common_dates_set = set(df_p3['调入日期']) & set(df_p4['调入日期']) & set(df_ts['调入日期'])
    common_dates = sorted(list(common_dates_set))
    print(f"找到 {len(common_dates)} 个共同的调仓日期进行比较。")

    fusion_portfolio_list = []
    for date in common_dates:
        # 筛选 pbroe3.1 低残差股票
        group3 = df_p3[df_p3['调入日期'] == date]
        if group3.empty or 'residual_zscore' not in group3.columns or len(group3) == 0: continue
        threshold3 = group3['residual_zscore'].quantile(cross_sectional_quantile)
        portfolio3_stocks = set(group3[group3['residual_zscore'] <= threshold3]['stkcd'])

        # 筛选 pbroe4.1 低残差股票
        group4 = df_p4[df_p4['调入日期'] == date]
        if group4.empty or 'residual_zscore_adj' not in group4.columns or len(group4) == 0: continue
        threshold4 = group4['residual_zscore_adj'].quantile(cross_sectional_quantile)
        portfolio4_stocks = set(group4[group4['residual_zscore_adj'] <= threshold4]['stkcd'])

        # 筛选时序残差分位数低于阈值的股票
        group_ts = df_ts[df_ts['调入日期'] == date]
        if group_ts.empty or ts_quantile_col not in group_ts.columns or len(group_ts) == 0: continue
        # 直接使用 time_series_quantile_threshold 进行筛选，因为 ts_quantile_col 已经是分位数
        portfolio_ts_stocks = set(group_ts[group_ts[ts_quantile_col] <= time_series_quantile_threshold]['stkcd'])

        # 取三个集合的交集
        intersection_stocks = list(portfolio3_stocks.intersection(portfolio4_stocks).intersection(portfolio_ts_stocks))

        if intersection_stocks:
            temp_df = pd.DataFrame({'调入日期': date, 'stkcd': intersection_stocks})
            fusion_portfolio_list.append(temp_df)

    if not fusion_portfolio_list:
        print("错误：未能构建任何有效的 PBROE 5.0 融合投资组合。可能是数据不足或筛选条件过严。")
        return None

    strategy_df = pd.concat(fusion_portfolio_list)
    print(f"已成功构建 PBROE 5.0 融合投资组合，共包含 {len(strategy_df)} 条持仓记录。\n")
    return strategy_df


def build_single_ts_quantile_portfolio(file_ts_quantile: Path, ts_quantile_col: str,
                                       time_series_quantile_threshold: float, start_date, end_date) -> pd.DataFrame:
    """
    【新增】构建单一时序残差分位数策略投资组合。
    参数:
        file_ts_quantile (Path): 包含时序残差分位数数据的文件路径。
        ts_quantile_col (str): 时序残差分位数列的名称（例如 'residual_quantile_10m'）。
        time_series_quantile_threshold (float): 时序残差分位数的筛选阈值（例如 0.2 代表低于 20%）。
        start_date (pd.Timestamp): 回测开始日期。
        end_date (pd.Timestamp): 回测结束日期。
    返回:
        pd.DataFrame: 包含选定股票的投资组合DataFrame。
    """
    print(f"--- 步骤 1: 构建单一时序残差分位数投资组合 (来自: {file_ts_quantile.name}) ---")
    try:
        df_ts = pd.read_csv(file_ts_quantile)
        df_ts['stkcd'] = df_ts['stkcd'].astype(str).str.zfill(6)
        df_ts['调入日期'] = pd.to_datetime(df_ts['调入日期'])
        df_ts = df_ts[(df_ts['调入日期'] >= start_date) & (df_ts['调入日期'] <= end_date)].copy()  # 过滤时序分位数文件

        print(f"时序分位数文件加载并过滤成功 ({len(df_ts)}条)。")
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {file_ts_quantile}。")
        return None
    except KeyError as e:
        print(f"错误: 时序分位数文件中未找到列 '{e}'。请检查 'ts_quantile_col' 配置或文件内容。")
        return None

    portfolio_list = []
    for date, group in df_ts.groupby('调入日期'):
        if group.empty or ts_quantile_col not in group.columns or len(group) == 0: continue

        # 筛选时序残差分位数低于阈值的股票
        selected_stocks = group[group[ts_quantile_col] <= time_series_quantile_threshold]

        if not selected_stocks.empty:
            portfolio_list.append(selected_stocks)

    if not portfolio_list:
        print("错误：未能构建任何有效的单一时序残差分位数投资组合。可能是数据不足或筛选条件过严。")
        return None

    strategy_df = pd.concat(portfolio_list)
    print(f"已成功构建单一时序残差分位数投资组合，共包含 {len(strategy_df)} 条持仓记录。\n")
    return strategy_df


# =================================================================== #
#                  【后续模块与之前版本相同】                         #
# =================================================================== #

def load_and_preprocess_data(returns_path, benchmark_path, benchmark_code, start_date, end_date):
    """
    加载并预处理股票收益率和基准指数数据。
    参数:
        returns_path (Path): 股票月度收益率数据文件的路径。
        benchmark_path (Path): 基准指数月度收益率数据文件的路径。
        benchmark_code (str): 基准指数代码。
        start_date (pd.Timestamp): 回测开始日期。
        end_date (pd.Timestamp): 回测结束日期。
    返回:
        tuple: (returns_df, benchmark_df)，如果加载失败则返回 (None, None)。
    """
    print("--- 步骤 2: 加载并预处理收益与基准数据 ---")
    returns_df = None
    all_benchmarks_df = None
    try:
        # 尝试使用 utf-8-sig 编码加载股票收益率数据，失败则尝试 gbk
        try:
            returns_df = pd.read_csv(returns_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            returns_df = pd.read_csv(returns_path, encoding='gbk')
        print(f"股票收益率文件 '{returns_path.name}' 加载成功。")

        # 尝试使用 utf-8-sig 编码加载基准指数数据，失败则尝试 gbk
        try:
            all_benchmarks_df = pd.read_csv(benchmark_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            all_benchmarks_df = pd.read_csv(benchmark_path, encoding='gbk')
        print(f"基准指数文件 '{benchmark_path.name}' 加载成功。")

    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件 {e.filename}。")
        return None, None
    except Exception as e:  # 捕获其他潜在的加载错误
        print(f"加载数据文件时发生错误: {e}")
        return None, None

    # 如果任一DataFrame加载失败，则直接返回
    if returns_df is None or all_benchmarks_df is None:
        return None, None

    # 处理股票收益率数据
    returns_df.rename(columns={'Stkcd': 'stkcd', 'Trdmnt': 'date', 'Mretwd': 'stock_return'}, inplace=True)
    returns_df['stkcd'] = returns_df['stkcd'].astype(str).str.zfill(6)
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    returns_df['stock_return'] = pd.to_numeric(returns_df['stock_return'], errors='coerce')
    # 过滤股票收益数据到回测区间
    returns_df = returns_df[(returns_df['date'] >= start_date) & (returns_df['date'] <= end_date)].copy()
    print(f"股票收益率数据已过滤至回测区间，共 {len(returns_df)} 条。")

    # 处理基准指数数据
    # 查找包含 'Indexcd' 字符串的列名 (不区分大小写，去除空格)
    index_col = next((col for col in all_benchmarks_df.columns if 'indexcd' in col.lower().replace(' ', '')), None)
    month_col = next((col for col in all_benchmarks_df.columns if 'month' in col.lower().replace(' ', '')), None)
    idxrtn_col = next((col for col in all_benchmarks_df.columns if 'idxrtn' in col.lower().replace(' ', '')), None)

    if not all([index_col, month_col, idxrtn_col]):
        print(f"错误: 基准指数文件 '{benchmark_path.name}' 中未找到必要的列。")
        print(f"期望列包含 'Indexcd', 'Month', 'Idxrtn'。实际可用列: {all_benchmarks_df.columns.tolist()}")
        return None, None

    all_benchmarks_df[index_col] = all_benchmarks_df[index_col].astype(str).str.zfill(6)
    benchmark_df = all_benchmarks_df[all_benchmarks_df[index_col] == benchmark_code].copy()
    if benchmark_df.empty:
        print(f"错误: 未找到基准指数代码 '{benchmark_code}' 的数据。")
        return None, None

    benchmark_df['date'] = pd.to_datetime(benchmark_df[month_col], format='%Y-%m')
    benchmark_df.rename(columns={idxrtn_col: 'benchmark_return'}, inplace=True)
    benchmark_df = benchmark_df[['date', 'benchmark_return']]
    # 过滤基准收益数据到回测区间
    benchmark_df = benchmark_df[(benchmark_df['date'] >= start_date) & (benchmark_df['date'] <= end_date)].copy()
    print(f"基准收益率数据已过滤至回测区间，共 {len(benchmark_df)} 条。")

    print("数据预处理完成。\n")
    return returns_df, benchmark_df


def run_backtest_vectorized(strategy_df, returns_df, start_date_str, end_date_str):
    """
    执行向量化回测，计算投资组合月度收益率。
    参数:
        strategy_df (pd.DataFrame): 包含调仓日期和股票代码的投资组合DataFrame。
        returns_df (pd.DataFrame): 股票月度收益率数据。
        start_date_str (str): 回测开始日期字符串。
        end_date_str (str): 回测结束日期字符串。
    返回:
        pd.DataFrame: 包含月度投资组合收益率的DataFrame。
    """
    print("--- 步骤 3: 执行回测 (向量化加速版) ---")
    # 这里的日期过滤在 load_and_preprocess_data 中已经完成，这里仅作兼容性处理
    strategy_df.rename(columns={'调入日期': 'date'}, inplace=True, errors='ignore')
    # start_date = pd.to_datetime(start_date_str)
    # end_date = pd.to_datetime(end_date_str)
    # strategy_df = strategy_df[(strategy_df['date'] >= start_date) & (strategy_df['date'] <= end_date)]
    if strategy_df.empty:
        print("回测期间策略组合为空，无法执行回测。")
        return pd.DataFrame()

    merged_returns = pd.merge(strategy_df, returns_df, on=['date', 'stkcd'], how='inner')
    if merged_returns.empty:
        print("合并收益数据后，没有匹配的股票收益率，请检查数据和日期范围。")
        return pd.DataFrame()

    portfolio_returns_df = merged_returns.groupby('date')['stock_return'].mean().reset_index()
    portfolio_returns_df.rename(columns={'stock_return': 'portfolio_return'}, inplace=True)
    print(f"回测完成，已生成 {len(portfolio_returns_df)} 条月度收益记录。\n")
    return portfolio_returns_df


def calculate_annual_turnover_vectorized(strategy_df, start_date_str, end_date_str):
    """
    计算投资组合的年化换手率。
    参数:
        strategy_df (pd.DataFrame): 包含调仓日期和股票代码的投资组合DataFrame。
        start_date_str (str): 回测开始日期字符串。
        end_date_str (str): 回测结束日期字符串。
    返回:
        float: 年化换手率。
    """
    print("--- 步骤 4: 计算年化换手率 (向量化加速版) ---")
    strategy_df.rename(columns={'调入日期': 'date'}, inplace=True, errors='ignore')
    # 这里的日期过滤在 build_portfolio 函数中已经完成，这里仅作兼容性处理
    # start_date = pd.to_datetime(start_date_str)
    # end_date = pd.to_datetime(end_date_str)
    # strategy_df = strategy_df[(strategy_df['date'] >= start_date) & (strategy_df['date'] <= end_date)]
    if strategy_df.empty or len(strategy_df['date'].unique()) < 2:
        print("策略组合数据不足，无法计算换手率。")
        return 0.0

    # 创建一个矩阵，行是日期，列是股票代码，值为1表示持有，0表示不持有
    holdings_matrix = strategy_df.pivot_table(index='date', columns='stkcd', aggfunc=lambda x: 1, fill_value=0)

    # 计算每日持仓变化，-1表示卖出，1表示买入
    holdings_diff = holdings_matrix.diff().fillna(0)

    # 计算每天卖出的股票数量
    stocks_sold = (holdings_diff == -1).sum(axis=1)

    # 计算前一天的持仓股票总数
    prev_holdings_count = holdings_matrix.shift(1).sum(axis=1)

    # 计算每个调仓期的换手率 (卖出股票数 / 期初持仓总数)
    # 避免除以零的情况，将 inf 替换为 0
    period_turnover = (stocks_sold / prev_holdings_count).replace([np.inf, -np.inf], 0).fillna(0)

    # 排除第一个调仓期（因为没有前一期持仓）
    if not period_turnover.empty:
        period_turnover = period_turnover.iloc[1:]  # 排除第一个NaN或0，确保计算有效

    avg_period_turnover = period_turnover.mean() if not period_turnover.empty else 0.0
    rebalances_per_year = 12  # 假设每月调仓
    annual_turnover = avg_period_turnover * rebalances_per_year
    print(f"计算完成。预估年化换手率: {annual_turnover:.2%}\n")
    return annual_turnover


def calculate_performance_and_save(portfolio_returns_df, benchmark_df, annual_turnover, config):
    """
    计算策略绩效指标并保存结果。
    参数:
        portfolio_returns_df (pd.DataFrame): 投资组合月度收益率数据。
        benchmark_df (pd.DataFrame): 基准指数月度收益率数据。
        annual_turnover (float): 年化换手率。
        config (dict): 策略配置字典。
    返回:
        tuple: (merged_df, metrics)，包含合并后的收益数据和绩效指标字典。
    """
    print("--- 步骤 5: 计算绩效并保存结果 ---")
    merged_df = pd.merge(portfolio_returns_df, benchmark_df, on='date', how='left')
    merged_df['benchmark_return'].fillna(0.0, inplace=True)  # 填充缺失的基准收益
    total_months = len(merged_df)
    if total_months == 0:
        print("没有可用于绩效计算的月度收益数据。")
        return None, None

    # 确保收益率是数值类型
    merged_df['portfolio_return'] = pd.to_numeric(merged_df['portfolio_return'], errors='coerce')
    merged_df['benchmark_return'] = pd.to_numeric(merged_df['benchmark_return'], errors='coerce')
    merged_df.dropna(subset=['portfolio_return', 'benchmark_return'], inplace=True)  # 移除NaN收益率

    if merged_df.empty:
        print("移除NaN收益率后，没有可用于绩效计算的数据。")
        return None, None

    # 确保累计收益从1开始
    merged_df['cumulative_return'] = (1 + merged_df['portfolio_return']).cumprod()
    final_cumulative_return = merged_df['cumulative_return'].iloc[-1]

    # 年化收益率
    annualized_return = final_cumulative_return ** (12 / total_months) - 1
    # 年化波动率
    annualized_volatility = merged_df['portfolio_return'].std() * np.sqrt(12)
    # 夏普比率
    sharpe_ratio = (annualized_return - config[
        'RISK_FREE_RATE']) / annualized_volatility if annualized_volatility != 0 else 0

    # 最大回撤
    rolling_max = merged_df['cumulative_return'].expanding().max()
    drawdown = (merged_df['cumulative_return'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 基准年化收益率
    annualized_benchmark_return = (1 + merged_df['benchmark_return']).prod() ** (12 / total_months) - 1
    # 超额收益
    excess_return = merged_df['portfolio_return'] - merged_df['benchmark_return']
    annualized_excess_return = annualized_return - annualized_benchmark_return
    # 跟踪误差
    tracking_error = excess_return.std() * np.sqrt(12)
    # 信息比率
    information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else 0

    metrics = {
        '年化收益率': annualized_return,
        '年化波动率': annualized_volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '年化换手率': annual_turnover,
        '年化超额收益率': annualized_excess_return,
        '信息比率': information_ratio,
        '跟踪误差': tracking_error,
        '基准年化收益率': annualized_benchmark_return,
        '累计收益率': final_cumulative_return - 1
    }

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
        # 将字符串日期转换为Timestamp对象，以便在加载和过滤数据时使用
        start_date_ts = pd.to_datetime(config['BACKTEST_START_DATE'])
        end_date_ts = pd.to_datetime(config['BACKTEST_END_DATE'])

        # --- 根据策略类型选择不同的组合构建函数 ---
        strategy_df = None
        if config['STRATEGY_TYPE'] == 'single':
            strategy_df = build_single_portfolio(
                config['RESIDUAL_FILE'],
                config['RESIDUAL_QUANTILE'],
                config['RESIDUAL_COL'],
                start_date_ts,  # 传递日期参数
                end_date_ts  # 传递日期参数
            )
        elif config['STRATEGY_TYPE'] == 'pbroe4_2_intersection':
            strategy_df = build_pbroe4_2_portfolio(
                config['RESIDUAL_FILE_P3'],
                config['RESIDUAL_FILE_P4'],
                config['RESIDUAL_QUANTILE'],
                start_date_ts,  # 传递日期参数
                end_date_ts  # 传递日期参数
            )
        elif config['STRATEGY_TYPE'] == 'pbroe5_0_fusion':
            strategy_df = build_pbroe5_0_portfolio(
                config['RESIDUAL_FILE_P3'],
                config['RESIDUAL_FILE_P4'],
                config['RESIDUAL_FILE_TS_QUANTILE'],
                config['TS_QUANTILE_COL'],
                config['CROSS_SECTIONAL_QUANTILE'],
                config['TIME_SERIES_QUANTILE_THRESHOLD'],
                start_date_ts,  # 传递日期参数
                end_date_ts  # 传递日期参数
            )
        elif config['STRATEGY_TYPE'] == 'single_ts_quantile':  # 新增的单一时序分位数策略
            strategy_df = build_single_ts_quantile_portfolio(
                config['RESIDUAL_FILE_TS_QUANTILE'],
                config['TS_QUANTILE_COL'],
                config['TIME_SERIES_QUANTILE_THRESHOLD'],
                start_date_ts,
                end_date_ts
            )
        else:
            print(f"错误: 未知的策略类型 '{config['STRATEGY_TYPE']}'")
            return None, None  # 返回 None, None 以便主循环判断

        if strategy_df is None:
            print("未能成功构建投资组合，回测中止。")
            return None, None

        returns_df, benchmark_df = load_and_preprocess_data(
            config['RETURNS_FILE'],
            config['BENCHMARK_FILE'],
            config['BENCHMARK_CODE'],
            start_date_ts,  # 传递日期参数
            end_date_ts  # 传递日期参数
        )
        if returns_df is None or benchmark_df is None:
            print("收益或基准数据加载失败，回测中止。")
            return None, None

        portfolio_returns_df = run_backtest_vectorized(
            strategy_df.copy(),  # 传递副本
            returns_df,
            config['BACKTEST_START_DATE'],
            config['BACKTEST_END_DATE']
        )
        if portfolio_returns_df.empty:
            print("投资组合收益数据为空，无法进行绩效计算。")
            return None, None

        annual_turnover = calculate_annual_turnover_vectorized(
            strategy_df.copy(),  # 传递副本
            config['BACKTEST_START_DATE'],
            config['BACKTEST_END_DATE']
        )

        merged_df, metrics = calculate_performance_and_save(
            portfolio_returns_df,
            benchmark_df,
            annual_turnover,
            config
        )
        print("\n回测流程结束。")
        return merged_df, metrics  # 返回合并后的收益数据和绩效指标

    except Exception as e:
        print(f"\n执行过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None  # 出现异常时也返回 None, None
