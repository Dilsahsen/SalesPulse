import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import logging
from tabulate import tabulate
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_sales_data(csv_path: Path, drop_invalid: bool = True) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if df.empty:
        raise ValueError("Dataset is empty")

    if "sales" not in df.columns:
        df["sales"] = df["units"] * df["unit_price"]
    
    df = clean_numeric(df, ["units", "unit_price", "sales"], drop_invalid)

    full_dates = pd.date_range(df["date"].min(), df["date"].max())
    missing_dates = full_dates.difference(df["date"])
    if len(missing_dates) > 0 :
        logging.warning(f"Missing dates in data: {missing_dates}")
    
    required_columns = {"date", "product", "units", "unit_price", "sales"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if df["sales"].min() < 0:
        logging.warning("Found negative sales values - this might indicate data quality issues")
    
    if df["units"].min() <= 0:
        logging.warning("Found zero or negative units - removing these rows")
        df = df[df["units"] > 0]
    
    if df["unit_price"].min() <= 0:
        logging.warning("Found zero or negative prices - removing these rows")
        df = df[df["unit_price"] > 0]
    
    date_range = df["date"].max() - df["date"].min()
    logging.info(f"Data spans {date_range.days} days from {df['date'].min().date()} to {df['date'].max().date()}")
    
    logging.info(f"Loaded {len(df)} valid rows from {csv_path}")
    logging.info(f"Products: {df['product'].nunique()}, Date range: {df['date'].nunique()} unique dates")
    
    return df

def clean_numeric(df: pd.DataFrame, numeric_cols: list, drop_invalid=True) -> pd.DataFrame:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logging.warning(f"Found {null_count} invalid values in {col}")
            if drop_invalid:
                df = df.dropna(subset=[col])
    return df

def analyze_totals_by_product(df: pd.DataFrame, column: str = "sales") -> pd.DataFrame:
    totals = (
        df.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    )
    totals.rename(columns={"sales": "total_sales"}, inplace=True)
    totals["percent"] = (totals[f"total_{column}"] / totals[f"total_{column}"].sum()) * 100
    logging.info(f"Top products by {column}:\n{totals}")
    return totals


def monthly_trends(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    monthly = df.set_index("date").resample(freq).agg(
        sales=("sales", "sum"),
        avg_sales=("sales", "mean"),
        transactions=("sales", "count"),
        avg_price=("unit_price", "mean")
    )
    monthly.index.name = "date"
    monthly = monthly.asfreq(freq, fill_value=0)
    monthly["pct_change"] = monthly["sales"].pct_change() * 100
    monthly["sales_per_transaction"] = monthly["sales"] / monthly["transactions"].replace(0, np.nan)
    logging.info (
        f"Trends ({freq}): {len(monthly)} periods from {monthly.index.min().date()} to {monthly.index.max().date()} "
    )
    
    return monthly

def comprehensive_sales_statistics(df: pd.DataFrame) -> dict:
    stats_dict = {}
    stats_dict['basic_stats'] = {
    'total_revenue': df['sales'].sum(),
    'total_transactions': len(df),
    'average_transaction_value': df['sales'].mean(),
    'median_transaction_value': df['sales'].median(),
    'std_transaction_value': df['sales'].std(),
    'min_transaction': df['sales'].min(),
    'max_transaction': df['sales'].max(),
    'coefficient_of_variation': df['sales'].std() / df['sales'].mean() if df['sales'].mean() > 0 else None,
    'null_sales_count': df['sales'].isnull().sum()}

    product_stats = df.groupby('product').agg({
        'sales': ['sum', 'mean', 'count', 'std'],
        'units': ['sum', 'mean'],
        'unit_price': ['mean', 'std']
    }).round(2)
    product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]

    total_sales = product_stats['sales_sum'].sum()
    product_stats['sales_percent'] = (product_stats['sales_sum'] / total_sales * 100).round(2)
    product_stats['avg_transaction_value'] = (product_stats['sales_sum'] / product_stats['sales_count']).round(2)
    product_stats['sales_cv'] = (product_stats['sales_std'] / product_stats['sales_mean']).round(2) 
    stats_dict['product_performance'] = product_stats
        
    df_time = df.copy()
    df_time['year_month'] = df_time['date'].dt.to_period('M')
    df_time['day_of_week'] = df_time['date'].dt.day_name()
    df_time['month'] = df_time['date'].dt.month_name()
    
    monthly_sales = df_time.groupby('year_month')['sales'].sum()
    monthly_growth = monthly_sales.pct_change().dropna()

    quarterly_sales = df_time.groupby(df_time['date'].dt.to_period('Q'))['sales'].sum()
    
    stats_dict['time_analysis'] = {
        'monthly_growth_rate_avg': monthly_growth.mean(),
        'monthly_growth_rate_std': monthly_growth.std(),
        'best_month': monthly_sales.idxmax(),
        'worst_month': monthly_sales.idxmin(),
        'seasonal_coefficient_of_variation': monthly_sales.std() / monthly_sales.mean(),
        'best_quarter': quarterly_sales.idxmax(),
        'worst_quarter': quarterly_sales.idxmin(),
        'weekday_avg': df_time.groupby('day_of_week')['sales'].mean().to_dict()}

    if df['unit_price'].nunique() > 1 and df['units'].nunique() > 1:
        price_elasticity_proxy = stats.pearsonr(df['unit_price'], df['units'])[0]
    else:
        price_elasticity_proxy = 0

    n_top = max(1, int(len(df['product'].unique())*0.2))
    top_20_products_share = df.groupby('product')['sales'].sum().sort_values(ascending=False).head(n_top).sum() / df['sales'].sum()

    stats_dict['customer_insights'] = {
        'avg_items_per_transaction': df['units'].mean(),
        'price_elasticity_proxy': price_elasticity_proxy,
        'revenue_concentration': (df.groupby('product')['sales'].sum().std() / df.groupby('product')['sales'].sum().mean()) ,
        'top_20_products_share' : top_20_products_share }
    return stats_dict


def analyze_product_performance(df: pd.DataFrame) -> pd.DataFrame:
    product_analysis = df.groupby('product').agg({
        'sales': ['sum', 'mean', 'count', 'std'],
        'units': ['sum', 'mean', 'std'],
        'unit_price': ['mean', 'std'],
        'date': ['min', 'max']
    })
    monthly_sales = df.groupby([df['product'], df['date'].dt.to_period('M')])['sales'].sum().unstack(fill_value=0)
    product_analysis = product_analysis.join(monthly_sales)
    product_analysis['peak_month'] = df.groupby(['product', df['date'].dt.month])['sales'].sum().groupby('product').idxmax().apply(lambda x: x[1])
    product_analysis.columns = ['_'.join(col).strip() for col in product_analysis.columns.values]
    
    product_analysis['revenue_share'] = (product_analysis['sales_sum'] / product_analysis['sales_sum'].sum() * 100).round(2)
    product_analysis['avg_monthly_sales'] = (product_analysis['sales_sum'] / 
                                           ((pd.to_datetime(product_analysis['date_max']) - 
                                             pd.to_datetime(product_analysis['date_min'])).dt.days / 30.44)).round(2)
    
    product_analysis['sales_consistency'] = (product_analysis['sales_std'] / product_analysis['sales_mean']).round(3)
    
    column_mapping = {
        'sales_sum': 'total_revenue',
        'sales_mean': 'avg_transaction',
        'sales_count': 'transaction_count',
        'units_sum': 'total_units_sold',
        'units_mean': 'avg_units_per_transaction',
        'unit_price_mean': 'avg_price'
    }
    product_analysis = product_analysis.rename(columns=column_mapping)
    product_analysis['top_20_percent_contributor'] = product_analysis['total_revenue'].cumsum() / product_analysis['total_revenue'].sum() <= 0.2

    return product_analysis.sort_values('total_revenue', ascending=False)


def plot_product_totals(totals_by_product: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    max_idx = totals_by_product["total_sales"].idxmax()
    colors=["#FF9999" if i == max_idx else "#66B2FF" for i in range(len(totals_by_product))]
    bars = plt.bar(totals_by_product["product"], totals_by_product["total_sales"], color=colors)
    total = totals_by_product["total_sales"].sum()
    plt.title("Total Sales by Product\n(Total: ${total:,.0f})")
    plt.xlabel("Product")
    plt.ylabel("Sales ($)")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    for bar, pct in zip(bars, totals_by_product["percent"]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{pct:.1f}%" , ha='center' , va='bottom' )
    output_path = output_dir / "product_total_sales.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    logging.info(f"Saved product totals plot to {output_path}")
    return output_path

def plot_product_pie(totals_by_product, output_dir: Path):
    plt.figure(figsize=(6,6))
    plt.pie(totals_by_product["total_sales"], labels=totals_by_product["product"],
            autopct="%1.1f%%", startangle=140)
    plt.title("Sales Share by Product")
    plt.tight_layout()
    output_path = output_dir / "product_sales_share.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

def plot_monthly_trend(monthly: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    plt.plot(monthly.index, monthly["sales"], marker="o", linestyle="--", color="#4C78A8")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    monthly["pct_change"] = monthly["sales"].pct_change() * 100
    max_idx = monthly["sales"].idxmax()
    max_value = monthly.loc[max_idx, "sales"]
    plt.annotate('Max' ,
    xy=(max_idx, max_value), xytext=(0,10),
    textcoords='offset points' , ha = 'center' , fontsize = 10 , fontweight='bold', color='red',
    arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5))
    plt.title("Monthly Total Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales ($)")
    plt.grid(axis="y" , linestyle="--" , alpha = 0.5)
    plt.tight_layout()
    output_path = output_dir / "monthly_sales_trend.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    return output_path


def create_advanced_visualizations(df: pd.DataFrame, output_dir: Path) -> dict:
    plot_paths = {}
    stats_dict = comprehensive_sales_statistics(df)
    product_analysis = analyze_product_performance(df)
    totals_by_product = analyze_totals_by_product(df)

    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('year_month')['sales'].sum().to_timestamp().to_frame('sales')

    plot_paths['product_totals'] = plot_product_totals(totals_by_product , output_dir)
    plot_paths['product_pie'] = plot_product_pie(totals_by_product , output_dir)
    plot_paths['monthly_trend'] = plot_monthly_trend(monthly , output_dir)
    plot_paths.update(create_advanced_visualizations_core(df, monthly, output_dir))
    plot_paths['dashboard'] = plot_performance_dashboard(stats_dict, product_analysis, output_dir)

    logging.info("All analyses and visualizations created successfully.")
    return plot_paths

    
def create_advanced_visualizations_core(df: pd.DataFrame, monthly: pd.DataFrame, output_dir: Path) -> dict:
    plot_paths={}
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.hist(df['sales'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Sales Distribution')
    ax1.set_xlabel('Sales Amount ($)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['sales'].mean(), color='red', linestyle='--', label=f'Mean: ${df["sales"].mean():.2f}')
    ax1.legend()
    
    df.boxplot(column='sales', by='product', ax=ax2)
    ax2.set_title('Sales Distribution by Product')
    ax2.set_xlabel('Product')
    ax2.set_ylabel('Sales ($)')

    ax3.plot(monthly.index, monthly['sales'], marker='o', linewidth=2)
    ax3.set_title('Monthly Sales Trend')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Sales ($)')
    ax3.grid(True, alpha=0.3)
    
    ax4.scatter(df['units'], df['sales'], alpha=0.6)
    ax4.set_title('Sales vs Units Sold')
    ax4.set_xlabel('Units')
    ax4.set_ylabel('Sales ($)')
    
    corr = df['units'].corr(df['sales'])
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_paths['distribution_analysis'] = output_dir / "sales_distribution_analysis.png"
    plt.savefig(plot_paths['distribution_analysis'], dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    df_pivot = df.copy()
    df_pivot['month'] = df_pivot['date'].dt.strftime('%Y-%m')
    pivot_data = df_pivot.groupby(['product', 'month'])['sales'].sum().unstack(fill_value=0)
    
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Sales ($)'})
    plt.title('Monthly Sales Heatmap by Product')
    plt.xlabel('Month')
    plt.ylabel('Product')
    plt.tight_layout()
    
    plot_paths['heatmap'] = output_dir / "product_monthly_heatmap.png"
    plt.savefig(plot_paths['heatmap'], dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    
    monthly_sales = monthly['sales'].resample('MS').sum()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    axes[0].plot(monthly_sales.index, monthly_sales.values, linewidth=2)
    axes[0].set_title('Original Monthly Sales')
    axes[0].set_ylabel('Sales ($)')
    axes[0].grid(True, alpha=0.3)
    
    rolling_mean = monthly_sales.rolling(window=3, center=True).mean()
    axes[1].plot(monthly_sales.index, rolling_mean.values, color='orange', linewidth=2)
    axes[1].set_title('Trend (3-Month Rolling Average)')
    axes[1].set_ylabel('Sales ($)')
    axes[1].grid(True, alpha=0.3)
    
    seasonal = monthly_sales - rolling_mean
    axes[2].plot(monthly_sales.index, seasonal.values, color='green', linewidth=2)
    axes[2].set_title('Seasonal Component')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Sales ($)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_paths['decomposition'] = output_dir / "time_series_decomposition.png"
    plt.savefig(plot_paths['decomposition'], dpi=150, bbox_inches='tight')
    plt.close()
    logging.info("Advanced visualizations created successfully.")
    
    return plot_paths

def plot_performance_dashboard(stats_dict: dict, product_analysis: pd.DataFrame, output_dir: Path) -> Path:
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    growth = stats_dict['time_analysis']['monthly_growth_rate_avg']
    color = 'green' if growth > 0 else 'red'
    metrics_text = (
        f"KEY METRICS\n\n"
        f"Total Revenue: ${stats_dict['basic_stats']['total_revenue']:,.2f}\n"
        f"Transactions: {stats_dict['basic_stats']['total_transactions']:,}\n"
        f"Avg Transaction: ${stats_dict['basic_stats']['average_transaction_value']:.2f}\n"
        f"Growth Rate: {growth*100:.1f}%/month"
    )
    ax1.text(0.05, 0.6, metrics_text, fontsize=11, family="monospace", va="top")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1:3])
    top_products = product_analysis.head(5)
    labels = top_products['product_name'] if 'product_name' in top_products.columns else top_products.index
    explode = [0.1] + [0 for _ in range(len(top_products) - 1)]
    ax2.pie(top_products['total_revenue'], labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
    ax2.set_title('Revenue Share by Product (Top 5)', fontsize=12, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        product_analysis['total_revenue'],
        product_analysis['sales_consistency'],
        s=product_analysis['total_units_sold'] * 0.5,
        alpha=0.6
    )
    z = np.polyfit(product_analysis['total_revenue'], product_analysis['sales_consistency'], 1)
    p = np.poly1d(z)
    ax3.plot(product_analysis['total_revenue'], p(product_analysis['total_revenue']), "r--")
    ax3.set_xlabel('Total Revenue ($)')
    ax3.set_ylabel('Sales Consistency')
    ax3.set_title('Revenue vs Consistency', fontsize=12, fontweight='bold')


    ax4 = fig.add_subplot(gs[1, 1:])
    top_products = product_analysis.head(6)
    colors = plt.cm.Blues(top_products['total_revenue'] / max(top_products['total_revenue'].max(), 1))
    bars = ax4.bar(range(len(top_products)), top_products['total_revenue'], color=colors)
    ax4.set_title('Top Products by Revenue', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Product')
    ax4.set_ylabel('Revenue ($)')
    ax4.set_xticks(range(len(top_products)))
    ax4.set_xticklabels(top_products.index, rotation=45, ha='right')
    for bar, value in zip(bars, top_products['total_revenue']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'${value:,.0f}', ha='center', va='bottom', fontsize=9)


    ax5 = fig.add_subplot(gs[2, :2])
    ax5.scatter(product_analysis['total_units_sold'], product_analysis['total_revenue'], s=100, alpha=0.7)
    ax5.set_xlabel('Total Units Sold')
    ax5.set_ylabel('Total Revenue ($)')
    ax5.set_title('Units Sold vs Revenue', fontsize=12, fontweight='bold')
    for idx, row in product_analysis.iterrows():
        ax5.annotate(idx, (row['total_units_sold'], row['total_revenue']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)


    ax6 = fig.add_subplot(gs[2, 2])
    ax6.bar(product_analysis.index, product_analysis['transaction_count'], color='#6A5ACD', alpha=0.8)
    ax6.set_title('Transaction Frequency by Product', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Product')
    ax6.set_ylabel('Transactions')
    ax6.tick_params(axis='x', rotation=45)


    plt.suptitle('📊 Sales Performance Dashboard', fontsize=16, fontweight='bold', color='#333333')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    dashboard_path = output_dir / "performance_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    return dashboard_path

    
def fit_linear_time_model(
    monthly: pd.DataFrame, 
    degree: int = 2, 
    test_size: float = 0.2, 
    use_cv: bool = True
) -> tuple[Pipeline, pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    monthly = monthly.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.asfreq('MS').fillna(method='ffill')

    monthly['t'] = np.arange(len(monthly))
    monthly['month'] = monthly.index.month
    monthly['month_sin'] = np.sin(2*np.pi*monthly['month']/12)
    monthly['month_cos'] = np.cos(2*np.pi*monthly['month']/12)
    monthly['lag_1'] = monthly['sales'].shift(1)
    monthly['rolling_3'] = monthly['sales'].rolling(3).mean()
    monthly = monthly.dropna()

    X =  monthly[['t','month_sin','month_cos','lag_1','rolling_3']]
    y = monthly['sales'].values

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('ridge', Ridge(alpha=1.0))])
    
    if use_cv:
        tscv = TimeSeriesSplit(n_splits=4)
        param_grid = {'poly__degree': [1,2], 'ridge__alpha': [0.1,1.0,10.0]}
        grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = pipe.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2=r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    return model, monthly, (X_test, y_test, y_pred)

def plot_forecast_results(model, t: np.ndarray, y: np.ndarray, r2: float, mse: float, output_dir: Path) -> Path:
    plt.figure(figsize=(8,5))
    plt.scatter(t, y, label="Gerçek Satışlar", color="#66B2FF")

    y_pred = model.predict(t)
    plt.plot(t, y_pred, color="#FF9999", linewidth=2, label="Model Tahmini")
    plt.title("Satış Trendi Modeli (Zaman Serisi)")
    plt.xlabel("Ay")
    plt.ylabel("Toplam satış ($)")
    plt.legend()
    plt.text(0.05, 0.95, f"R²={r2:.2f}\nMSE={mse:.2f}",
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
    plt.tight_layout()

    output_path = output_dir / "forecast_results.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    logging.info(f"Saved forecast plot to {output_path}")
    return output_path

def advanced_forecasting_models(monthly: pd.DataFrame) -> dict:

    if len(monthly) < 4:
        raise ValueError("Need at least 4 months for advanced forecasting.")
    

    monthly_sorted = monthly.sort_index().copy()
    monthly_sorted['month_num'] = range(len(monthly_sorted))
    monthly_sorted['month'] = monthly_sorted.index.month
    monthly_sorted['quarter'] = monthly_sorted.index.quarter
    monthly_sorted['sales_lag1'] = monthly_sorted['sales'].shift(1)
    monthly_sorted['sales_lag2'] = monthly_sorted['sales'].shift(2)
    monthly_sorted['rolling_mean_3'] = monthly_sorted['sales'].rolling(3).mean()
    
    monthly_sorted = monthly_sorted.dropna()
    
    if len(monthly_sorted) < 3:
        raise ValueError("Insufficient data after feature engineering.")
    
    
    feature_cols = ['month_num', 'month', 'quarter', 'sales_lag1', 'sales_lag2', 'rolling_mean_3']
    X = monthly_sorted[feature_cols]
    y = monthly_sorted['sales']
    
    split_idx = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {}
    predictions = {}
    metrics = {}

    poly_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2),
        Ridge(alpha=1.0)
    )
    poly_model.fit(X_train, y_train)
    models['polynomial_regression'] = poly_model
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_model.fit(X_train, y_train)
    models['linear_regression'] = linear_model
    
    for name, model in models.items():
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            metrics[name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            y_pred = model.predict(X_train)
            metrics[name] = {
                'mse': mean_squared_error(y_train, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
                'mae': mean_absolute_error(y_train, y_pred),
                'r2': r2_score(y_train, y_pred)
            }
    
    last_row = monthly_sorted.iloc[-1].copy()
    next_features = pd.DataFrame([{
        'month_num': last_row['month_num'] + 1,
        'month': (monthly_sorted.index[-1] + pd.offsets.MonthBegin(1)).month,
        'quarter': (monthly_sorted.index[-1] + pd.offsets.MonthBegin(1)).quarter,
        'sales_lag1': last_row['sales'],
        'sales_lag2': monthly_sorted.iloc[-2]['sales'] if len(monthly_sorted) > 1 else last_row['sales'],
        'rolling_mean_3': monthly_sorted['sales'].tail(3).mean()
    }])
    
    next_month = (monthly_sorted.index[-1] + pd.offsets.MonthBegin(1)).normalize()
    
    for name, model in models.items():
        predictions[name] = float(model.predict(next_features)[0])
    

    predictions['ensemble'] = np.mean(list(predictions.values()))
    
    window = min(3, len(monthly_sorted))
    predictions['moving_average'] = float(monthly_sorted['sales'].tail(window).mean())
    
    return {
        'next_month': next_month.strftime("%Y-%m"),
        'predictions': predictions,
        'model_metrics': metrics,
        'best_model': min(metrics.keys(), key=lambda k: metrics[k]['rmse']) if metrics else 'ensemble'
    }

def plot_forecast_comparison(results: dict, output_dir: Path) -> Path:
    preds = results["predictions"]
    models = list(preds.keys())
    values = list(preds.values())

    plt.figure(figsize=(8,5))
    colors = ["#66B2FF" if m != "ensemble" else "#FF9999" for m in models]
    plt.bar(models, values, color=colors)
    plt.title(f"Model Tabanlı Tahmin Karşılaştırması ({results['next_month']})")
    plt.xlabel("Model")
    plt.ylabel("Tahmini Satışlar ($)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = output_dir / "model_forecast_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    return output_path

def plot_forecast_timeline(monthly: pd.DataFrame, forecast_results: dict, output_dir: Path) -> Path:
    plt.figure(figsize=(9, 5))
    plt.plot(monthly.index, monthly["sales"], marker="o", label="Gerçek Satışlar")

    next_month = pd.to_datetime(forecast_results["next_month"])
    plt.scatter(next_month, forecast_results["predictions"]["ensemble"], color="red", label="Tahmin (Ensemble)")

    plt.title("Gerçek Satışlar ve Gelecek Ay Tahmini")
    plt.xlabel("Tarih")
    plt.ylabel("Satış ($)")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / "forecast_timeline.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
    return output_path


def predict_next_month_sales(monthly: pd.DataFrame) -> dict:
    
    if len(monthly) < 2 :
        raise ValueError("At least 2 months of data are required for forecasting.")

    try:
        logging.info("Running advanced forecasting models...")
        advanced_results = advanced_forecasting_models(monthly)
        results = {
            "status": "success",
            "method": "advanced",
            "next_month": advanced_results["next_month"],
            "predictions": advanced_results["predictions"],
            "best_model": advanced_results["best_model"],
            "model_metrics": advanced_results["model_metrics"]
        }

        best_model_name = results["best_model"]
        best_forecast_value = results["predictions"][best_model_name]

        logging.info(f"Advanced forecasting succeeded with best model: {best_model_name}")
        logging.info(f"Predicted sales for {results['next_month']}: {best_forecast_value:.2f}")

        return results
    
    except Exception as e:
        logging.warning(f"⚠️ Advanced forecasting failed: {e}")
        logging.info("Falling back to simple moving average forecast...")

        try:
            window = min(3, len(monthly))
            ma_forecast = float(monthly['sales'].tail(window).mean())
            last_month = monthly.index.max()
            next_month = (last_month + pd.offsets.MonthBegin(1)).normalize()

            fallback_results = {
                "status": "fallback",
                "method": "moving_average",
                "next_month": next_month.strftime("%Y-%m"),
                "predictions": {
                    "moving_average": ma_forecast
                },
                "best_model": "moving_average",
                "model_metrics": None
            }
            
            logging.info(f"Fallback forecast for {fallback_results['next_month']}: {ma_forecast:.2f}")
            return fallback_results

        except Exception as fallback_error:
            logging.error(f"❌ Fallback forecasting also failed: {fallback_error}")
            raise RuntimeError("Both advanced and fallback forecasting failed.") from fallback_error


def main() -> None:
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "sales_example.csv"
    output_dir = project_root / "outputs"
    ensure_output_dir(output_dir)
    
    print("=== Loading Data ===")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_sales_data(data_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}\n")
    
    print("=== Sales Analysis ===")
    totals_by_product = analyze_totals_by_product(df)
    monthly = monthly_trends(df)

    print("Total sales by product ($) and contribution (%): ")
    print(tabulate(
        totals_by_product,
        headers="keys", tablefmt="fancy_grid", floatfmt=(".2f", ".2f")))
    print()

    print("Monthly sales summary: ")
    print(monthly["sales"].agg(["min", "max", "mean", "sum"]).to_string())
    print()

    print("=== Generating Visualizations ===")
    product_plot_path = plot_product_totals(totals_by_product, output_dir)
    monthly_plot_path = plot_monthly_trend(monthly, output_dir)
    print(f"Saved plots to: {product_plot_path} and {monthly_plot_path}")

    print("=== Predicting Next Month Sales (Basic) ===")
    forecast = predict_next_month_sales(monthly)
    for k, v in forecast.items():
        if k == "next_month":
            continue
        try:
            print(f"  {k.replace('_', ' ').title()} for {forecast['next_month']}: {float(v):,.2f}")
        except (ValueError, TypeError):
            print(f"  {k.replace('_', ' ').title()} for {forecast['next_month']}: {v}")
        

    try:
        next_month_idx = len(monthly)
        plt.figure(figsize=(10,5))
        plt.plot(monthly.index, monthly["sales"], marker="o", label="Historical Sales")
        plt.scatter(
            monthly.index[-1] + pd.offsets.MonthBegin(1),
            forecast["linear_regression"],
            color="red", label="Next Month Forecast"
        )
        plt.title("Monthly Sales Trend with Forecast")
        plt.xlabel("Month")
        plt.ylabel("Sales ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        forecast_plot_path = output_dir / "monthly_sales_forecast.png"
        plt.savefig(forecast_plot_path, dpi=150)
        plt.close()
        print(f"Forecast plot saved to: {forecast_plot_path}")
    except Exception as e:
        print(f"Could not generate forecast plot: {e}")


    print("\n=== Advanced Forecasting Models ===")
    results = advanced_forecasting_models(monthly)

    metrics_df = pd.DataFrame(results["model_metrics"]).T
    metrics_csv_path = output_dir / "model_performance.csv"
    metrics_df.to_csv(metrics_csv_path, index=True)
    print(f"Model performance metrics saved to: {metrics_csv_path}")

    plot_forecast_comparison(results, output_dir)
    plot_forecast_timeline(monthly, results, output_dir)

    model, t, y = fit_linear_time_model(monthly)
    plot_forecast_results(model, t, y, output_dir)

    print(f"\nBest model: {results['best_model']}")
    print(f"Forecast for {results['next_month']}: {results['predictions'][results['best_model']]:.2f}")

if __name__ == "__main__":
    main()

