import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # no display needed
import io
import base64
import os

def analyze_csv(filepath):
    df = pd.read_csv(filepath)
    
    report = {}
    
    # basic info
    report['rows']    = len(df)
    report['columns'] = len(df.columns)
    report['col_names'] = list(df.columns)
    
    # types
    report['dtypes'] = {col: str(df[col].dtype) for col in df.columns}
    
    # missing values
    report['missing'] = df.isnull().sum().to_dict()
    
    # numeric stats
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        report['stats'] = df[numeric_cols].describe().to_dict()
    
    # top values for categorical
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    report['top_values'] = {}
    for col in cat_cols[:3]:
        report['top_values'][col] = df[col].value_counts().head(5).to_dict()
    
    return df, report

def generate_charts(df):
    charts = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # 1 — distribution histogram for first numeric col
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[numeric_cols[0]].hist(ax=ax, bins=20, color='#00ff88', edgecolor='#000')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        ax.set_title(f'Distribution of {numeric_cols[0]}', color='white')
        ax.set_xlabel(numeric_cols[0], color='#888')
        ax.set_ylabel('Count', color='#888')
        charts.append(('Distribution', fig_to_base64(fig)))
        plt.close()

    # 2 — bar chart for top categorical column
    if cat_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        top = df[cat_cols[0]].value_counts().head(10)
        bars = ax.bar(top.index, top.values, color='#00aaff', edgecolor='#000')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white', axis='both')
        plt.xticks(rotation=45, ha='right', color='white', fontsize=9)
        ax.set_title(f'Top values in {cat_cols[0]}', color='white')
        ax.set_ylabel('Count', color='#888')
        charts.append(('Top Values', fig_to_base64(fig)))
        plt.close()

    # 3 — scatter plot for first two numeric cols
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]],
                   color='#ff00ff', alpha=0.7, edgecolors='#000', s=60)
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        ax.set_xlabel(numeric_cols[0], color='#888')
        ax.set_ylabel(numeric_cols[1], color='#888')
        ax.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}', color='white')
        charts.append(('Scatter', fig_to_base64(fig)))
        plt.close()

    # 4 — line chart for numeric trend
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df[numeric_cols[0]].values, color='#00ff88',
                linewidth=2, marker='o', markersize=4, markerfacecolor='#00aaff')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        ax.set_title(f'{numeric_cols[0]} trend', color='white')
        ax.set_xlabel('Index', color='#888')
        ax.set_ylabel(numeric_cols[0], color='#888')
        charts.append(('Trend', fig_to_base64(fig)))
        plt.close()

    # 5 — correlation heatmap
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', color='white', fontsize=9)
        ax.set_yticklabels(numeric_cols, color='white', fontsize=9)
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_title('Correlation Matrix', color='white')
        plt.colorbar(im, ax=ax)
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', color='white', fontsize=8)
        charts.append(('Correlation', fig_to_base64(fig)))
        plt.close()

    return charts

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def summarize(report):
    lines = []
    lines.append(f"Dataset has {report['rows']:,} rows and {report['columns']} columns.")

    missing = {k: v for k, v in report['missing'].items() if v > 0}
    if missing:
        total_missing = sum(missing.values())
        lines.append(f"Found {total_missing} missing values across {len(missing)} columns: {', '.join(missing.keys())}.")
    else:
        lines.append("No missing values found — dataset is complete.")

    if 'stats' in report:
        for col, stats in list(report['stats'].items())[:3]:
            mean = stats.get('mean', 0)
            mn   = stats.get('min', 0)
            mx   = stats.get('max', 0)
            lines.append(f"{col}: average {mean:.2f}, range {mn:.0f}–{mx:.0f}.")

    if report.get('top_values'):
        for col, vals in list(report['top_values'].items())[:1]:
            top = list(vals.keys())[0]
            lines.append(f"Most common {col}: '{top}'.")

    return ' '.join(lines)