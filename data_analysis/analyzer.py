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
    
    if not numeric_cols:
        return charts
    
    # histogram for first numeric column
    fig, ax = plt.subplots(figsize=(8, 4))
    df[numeric_cols[0]].hist(ax=ax, bins=20, color='#00ff88', edgecolor='black')
    ax.set_facecolor('#0a0a0f')
    fig.patch.set_facecolor('#0a0a0f')
    ax.tick_params(colors='white')
    ax.set_title(f'Distribution of {numeric_cols[0]}', color='white')
    charts.append(fig_to_base64(fig))
    plt.close()

    # correlation heatmap if multiple numeric cols
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        im = ax.imshow(corr, cmap='coolwarm')
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, color='white')
        ax.set_yticklabels(numeric_cols, color='white')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_title('Correlation Matrix', color='white')
        plt.colorbar(im, ax=ax)
        charts.append(fig_to_base64(fig))
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
        lines.append(f"Missing values found in: {', '.join(missing.keys())}.")
    else:
        lines.append("No missing values found.")
    
    if 'stats' in report:
        for col, stats in list(report['stats'].items())[:2]:
            mean = stats.get('mean', 0)
            lines.append(f"{col}: average is {mean:.2f}.")
    
    return ' '.join(lines)