"""Train and evaluate a one-off scikit-learn model from an uploaded CSV.

This is tabular supervised learning, not Sainyx neural-model training. The
fitted estimator is used to produce this request's metrics and charts; it is
not saved as a reusable model.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def detect_task(df, target_col):
    """Choose classification for categorical/small-cardinality targets."""
    target = df[target_col]
    if target.dtype == 'object' or target.nunique() <= 10:
        return 'classification'
    return 'regression'

def prepare_data(df, target_col):
    """Encode categorical/boolean columns and separate features from target."""
    df = df.copy()
    
    # encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # encode bool
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def train_model(df, target_col):
    """Fit a Random Forest, score a holdout split, and build result charts."""
    task = detect_task(df, target_col)
    X, y = prepare_data(df, target_col)
    
    # Convert class labels to integers when the selected target is categorical.
    if task == 'classification' and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    
    # Scale the feature matrix before fitting the model.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Keep a fixed holdout set for a repeatable, simple evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Select the estimator and metric based on the inferred target type.
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        metric_name = 'Accuracy'
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        metric_name = 'R² Score'
    
    # Sort feature importances for both the response and the chart.
    importance = dict(zip(X.columns, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # Return chart images inline with this one-off training result.
    charts = []
    
    # feature importance chart
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = list(importance.keys())[:10]
    vals = [importance[c] for c in cols]
    bars = ax.barh(cols[::-1], vals[::-1], color='#00ff88')
    ax.set_facecolor('#0a0a0f')
    fig.patch.set_facecolor('#0a0a0f')
    ax.tick_params(colors='white')
    ax.set_title('Feature Importance', color='white')
    ax.set_xlabel('Importance', color='#888')
    charts.append(('Feature Importance', fig_to_base64(fig)))
    plt.close()

    # prediction vs actual
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_test, y_pred, color='#00aaff', alpha=0.6, s=40)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
            color='#00ff88', linewidth=2, linestyle='--')
    ax.set_facecolor('#0a0a0f')
    fig.patch.set_facecolor('#0a0a0f')
    ax.tick_params(colors='white')
    ax.set_title('Predicted vs Actual', color='white')
    ax.set_xlabel('Actual', color='#888')
    ax.set_ylabel('Predicted', color='#888')
    charts.append(('Predicted vs Actual', fig_to_base64(fig)))
    plt.close()

    return {
        'task': task,
        'metric_name': metric_name,
        'score': round(float(score), 4),
        'feature_importance': {k: round(float(v), 4) for k, v in importance.items()},
        'n_train': len(X_train),
        'n_test': len(X_test),
        'target': target_col,
        'features': list(X.columns),
        'charts': [{'title': t, 'data': d} for t, d in charts]
    }

def fig_to_base64(fig):
    """Encode a Matplotlib figure as PNG for the browser result view."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')