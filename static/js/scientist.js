async function runScientist(file) {
    currentCSVFile = file;

    // get columns
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/scientist-columns', { method: 'POST', body: formData });
    const data = await res.json();

    const opts = data.columns.map(c => `<option value="${c}">${c}</option>`).join('');

    addBotRaw(`
        <div class="msg-label">Sainyx</div>
        <div class="msg-bubble">Which column should I predict from <strong>${file.name}</strong>?</div>
        <div class="action-row" style="flex-direction:column;align-items:flex-start;gap:8px;">
            <select class="col-select" id="target-select">
                <option value="">-- select target column --</option>
                ${opts}
            </select>
            <div style="display:flex;gap:8px;">
                <button class="act-btn primary" onclick="startTraining()">⚡ Generate & Train</button>
                <button class="act-btn" onclick="this.closest('.msg-wrap').remove()">✕ Cancel</button>
            </div>
        </div>
    `);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function startTraining() {
    const target = document.getElementById('target-select')?.value;
    if (!target) { alert('Please select a target column.'); return; }

    // remove picker
    document.getElementById('target-select')?.closest('.msg-wrap')?.remove();

    addBotMsg(`Got it — predicting <strong>${target}</strong>. Let me write the code first.`);
    await sleep(600);

    // ── STEP 1: Show code being written ──────────
    const codeWrap = document.createElement('div');
    codeWrap.className = 'msg-wrap bot';
    const codeId = 'code-' + Date.now();
    codeWrap.innerHTML = `
        <div class="msg-label">Sainyx</div>
        <div class="code-block">
            <div class="code-header">
                <span class="code-lang">python</span>
                <span>sklearn · RandomForest</span>
            </div>
            <div class="code-body" id="${codeId}"><span class="code-cursor"></span></div>
        </div>
    `;
    chatBox.appendChild(codeWrap);
    chatBox.scrollTop = chatBox.scrollHeight;

    // type the code live
    const codeEl = document.getElementById(codeId);
    const codeLines = [
        `import pandas as pd`,
        `from sklearn.model_selection import train_test_split`,
        `from sklearn.ensemble import RandomForestClassifier`,
        `from sklearn.preprocessing import LabelEncoder, StandardScaler`,
        `from sklearn.metrics import accuracy_score`,
        ``,
        `# Load dataset`,
        `df = pd.read_csv('${currentCSVFile.name}')`,
        ``,
        `# Prepare features`,
        `X = df.drop(columns=['${target}'])`,
        `y = df['${target}']`,
        ``,
        `# Encode categorical columns`,
        `le = LabelEncoder()`,
        `for col in X.select_dtypes(include='object').columns:`,
        `    X[col] = le.fit_transform(X[col])`,
        ``,
        `# Scale features`,
        `scaler = StandardScaler()`,
        `X_scaled = scaler.fit_transform(X)`,
        ``,
        `# Train/test split`,
        `X_train, X_test, y_train, y_test = train_test_split(`,
        `    X_scaled, y, test_size=0.2, random_state=42`,
        `)`,
        ``,
        `# Train model`,
        `model = RandomForestClassifier(n_estimators=100)`,
        `model.fit(X_train, y_train)`,
        ``,
        `# Evaluate`,
        `y_pred = model.predict(X_test)`,
        `accuracy = accuracy_score(y_test, y_pred)`,
        `print(f'Accuracy: {accuracy:.2%}')`,
    ];

    let fullCode = '';
    for (const line of codeLines) {
        for (const char of line) {
            fullCode += char;
            codeEl.innerHTML = fullCode + '<span class="code-cursor"></span>';
            chatBox.scrollTop = chatBox.scrollHeight;
            await sleep(12);
        }
        fullCode += '\n';
        codeEl.innerHTML = fullCode + '<span class="code-cursor"></span>';
        await sleep(60);
    }

    // remove cursor
    codeEl.innerHTML = fullCode;
    await sleep(400);

    // ── STEP 2: Show execution steps ─────────────
    const stepsWrap = document.createElement('div');
    stepsWrap.className = 'msg-wrap bot';
    stepsWrap.innerHTML = `
        <div class="msg-label">Sainyx</div>
        <div class="steps" id="steps-container">
            <div class="step" id="step-1"><div class="step-icon">○</div><span>Loading dataset</span></div>
            <div class="step" id="step-2"><div class="step-icon">○</div><span>Encoding features</span></div>
            <div class="step" id="step-3"><div class="step-icon">○</div><span>Splitting data</span></div>
            <div class="step" id="step-4"><div class="step-icon">○</div><span>Training RandomForest</span></div>
            <div class="step" id="step-5"><div class="step-icon">○</div><span>Evaluating model</span></div>
            <div class="step" id="step-6"><div class="step-icon">○</div><span>Generating charts</span></div>
        </div>
    `;
    chatBox.appendChild(stepsWrap);
    chatBox.scrollTop = chatBox.scrollHeight;

    // animate steps while training
    const stepDone = async (id, next) => {
        const el = document.getElementById('step-' + id);
        if (el) { el.classList.add('done'); el.querySelector('.step-icon').textContent = '✓'; }
        if (next) {
            const nel = document.getElementById('step-' + next);
            if (nel) nel.classList.add('active');
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    document.getElementById('step-1').classList.add('active');
    startOverlay();

    // ── STEP 3: Actually train ────────────────────
    const trainForm = new FormData();
    trainForm.append('file', currentCSVFile);
    trainForm.append('target', target);

    // animate steps during training
    await stepDone(1, 2); await sleep(400);
    await stepDone(2, 3); await sleep(400);
    await stepDone(3, 4); await sleep(300);

    const trainRes = await fetch('/scientist', { method: 'POST', body: trainForm });
    const result = await trainRes.json();

    await stepDone(4, 5); await sleep(300);
    await stepDone(5, 6); await sleep(300);
    await stepDone(6); await sleep(400);

    stopOverlay();

    // ── STEP 4: Show results ──────────────────────
    if (result.error) {
        addBotMsg('❌ Error: ' + result.error);
        return;
    }

    const scoreColor = result.score > 0.8 ? '#00ff88' : result.score > 0.6 ? '#ffaa00' : '#ff4444';
    const maxImp = Math.max(...Object.values(result.feature_importance));

    const featRows = Object.entries(result.feature_importance).slice(0,8).map(([f,v]) => {
        const pct = ((v/maxImp)*100).toFixed(0);
        return `<tr><td>${f}</td><td><div class="imp-bar" style="width:${pct}%"></div></td><td>${(v*100).toFixed(1)}%</td></tr>`;
    }).join('');

    const chartHtml = (result.charts||[]).map(c =>
        `<div class="inline-chart"><div class="ctitle">${c.title}</div><img src="data:image/png;base64,${c.data}"></div>`
    ).join('');

    const resultCard = document.createElement('div');
    resultCard.className = 'msg-wrap bot';
    resultCard.innerHTML = `
        <div class="result-card">
            <div class="result-card-header">🧪 Model Results — ${target}</div>
            <div class="result-card-body">
                <div class="mini-stats">
                    <div class="mini-stat"><div class="ml">Task</div><div style="margin-top:4px;"><span class="task-badge ${result.task}">${result.task}</span></div></div>
                    <div class="mini-stat"><div class="ml">${result.metric_name}</div><div class="mv" style="color:${scoreColor}">${(result.score*100).toFixed(1)}%</div></div>
                    <div class="mini-stat"><div class="ml">Train</div><div class="mv blue">${result.n_train}</div></div>
                    <div class="mini-stat"><div class="ml">Test</div><div class="mv purple">${result.n_test}</div></div>
                    <div class="mini-stat"><div class="ml">Features</div><div class="mv">${result.features.length}</div></div>
                </div>
                <table class="mini-table">
                    <thead><tr><th>Feature</th><th>Importance</th><th>Score</th></tr></thead>
                    <tbody>${featRows}</tbody>
                </table>
                <div class="charts-row">${chartHtml}</div>
                <div class="result-actions">
                    <button class="act-btn primary" onclick="runScientist(currentCSVFile)">↺ Try Different Target</button>
                    <button class="act-btn" onclick="runAnalysis(currentCSVFile)">📊 Full Analysis</button>
                </div>
            </div>
        </div>
    `;
    chatBox.appendChild(resultCard);
    chatBox.scrollTop = chatBox.scrollHeight;
}