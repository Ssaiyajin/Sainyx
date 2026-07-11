async function runAnalysis(file) {
    addBotMsg('Analyzing <strong>' + file.name + '</strong>...');
    showTyping();
    startOverlay();

    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await res.json();

    removeTyping();
    stopOverlay();

    addBotMsg(data.summary);

    const maxMissing = Object.values(data.report.missing).some(v=>v>0);
    const numericCount = Object.values(data.report.dtypes).filter(t=>t.includes('int')||t.includes('float')).length;

    const colRows = data.report.col_names.map(col => {
        const m = data.report.missing[col] || 0;
        return `<tr><td>${col}</td><td>${data.report.dtypes[col]}</td><td class="${m>0?'err-text':'ok-text'}">${m>0?m+' missing':'✓'}</td></tr>`;
    }).join('');

    const chartHtml = (data.charts||[]).map(c =>
        `<div class="inline-chart"><div class="ctitle">${c.title}</div><img src="data:image/png;base64,${c.data}"></div>`
    ).join('');

    // store for PDF
    window._lastAnalysis = data;

    const card = document.createElement('div');
    card.className = 'msg-wrap bot';
    card.innerHTML = `
        <div class="result-card">
            <div class="result-card-header">📊 Analysis — ${file.name}</div>
            <div class="result-card-body">
                <div class="mini-stats">
                    <div class="mini-stat"><div class="ml">Rows</div><div class="mv">${data.report.rows.toLocaleString()}</div></div>
                    <div class="mini-stat"><div class="ml">Columns</div><div class="mv">${data.report.columns}</div></div>
                    <div class="mini-stat"><div class="ml">Missing</div><div class="mv" style="color:${maxMissing?'#ff4444':'#00ff88'}">${Object.values(data.report.missing).reduce((a,b)=>a+b,0)}</div></div>
                    <div class="mini-stat"><div class="ml">Numeric</div><div class="mv blue">${numericCount}</div></div>
                </div>
                <table class="mini-table">
                    <thead><tr><th>Column</th><th>Type</th><th>Status</th></tr></thead>
                    <tbody>${colRows}</tbody>
                </table>
                <div class="charts-row">${chartHtml}</div>
                <div class="result-actions">
                    <button class="act-btn primary" onclick="downloadPDF()">⬇ PDF Report</button>
                    <button class="act-btn purple-btn" onclick="runScientist(currentCSVFile)">🧪 Train a Model</button>
                </div>
            </div>
        </div>
    `;
    chatBox.appendChild(card);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function downloadPDF() {
    const data = window._lastAnalysis;
    if (!data) return;
    const res = await fetch('/download-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report: data.report, summary: data.summary, charts: data.charts })
    });
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sainyx_report.pdf';
    a.click();
}