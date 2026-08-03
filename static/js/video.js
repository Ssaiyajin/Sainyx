async function generateVideo(message) {
    addBotMsg(`Generating frame${message ? ': <em>' + message + '</em>' : ''}...`);
    showTyping();
    startOverlay();

    const res = await fetch('/generate-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });

    const data = await res.json();
    removeTyping();
    stopOverlay();

    if (data.error) {
        addBotMsg('❌ ' + data.error);
        return;
    }

    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot';
    wrap.innerHTML = `
        <div class="msg-label">Sainyx</div>
        <div class="result-card" style="max-width:520px;">
            <div class="result-card-header">🎬 Video Gen (Tier 1 — single frame)</div>
            <img src="data:image/png;base64,${data.image}"
                 style="width:100%;display:block;" alt="video frame">
            <div style="padding:10px 14px;display:flex;gap:8px;border-top:1px solid var(--border);flex-wrap:wrap;">
                <button class="act-btn primary" onclick="downloadVideoFrame(this)">⬇ Download</button>
                <button class="act-btn" onclick="generateVideo('')">↺ Regenerate</button>
            </div>
        </div>
    `;

    const dlBtn = wrap.querySelector('.act-btn.primary');
    dlBtn.dataset.img = data.image;

    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function downloadVideoFrame(btn) {
    const a = document.createElement('a');
    a.href = 'data:image/png;base64,' + btn.dataset.img;
    a.download = 'sainyx-video-frame-' + Date.now() + '.png';
    a.click();
}