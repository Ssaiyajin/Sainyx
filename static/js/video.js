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

// Mirrors isImageRequest() in image.js - same pattern, video-specific triggers.
// Only matters while in 'chat' mode; once video generation is truly conditioned
// on a prompt (past Tier 1), extractVideoPrompt() can be added the same way
// extractImagePrompt() works today.
function isVideoRequest(msg) {
    const triggers = [
        'generate video', 'create video', 'make a video', 'make video',
        'generate a video', 'create a video', 'video of', 'animate this'
    ];
    return triggers.some(t => msg.toLowerCase().includes(t));
}

function downloadVideoFrame(btn) {
    const a = document.createElement('a');
    a.href = 'data:image/png;base64,' + btn.dataset.img;
    a.download = 'sainyx-video-frame-' + Date.now() + '.png';
    a.click();
}