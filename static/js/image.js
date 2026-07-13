async function generateImage(prompt) {
    if (!prompt || prompt.trim().length < 3) {
        addBotMsg('Please describe what you want to generate. Example: <em>Goku in ultra instinct form, anime art style, detailed, 4k</em>');
        return;
    }

    addBotMsg(`Generating: <em>${prompt}</em>`);
    showTyping();
    startOverlay();

    const res = await fetch('/generate-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
    });

    const data = await res.json();
    removeTyping();
    stopOverlay();

    if (data.error) {
        addBotMsg('❌ ' + data.error);
        return;
    }

    const safePrompt = prompt.replace(/'/g, "\\'").slice(0, 50);

    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot';
    wrap.innerHTML = `
        <div class="msg-label">Sainyx</div>
        <div class="result-card" style="max-width:520px;">
            <div class="result-card-header">🎨 Image Generation</div>
            <img src="data:image/png;base64,${data.image}"
                 style="width:100%;display:block;"
                 alt="${prompt}">
            <div style="padding:10px 14px;display:flex;gap:8px;border-top:1px solid var(--border);flex-wrap:wrap;">
                <button class="act-btn primary" onclick="downloadImage(this)">⬇ Download</button>
                <button class="act-btn" onclick="generateImage('${safePrompt}')">↺ Regenerate</button>
                <button class="act-btn" onclick="generateImage('${safePrompt}, game asset, transparent background')">🎮 Game Asset</button>
                <button class="act-btn" onclick="generateImage('${safePrompt}, concept art')">🎭 Concept Art</button>
            </div>
        </div>
    `;

    // store b64 on the button for download
    const dlBtn = wrap.querySelector('.act-btn.primary');
    dlBtn.dataset.img = data.image;
    dlBtn.dataset.name = prompt.slice(0, 20).replace(/\s+/g, '-');

    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function downloadImage(btn) {
    const a = document.createElement('a');
    a.href = 'data:image/png;base64,' + btn.dataset.img;
    a.download = 'sainyx-' + btn.dataset.name + '.png';
    a.click();
}

function isImageRequest(msg) {
    const triggers = [
        'generate', 'draw', 'create image', 'make image',
        'paint', 'illustrate', 'show me', 'render', 'design'
    ];
    return triggers.some(t => msg.toLowerCase().includes(t));
}

function extractImagePrompt(msg) {
    return msg
        .replace(/generate|draw|create|make|paint|illustrate|render|show me|design|an image of|a picture of|image of|picture of/gi, '')
        .trim();
}