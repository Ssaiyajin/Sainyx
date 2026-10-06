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
    const bytes = Uint8Array.from(atob(btn.dataset.img), char => char.charCodeAt(0));
    const imageUrl = URL.createObjectURL(new Blob([bytes], { type: 'image/png' }));
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = 'sainyx-' + (btn.dataset.name || 'image') + '.png';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(imageUrl);
}

function extractImagePrompt(msg) {
    return msg
        .replace(/^\s*(?:(?:please|can you|could you|would you|i want you to|i need you to|i want to|i need to|i'd like to|i want|i need|i'd like)\s+)*(?:(?:generate|draw|create|make|paint|illustrate|render|show|design|sketch)\s+)?(?:(?:me|us)\s+)?(?:(?:an?|the)\s+)?(?:(?:image|picture|drawing|art|illustration|photo)\s+(?:of\s+)?)?/i, '')
        .trim();
}