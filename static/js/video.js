async function generateVideo(prompt) {
    const description = prompt.trim();
    if (!description) {
        addBotMsg('Please describe the video you want to generate.');
        return;
    }

    addBotMsg(`Generating video: <em>${escapeHtml(description)}</em>`);
    showTyping();
    startOverlay();

    try {
        const res = await fetch('/generate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: description })
        });
        const data = await res.json();

        if (!data || typeof data !== 'object') {
            throw new Error('Video generation returned an invalid response.');
        }
        if (!res.ok || data.error) {
            addBotMsg('❌ ' + (data.error || `Video generation failed (${res.status}).`));
            return;
        }
        if (typeof data.gif !== 'string' || !data.gif) {
            throw new Error('Video generation returned no GIF.');
        }

        const wrap = document.createElement('div');
        wrap.className = 'msg-wrap bot';
        wrap.innerHTML = `
            <div class="msg-label">Sainyx</div>
            <div class="result-card" style="max-width:520px;">
                <div class="result-card-header">🎬 Video Generation</div>
                <img class="generated-video" style="width:100%;display:block;" alt="">
                <div style="padding:10px 14px;display:flex;gap:8px;border-top:1px solid var(--border);flex-wrap:wrap;">
                    <button class="act-btn primary" type="button">⬇ Download GIF</button>
                </div>
            </div>
        `;
        wrap.querySelector('.generated-video').src = `data:image/gif;base64,${data.gif}`;
        wrap.querySelector('.generated-video').alt = description;
        wrap.querySelector('.act-btn.primary').addEventListener('click', () => {
            const bytes = Uint8Array.from(atob(data.gif), char => char.charCodeAt(0));
            const videoUrl = URL.createObjectURL(new Blob([bytes], { type: 'image/gif' }));
            const link = document.createElement('a');
            link.href = videoUrl;
            link.download = 'sainyx-video.gif';
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(videoUrl), 0);
        });
        chatBox.appendChild(wrap);
        chatBox.scrollTop = chatBox.scrollHeight;
    } finally {
        removeTyping();
        stopOverlay();
    }
}

function extractVideoPrompt(message) {
    return message
        .replace(/^\s*(?:(?:please|can you|could you|would you|i want you to|i need you to|i want to|i need to|i'd like to|i want|i need|i'd like)\s+)*(?:(?:generate|make|create|show|produce|animate|render|design)\s+)?(?:(?:me|us)\s+)?(?:(?:a|an|the)\s+)?(?:(?:short|animated)\s+)?(?:(?:video|animation|clip|movie)\s+(?:of\s+)?)?/i, '')
        .replace(/\s+(?:video|animation|clip|movie)\s*$/i, '')
        .trim();
}

function escapeHtml(text) {
    return text.replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    })[char]);
}
