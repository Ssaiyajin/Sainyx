// ── STATE ─────────────────────────────────────────
let attachedFile  = null;
let currentMode   = 'chat';
let currentCSVFile = null;

const chatBox = document.getElementById('chat-box');
const input   = document.getElementById('user-input');

// ── CAPABILITY BAR ────────────────────────────────
function setMode(mode) {
    if (document.getElementById('cap-' + mode)?.classList.contains('soon')) return;
    currentMode = mode;
    document.querySelectorAll('.cap-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('cap-' + mode);
    if (btn) btn.classList.add('active');

    const hints = {
        chat:      'Ask anything...',
        data:      'Attach a CSV to analyze →',
        scientist: 'Attach a CSV to train a model →',
        image:     'Describe what to generate... e.g. "Goku ultra instinct, anime art, 4k',
        video:     'Coming soon...'
    };
    input.placeholder = hints[mode] || 'Ask anything...';

    if (mode === 'data' || mode === 'scientist') {
        document.getElementById('file-input').click();
    }
}

// ── WELCOME ───────────────────────────────────────
function showWelcome() {
    const w = document.createElement('div');
    w.className = 'welcome';
    w.id = 'welcome';
    w.innerHTML = `
        <h2>⚡ Sainyx Online</h2>
        <p>Your AI platform. Chat, analyze data, and train models — all in one place.</p>
        <div class="suggestions">
            <div class="suggestion" onclick="sendSuggestion('Who is Goku?')">Who is Goku?</div>
            <div class="suggestion" onclick="sendSuggestion('Tell me about One Piece')">One Piece</div>
            <div class="suggestion" onclick="sendSuggestion('What is Elden Ring?')">Elden Ring</div>
            <div class="suggestion" onclick="sendSuggestion('Who is Vegeta?')">Vegeta</div>
        </div>
    `;
    chatBox.appendChild(w);
}

function removeWelcome() {
    const w = document.getElementById('welcome');
    if (w) w.remove();
}

// ── FILE ATTACH ───────────────────────────────────
function handleAttach(file) {
    if (!file) return;
    attachedFile = file;
    document.getElementById('fp-name').textContent = file.name;
    document.getElementById('fp-mode').textContent = currentMode === 'scientist' ? 'TRAIN MODEL' : currentMode === 'data' ? 'ANALYZE' : 'ATTACH';
    document.getElementById('file-preview').style.display = 'flex';
    document.getElementById('file-input').value = '';
}

function removeFile() {
    attachedFile = null;
    document.getElementById('file-preview').style.display = 'none';
}

// ── MESSAGE HELPERS ───────────────────────────────
function addUserMsg(text, file) {
    removeWelcome();
    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap user';
    let html = '';
    if (file) html += `<div class="file-bubble"><span class="fi">📄</span><div><div class="fn">${file.name}</div><div class="fs">${(file.size/1024).toFixed(1)} KB</div></div></div>`;
    if (text) html += `<div class="msg-label">You</div><div class="msg-bubble">${text}</div>`;
    wrap.innerHTML = html;
    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addBotMsg(html) {
    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot';
    wrap.innerHTML = `<div class="msg-label">Sainyx</div><div class="msg-bubble">${html}</div>`;
    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
    return wrap;
}

function addBotRaw(html) {
    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot';
    wrap.innerHTML = html;
    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
    return wrap;
}

function showTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot'; wrap.id = 'typing';
    wrap.innerHTML = `<div class="msg-label">Sainyx</div><div class="typing"><span></span><span></span><span></span></div>`;
    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }

function sendSuggestion(text) { input.value = text; sendMessage(); }

// ── INPUT HANDLING ────────────────────────────────
input.addEventListener('input', () => { input.style.height='auto'; input.style.height=input.scrollHeight+'px'; });
input.addEventListener('keydown', (e) => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

// ── SEND ──────────────────────────────────────────
async function sendMessage() {
    const message = input.value.trim();
    const file    = attachedFile;

    if (!message && !file) return;

    addUserMsg(message, file);
    input.value = '';
    input.style.height = 'auto';
    removeFile();

    startOverlay();

    // CSV attached
    if (file && file.name.endsWith('.csv')) {
        currentCSVFile = file;
        await sleep(500);
        stopOverlay();

        if (currentMode === 'data') {
            runAnalysis(file);
        } else if (currentMode === 'scientist') {
            runScientist(file);
        } else {
            showCSVOptions(file);
        }
        return;
    }
    if (currentMode === 'image' || (currentMode === 'chat' && isImageRequest(message))) {
    const prompt = currentMode === 'image' ? message : extractImagePrompt(message);
    generateImage(prompt);
    return;
    }
    
    // text chat
    showTyping();
    const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    removeTyping();

    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot';
    const uid = 'stream-'+Date.now();
    wrap.innerHTML = `<div class="msg-label">Sainyx</div><div class="msg-bubble" id="${uid}"></div>`;
    chatBox.appendChild(wrap);

    const streamDiv = document.getElementById(uid);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value);
        const lines = buffer.split('\n\n');
        buffer = lines.pop();
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const char = line.slice(6);
                if (char === '[DONE]') { stopOverlay(); break; }
                streamDiv.textContent += char;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    }
}

// ── CSV OPTIONS ───────────────────────────────────
function showCSVOptions(file) {
    addBotRaw(`
        <div class="msg-label">Sainyx</div>
        <div class="msg-bubble">I see <strong>${file.name}</strong> — what would you like to do?</div>
        <div class="action-row">
            <button class="act-btn primary" onclick="runAnalysis(currentCSVFile); this.closest('.action-row').remove()">📊 Analyze Data</button>
            <button class="act-btn purple-btn" onclick="runScientist(currentCSVFile); this.closest('.action-row').remove()">🧪 Train a Model</button>
            <button class="act-btn" onclick="this.closest('.msg-wrap').remove()">✕ Cancel</button>
        </div>
    `);
}