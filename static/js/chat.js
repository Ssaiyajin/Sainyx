// ── STATE ─────────────────────────────────────────
let attachedFile  = null;
let currentMode   = 'chat';
let autoSwitched  = false; // true only when currentMode was reached via an
                            // automatic chat-intent redirect, not a manual tab click
let currentCSVFile = null;
let messageQueue = Promise.resolve();

const chatBox = document.getElementById('chat-box');
const input   = document.getElementById('user-input');

// ── CAPABILITY BAR ────────────────────────────────
// manual=true (the default) is what every onclick="setMode('x')" in chat.html
// already calls, so no HTML changes are needed. Auto-routing from sendMessage()
// passes { manual: false } explicitly, which is what makes the later
// "revert to chat" logic able to tell the two cases apart.
function setMode(mode, { manual = true } = {}) {
    currentMode = mode;
    autoSwitched = !manual && mode !== 'chat';
    const apiGuide = document.getElementById('api-guide');
    const roadmapTitle = document.getElementById('roadmap-title');
    const roadmapList = document.getElementById('roadmap-list');
    if (apiGuide) apiGuide.hidden = mode !== 'api';
    if (roadmapTitle) roadmapTitle.hidden = mode === 'api';
    if (roadmapList) roadmapList.hidden = mode === 'api';
    document.querySelectorAll('.cap-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('cap-' + mode);
    if (btn) btn.classList.add('active');

    const hints = {
        chat:      'Ask anything...',
        data:      'Attach a CSV to analyze →',
        scientist: 'Attach a CSV to train a model →',
        image:     'Describe what to generate... e.g. "Goku ultra instinct, anime art, 4k',
        video:     'Describe the frame to generate... (Tier 1 — single unconditional frame)',
        voice:     'Enter text for voice generation...',
        api:       'Use the API guide above...'
    };
    input.placeholder = hints[mode] || 'Ask anything...';

    if ((mode === 'data' || mode === 'scientist') && !attachedFile) {
        document.getElementById('file-input').click();
    }
}
function showRoadmapNotice(feature) {
    const notices = {
        voice_generation: 'A voice synthesis layer is being prepared for a future release once a trained model is available. <em>Coming soon</em>',
        api_layer: 'A more structured API experience is being organized for future public access. <em>Planned</em>'
    };
    if (notices[feature]) addBotMsg(notices[feature]);
}

function detectRequestMode(message) {
    const normalized = message.toLowerCase().replace(/[!?.,]/g, ' ').trim();
    const asksForInformation = /^(?:(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?)?(?:what|why|how|when|where|who|explain|describe|tell me about|define|meaning|definition)\b/.test(normalized);
    if (asksForInformation) return null;

    const asksToGenerate = /\b(make|create|generate|draw|paint|illustrate|render|design|show|produce|animate|sketch|want|need|like|turn|convert|read|speak|say|play|synthesize)\b/.test(normalized);
    if (/\b(api|api guide|api tab)\b/.test(normalized) &&
        /\b(show|open|view|go to|api guide|api tab)\b/.test(normalized)) {
        return 'api';
    }
    if (/\b(train|training|fit|predict|machine learning)\b/.test(normalized) &&
        /\b(model|machine learning|data|dataset|csv|prediction|predict)\b/.test(normalized)) {
        return 'scientist';
    }
    const asksForAnalysis = /\b(analy[sz]e|analysis|summari[sz]e|visuali[sz]e|explore|inspect|plot|chart|report)\b/.test(normalized);
    const mentionsData = /\b(data|dataset|csv|file|spreadsheet|table|column|this|these|my)\b/.test(normalized);
    if (asksForAnalysis && (mentionsData || /\b(analysis|plot|chart|report)\b/.test(normalized))) {
        return 'data';
    }
    const shortVideoPrompt = normalized.split(/\s+/).length <= 4 &&
        !/\b(?:is|are|was|were|watched|watch|saw|seen|like|liked|love|loved|hate|hated)\b/.test(normalized);
    if (/\b(video|animation|animated|animate|clip|movie)\b/.test(normalized) &&
        (asksToGenerate || shortVideoPrompt)) {
        return 'video';
    }
    const shortVoicePrompt = normalized.split(/\s+/).length <= 6 &&
        /\b(voice|speech|audio)\b/.test(normalized);
    if (/\b(voice|speech|audio)\b/.test(normalized) &&
        (asksToGenerate || shortVoicePrompt)) {
        return 'voice';
    }
    if (/\b(image|picture|drawing|art|illustration|photo)\b/.test(normalized) &&
        (asksToGenerate ||
            /\b[\w'-]+\s+(?:image|picture|drawing|art|illustration|photo)\b/.test(normalized) ||
            /^(?:(?:a|an|the)\s+)?(?:image|picture|drawing|art|illustration|photo)\b/.test(normalized))) {
        return 'image';
    }
    if (/^\s*(please\s+)?(draw|paint|illustrate|sketch)\b/.test(normalized)) {
        return 'image';
    }
    if (/\b(read|speak|say)\b/.test(normalized) && /\b(aloud|out loud|voice|audio|speech)\b/.test(normalized)) {
        return 'voice';
    }
    return null;
}

async function generateVoice(message) {
    addBotMsg(`Voice generation: <em>${message}</em>...`);
    const res = await fetch('/generate-voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message })
    });
    const data = await res.json();
    if (data.error) {
        addBotMsg('❌ ' + data.error);
        return;
    }

    const wrap = addBotMsg(`
        <div>🔊 Voice generation ready</div>
        <audio controls style="width:100%;margin-top:10px;" src="data:${data.mime_type};base64,${data.audio_base64}"></audio>
    `);
    wrap.querySelector('audio')?.play().catch(() => {});
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
input.addEventListener('input', () => {
    if (currentMode === 'api' && input.value.trim()) setMode('chat');
    input.style.height='auto';
    input.style.height=input.scrollHeight+'px';
});
input.addEventListener('keydown', (e) => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

// ── SEND ──────────────────────────────────────────
function sendMessage() {
    messageQueue = messageQueue.then(processMessage).catch(error => {
        stopOverlay();
        removeTyping();
        addBotMsg('❌ ' + (error.message || 'Request failed'));
    });
    return messageQueue;
}

async function processMessage() {
    const message = input.value.trim();
    const file    = attachedFile;

    if (!message && !file) return;

    addUserMsg(message, file);
    input.value = '';
    input.style.height = 'auto';
    removeFile();

    if (currentMode === 'api') {
        setMode('chat');
    }

    const requestedMode = currentMode === 'chat' ? detectRequestMode(message) : null;
    if (requestedMode) {
        setMode(requestedMode, { manual: false });
        if (requestedMode === 'api') {
            showApiGuide();
            return;
        }
    }

    startOverlay();

    if (currentMode === 'voice') {
        try {
            await generateVoice(message);
        } finally {
            stopOverlay();
            if (autoSwitched) setMode('chat');
        }
        return;
    }

    // CSV attached
    if (file && file.name.toLowerCase().endsWith('.csv')) {
        currentCSVFile = file;
        await sleep(500);
        stopOverlay();

        if (currentMode === 'data') {
            try {
                await runAnalysis(file);
            } finally {
                if (autoSwitched) setMode('chat');
            }
        } else if (currentMode === 'scientist') {
            try {
                await runScientist(file);
            } finally {
                if (autoSwitched) setMode('chat');
            }
        } else {
            showCSVOptions(file);
        }
        return;
    }

    if (currentMode === 'data' || currentMode === 'scientist') {
        stopOverlay();
        addBotMsg(currentMode === 'data'
            ? 'Attach a CSV file to analyze it.'
            : 'Attach a CSV file to train a model.');
        return;
    }

    if (currentMode === 'video' || currentMode === 'image') {
        try {
            if (currentMode === 'video') {
                await generateVideo(extractVideoPrompt(message) || message);
            } else {
                await generateImage(extractImagePrompt(message));
            }
        } finally {
            if (autoSwitched) setMode('chat');
        }
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

function showApiGuide() {
    const guide = document.getElementById('api-guide');
    const roadmapTitle = document.getElementById('roadmap-title');
    const roadmap = document.getElementById('roadmap-list');
    const card = document.getElementById('roadmap-card');
    if (guide) guide.hidden = false;
    if (roadmapTitle) roadmapTitle.hidden = true;
    if (roadmap) roadmap.hidden = true;
    updateApiGuide();
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function toggleApiGuide() {
    const guide = document.getElementById('api-guide');
    if (currentMode === 'api' && guide && !guide.hidden) {
        setMode('chat');
        return;
    }
    setMode('api');
    showApiGuide();
}

function updateApiGuide() {
    const baseUrl = getApiBaseUrl();
    const base = document.getElementById('api-base-url');
    const status = document.getElementById('api-status-command');
    const generate = document.getElementById('api-generate-command');
    const powershell = document.getElementById('api-powershell-command');
    const curlCommand = [
        `curl -X POST ${baseUrl}/generate`,
        '  -H "X-API-Key: YOUR_API_KEY"',
        '  -H "Content-Type: application/json"',
        '  -d \'{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}\''
    ].join('\n');
    if (base) base.textContent = baseUrl;
    if (status) status.textContent = `curl ${baseUrl}/status`;
    if (generate) generate.textContent = curlCommand;
    if (powershell) powershell.textContent = `$headers = @{ "X-API-Key" = "YOUR_API_KEY" }\nInvoke-RestMethod -Method Post -Uri "${baseUrl}/generate" -Headers $headers -ContentType "application/json" -Body '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'`;
}

function getApiBaseUrl() {
    const currentUrl = new URL(window.location.href);
    if (currentUrl.hostname === 'huggingface.co') {
        const spaceParts = currentUrl.pathname.split('/').filter(Boolean);
        if (spaceParts[0] === 'spaces' && spaceParts.length >= 3) {
            return `https://${spaceParts[1]}-${spaceParts[2]}.hf.space/api/v1`;
        }
    }
    return `${currentUrl.origin}/api/v1`;
}

function copyApiBaseUrl(button) {
    copyApiText(button, getApiBaseUrl());
}

function copyApiStatus(button) {
    copyApiText(button, `curl ${getApiBaseUrl()}/status`);
}

function copyApiGenerate(button) {
    const baseUrl = getApiBaseUrl();
    copyApiText(button, `curl -X POST ${baseUrl}/generate -H "X-API-Key: YOUR_API_KEY" -H "Content-Type: application/json" -d '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'`);
}

function copyApiPowerShell(button) {
    const baseUrl = getApiBaseUrl();
    copyApiText(button, `$headers = @{ "X-API-Key" = "YOUR_API_KEY" }; Invoke-RestMethod -Method Post -Uri "${baseUrl}/generate" -Headers $headers -ContentType "application/json" -Body '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'`);
}

async function copyApiText(button, text) {
    try {
        await navigator.clipboard.writeText(text);
        const original = button.textContent;
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = original; }, 1200);
    } catch {
        button.textContent = 'Copy failed';
    }
}
