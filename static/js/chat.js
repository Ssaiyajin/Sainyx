// ── STATE ─────────────────────────────────────────
let attachedFile  = null;
let currentMode   = 'chat';
let autoSwitched  = false; // true only when currentMode was reached via an
                            // automatic chat-intent redirect, not a manual tab click
let currentCSVFile = null;

const chatBox = document.getElementById('chat-box');
const input   = document.getElementById('user-input');

// ── CAPABILITY BAR ────────────────────────────────
// manual=true (the default) is what every onclick="setMode('x')" in chat.html
// already calls, so no HTML changes are needed. Auto-routing from sendMessage()
// passes { manual: false } explicitly, which is what makes the later
// "revert to chat" logic able to tell the two cases apart.
function setMode(mode, { manual = true } = {}) {
    currentMode = mode;
    autoSwitched = !manual;
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

    if (mode === 'data' || mode === 'scientist') {
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
    const normalized = message.toLowerCase().replace(/[!?.,]/g, ' ');
    if (/\b(video|animation|animate)\b/.test(normalized) &&
        /\b(make|create|generate|animate|show)\b/.test(normalized)) {
        return 'video';
    }
    if (/\b(voice|speech|audio)\b/.test(normalized) &&
        /\b(make|create|generate|convert|read|speak)\b/.test(normalized)) {
        return 'voice';
    }
    if (/\b(image|picture|drawing|art|illustration)\b/.test(normalized) &&
        /\b(make|create|generate|draw|paint|illustrate|render|design|show)\b/.test(normalized)) {
        return 'image';
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
async function sendMessage() {
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

    startOverlay();

    if (currentMode === 'voice') {
        stopOverlay();
        showRoadmapNotice(currentMode === 'voice' ? 'voice_generation' : 'api_layer');
        return;
    }

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

    // Media requests from Chat temporarily switch to the matching tab.
    const requestedMode = currentMode === 'chat' ? detectRequestMode(message) : null;
    if (requestedMode) {
        setMode(requestedMode, { manual: false });
        try {
            if (requestedMode === 'video') {
                await generateVideo(message);
            } else if (requestedMode === 'image') {
                await generateImage(extractImagePrompt(message));
            } else {
                await generateVoice(message);
            }
        } finally {
            if (autoSwitched) setMode('chat', { manual: false });
        }
        return;
    }

    // Video: manually selected Video mode stays selected after generation.
    if (currentMode === 'video') {
        generateVideo(message);
        return;
    }

    // Image: manually selected Image mode stays selected after generation.
    if (currentMode === 'image') {
        generateImage(message);
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
    const voice = document.getElementById('api-voice-command');
    const powershell = document.getElementById('api-powershell-command');
    const curlCommand = [
        `curl -X POST ${baseUrl}/generate`,
        '  -H "X-API-Key: your-generated-key"',
        '  -H "Content-Type: application/json"',
        '  -d \'{"type":"voice","text":"Hello from Sainyx"}\''
    ].join('\n');
    if (base) base.textContent = baseUrl;
    if (status) status.textContent = `curl ${baseUrl}/status`;
    if (voice) voice.textContent = curlCommand;
    if (powershell) powershell.textContent = `$headers = @{ "X-API-Key" = "your-generated-key" }\nInvoke-RestMethod -Method Post -Uri "${baseUrl}/generate" -Headers $headers -ContentType "application/json" -Body '{"type":"voice","text":"Hello from Sainyx"}'`;
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

function copyApiVoice(button) {
    const baseUrl = getApiBaseUrl();
    copyApiText(button, `curl -X POST ${baseUrl}/generate -H "X-API-Key: your-generated-key" -H "Content-Type: application/json" -d '{"type":"voice","text":"Hello from Sainyx"}'`);
}

function copyApiPowerShell(button) {
    const baseUrl = getApiBaseUrl();
    copyApiText(button, `$headers = @{ "X-API-Key" = "your-generated-key" }; Invoke-RestMethod -Method Post -Uri "${baseUrl}/generate" -Headers $headers -ContentType "application/json" -Body '{"type":"voice","text":"Hello from Sainyx"}'`);
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

