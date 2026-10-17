// ═══════════════════════════════════════════════════════════════
//  Paes Translator — app.js
// ═══════════════════════════════════════════════════════════════

let recognition = null, isRecording = false, recordingTimer = null, recordingStart = null;
let trainingPollInterval = null, downloadPollInterval = null, trainingLogLines = [];

async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        const languages = await response.json();
        const selectors = [document.getElementById('sourceLanguage'), document.getElementById('targetLanguage')];
        selectors.forEach(selector => {
            if (!selector) return;
            selector.innerHTML = '';
            languages.filter(language => String(language.active).toLowerCase() !== '0' && String(language.active).toLowerCase() !== 'false')
                .forEach(language => {
                    const option = document.createElement('option');
                    option.value = language.code;
                    option.textContent = language.name;
                    selector.appendChild(option);
                });
        });
        if (selectors[1]?.options.length > 1) selectors[1].selectedIndex = 1;
        updateLanguageLabels();
    } catch (error) {
        showError('No se pudo cargar el catálogo de idiomas');
    }
}

function updateLanguageLabels() {
    const source = document.getElementById('sourceLanguage');
    const target = document.getElementById('targetLanguage');
    const sourceLabel = document.getElementById('srcLangLabel');
    const targetLabel = document.getElementById('tgtLangLabel');
    if (sourceLabel) sourceLabel.textContent = source?.selectedOptions[0]?.textContent || 'Origen';
    if (targetLabel) targetLabel.textContent = target?.selectedOptions[0]?.textContent || 'Destino';
}

// ── NAVIGATION ────────────────────────────────────────────────

const SECTIONS = ['textInputSection','addWordSection','mediaSection','trainingSection'];

function showSection(activeId) {
    SECTIONS.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        if (id === activeId) {
            el.classList.remove('hidden');
            el.classList.add('active-section');
        } else {
            el.classList.add('hidden');
            el.classList.remove('active-section');
        }
    });
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === activeId);
    });
    if (activeId === 'trainingSection') { refreshModelInfo(); refreshTrainingStatus(); }
    if (activeId === 'mediaSection') {
        const tag = document.getElementById('inputText')?.value.trim() || '';
        const mt = document.getElementById('mediaTag');
        if (mt) mt.value = tag;
        refreshMedia('image', tag); refreshMedia('audio', tag);
    }
    if (activeId === 'textInputSection' && !recognition) initSpeech();
}

// Show text input by default (CSS sets active-section, JS takes over)
function showTextInput() { showSection('textInputSection'); }
function showAddWordInput() { showSection('addWordSection'); }
function showMediaInput() { showSection('mediaSection'); }
function showTrainingInput() { showSection('trainingSection'); }

// ── THEME ─────────────────────────────────────────────────────

function applyTheme(t) {
    document.documentElement.classList.toggle('light-mode', t === 'light');
    document.body.classList.toggle('light-mode', t === 'light');
    const dark = document.getElementById('themeIconDark');
    const light = document.getElementById('themeIconLight');
    if (dark) dark.style.display = t === 'light' ? 'none' : 'block';
    if (light) light.style.display = t === 'light' ? 'block' : 'none';
}

// ── TRANSLATION ───────────────────────────────────────────────

function setTranslateLoading(loading) {
    const btn     = document.getElementById('translateText');
    const label   = document.getElementById('translateBtnLabel');
    const spinner = document.getElementById('translateSpinner');
    if (btn) {
        btn.disabled = loading;
        btn.classList.toggle('loading', loading);
        btn.setAttribute('aria-busy', loading ? 'true' : 'false');
    }
    if (label)   label.textContent = loading ? 'Traduciendo...' : 'Traducir';
    if (spinner) spinner.classList.toggle('hidden', !loading);
}

async function doTranslate() {
    const btn  = document.getElementById('translateText');
    if (btn && btn.disabled) return;
    const text = document.getElementById('inputText').value.trim();
    const src  = document.getElementById('sourceLanguage').value;
    const tgt  = document.getElementById('targetLanguage').value;
    if (!text) { showError('Por favor ingrese texto para traducir'); return; }
    if (src === tgt) { showError('Los idiomas de origen y destino no pueden ser iguales'); return; }

    clearError();
    setTranslateLoading(true);
    document.getElementById('translationProgress').classList.remove('hidden');

    try {
        const res  = await fetch('/api/translate-text', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ text, source_lang: src, target_lang: tgt })
        });
        const data = await res.json();
        if (data.error) { showError(data.error); return; }

        document.getElementById('translatedText').value = data.translation;
        document.getElementById('originalText').value   = text;

        const badge = document.getElementById('methodBadge');
        if (badge) {
            badge.textContent = data.method || 'dict';
            badge.classList.remove('hidden');
        }
        const copyBtn = document.getElementById('copyTranslationBtn');
        if (copyBtn) copyBtn.classList.remove('hidden');
        document.getElementById('feedbackSection').classList.remove('hidden');

        // Update lang labels
        updateLanguageLabels();

    } catch(e) {
        showError('Error de conexión al traducir');
    } finally {
        document.getElementById('translationProgress').classList.add('hidden');
        setTranslateLoading(false);
    }
}

// ── SPEECH ────────────────────────────────────────────────────

function initSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { showError('Tu navegador no soporta reconocimiento de voz. Usa Chrome o Edge.'); return false; }
    recognition = new SR();
    recognition.continuous = true; recognition.interimResults = true;
    recognition.lang = document.getElementById('sourceLanguage')?.value || navigator.language;
    recognition.onstart = () => setVoiceUI(true);
    recognition.onend   = () => { if (isRecording) stopRecording(); };
    recognition.onerror = e => { showError('Error de voz: ' + e.error); stopRecording(); };
    recognition.onresult = e => {
        let final = '';
        for (let i = e.resultIndex; i < e.results.length; i++)
            if (e.results[i].isFinal) final += e.results[i][0].transcript;
        if (final) {
            document.getElementById('inputText').value = final;
            setTimeout(doTranslate, 800);
        }
    };
    return true;
}

function startRecording() {
    if (!recognition && !initSpeech()) return;
    clearError();
    recognition.start();
    isRecording = true; recordingStart = Date.now();
    // Show inline recording UI
    const micBtn = document.getElementById('startRecording');
    micBtn.classList.add('recording');
    document.getElementById('recordingIndicator').classList.add('recording');
    document.getElementById('recordingStatus').classList.remove('hidden');
    document.getElementById('recordingTime').classList.remove('hidden');
    document.getElementById('stopRecording').classList.remove('hidden');
    recordingTimer = setInterval(() => {
        const s = Math.floor((Date.now() - recordingStart) / 1000);
        document.getElementById('recordingTime').textContent =
            `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
    }, 1000);
}

function stopRecording() {
    if (recognition && isRecording) recognition.stop();
    isRecording = false;
    clearInterval(recordingTimer); recordingTimer = null;
    // Hide inline recording UI
    const micBtn = document.getElementById('startRecording');
    micBtn.classList.remove('recording');
    document.getElementById('recordingIndicator').classList.remove('recording');
    document.getElementById('recordingStatus').classList.add('hidden');
    document.getElementById('recordingTime').classList.add('hidden');
    document.getElementById('recordingTime').textContent = '00:00';
    document.getElementById('stopRecording').classList.add('hidden');
}

function setVoiceUI(active) {
    const micBtn = document.getElementById('startRecording');
    const dot  = document.getElementById('recordingIndicator');
    const stat = document.getElementById('recordingStatus');
    if (micBtn) micBtn.classList.toggle('recording', active);
    if (dot)  dot.classList.toggle('recording', active);
    if (stat) {
        stat.textContent = active ? 'Escuchando...' : '';
        stat.classList.toggle('hidden', !active);
    }
}

// ── ADD WORD ──────────────────────────────────────────────────

async function addNewWord() {
    const sourceText = document.getElementById('sourceText').value.trim();
    const targetText = document.getElementById('targetText').value.trim();
    const sourceLang = document.getElementById('sourceLanguage').value;
    const targetLang = document.getElementById('targetLanguage').value;
    const statusEl  = document.getElementById('addWordStatus');

    if (!sourceText || !targetText || !sourceLang || !targetLang) {
        showToast(statusEl, 'Por favor complete todos los campos.', 'error'); return;
    }
    try {
        const res  = await fetch('/add_word', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ source_text: sourceText, target_text: targetText, source_lang: sourceLang, target_lang: targetLang })
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('sourceText').value = '';
            document.getElementById('targetText').value = '';
            document.getElementById('wordContext').value = '';
            showToast(statusEl, 'Par lingüístico agregado al diccionario', 'success');
            rebuildDatasetSilently();
        } else {
            showToast(statusEl, data.error || 'Error al agregar', 'error');
        }
    } catch(e) {
        showToast(statusEl, 'Error de conexión', 'error');
    }
}

function showToast(el, msg, type) {
    if (!el) return;
    el.textContent = msg;
    el.className = 'status-toast ' + type;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 5000);
}

// ── FEEDBACK ──────────────────────────────────────────────────

async function submitFeedback() {
    const corrected = document.getElementById('correctedTranslation').value.trim();
    const original  = document.getElementById('inputText').value.trim();
    const src = document.getElementById('sourceLanguage').value;
    const tgt = document.getElementById('targetLanguage').value;
    if (!corrected) { showError('Ingresa la traducción corregida'); return; }
    if (!original)  { showError('No hay texto original'); return; }

    try {
        const res  = await fetch('/api/feedback', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ original_text: original, corrected_translation: corrected, source_lang: src, target_lang: tgt })
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('correctedTranslation').value = '';
            const msg = document.createElement('div');
            msg.style.cssText='background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);color:#10b981;padding:.65rem 1rem;border-radius:8px;font-size:.83rem;margin-top:.5rem;';
            msg.textContent = '✓ Corrección enviada. ¡Gracias por contribuir!';
            document.querySelector('.feedback-form').appendChild(msg);
            setTimeout(() => msg.remove(), 5000);
            rebuildDatasetSilently();
        } else {
            showError(data.error || 'Error al enviar');
        }
    } catch(e) { showError('Error de conexión'); }
}

// ── MEDIA ─────────────────────────────────────────────────────

async function uploadMedia(type) {
    const fileEl = document.getElementById(type === 'image' ? 'imageFile' : 'audioFile');
    if (!fileEl.files[0]) { showError('Selecciona un archivo'); return; }
    const tag = document.getElementById('mediaTag')?.value.trim() || '';
    const fd  = new FormData();
    fd.append('file', fileEl.files[0]); fd.append('type', type); fd.append('tag', tag);
    try {
        const res  = await fetch('/api/upload-media', { method:'POST', body: fd });
        const data = await res.json();
        if (data.error) { showError(data.error); return; }
        fileEl.value = '';
        refreshMedia(type, tag);
    } catch(e) { showError('Error al subir archivo'); }
}

async function refreshMedia(type, tag) {
    try {
        const url  = tag ? `/api/media-list?type=${type}&tag=${encodeURIComponent(tag)}` : `/api/media-list?type=${type}`;
        const res  = await fetch(url);
        const data = await res.json();
        if (type === 'image') renderImages(data.files || []);
        else renderAudios(data.files || []);
    } catch(e) {}
}

function renderImages(files) {
    const el = document.getElementById('imageGallery'); if (!el) return;
    el.innerHTML = files.length ? '' : '<div class="empty-state">Sin imágenes</div>';
    files.forEach(f => {
        const img = document.createElement('img');
        img.src = f.url; img.alt = f.filename; img.loading = 'lazy';
        el.appendChild(img);
    });
}

function renderAudios(files) {
    const el = document.getElementById('audioList'); if (!el) return;
    el.innerHTML = files.length ? '' : '<div class="empty-state">Sin audios</div>';
    files.forEach(f => {
        const wrap  = document.createElement('div'); wrap.className = 'audio-item';
        const audio = document.createElement('audio'); audio.controls = true; audio.src = f.url;
        const lbl   = document.createElement('div'); lbl.className = 'audio-label'; lbl.textContent = f.filename;
        wrap.append(audio, lbl); el.appendChild(wrap);
    });
}

// ── TRAINING ─────────────────────────────────────────────────

async function refreshModelInfo() {
    try {
        const [mr, dr] = await Promise.all([fetch('/api/model-info'), fetch('/api/training/download-status')]);
        const md = await mr.json(); const dd = await dr.json();
        const info = md.model_info || {}; const ds = dd.status || {};
        const map = {'fine-tuned':'🧠 Fine-tuneado','base':'📦 Base','none':'❌ No cargado'};
        setText('infoModelType',  map[info.model_type] || info.model_type || '—');
        setText('infoDictEntries', info.dictionary_entries ?? '—');
        setText('infoDevice',      info.device || '—');
        setText('infoWeightsReady', ds.model_ready ? '✅ Listos' : '⬇️ Pendiente');
    } catch(e) {}
}

async function startModelDownload() {
    const btn = document.getElementById('downloadModelBtn');
    const msg = document.getElementById('downloadStatusMsg');
    const bar = document.getElementById('downloadStatusBar');
    btn.disabled = true; btn.textContent = 'Iniciando...';
    setStep(msg, 'Iniciando descarga...', 'info');
    bar.classList.remove('hidden');
    try {
        const res  = await fetch('/api/training/download-model', { method:'POST' });
        const data = await res.json();
        if (data.already_downloaded) {
            setStep(msg, '✅ Modelo ya descargado.', 'success');
            btn.textContent = '✅ Ya descargado'; bar.classList.add('hidden'); return;
        }
        if (!data.success) {
            setStep(msg, data.message || 'Error', 'error');
            btn.disabled = false; btn.textContent = 'Descargar modelo'; return;
        }
        setStep(msg, 'Descargando (~2.4 GB)...', 'info');
        pollDownload();
    } catch(e) {
        setStep(msg, 'Error de conexión', 'error');
        btn.disabled = false; btn.textContent = 'Descargar modelo';
    }
}

function pollDownload() {
    if (downloadPollInterval) clearInterval(downloadPollInterval);
    downloadPollInterval = setInterval(async () => {
        try {
            const s = (await (await fetch('/api/training/download-status')).json()).status || {};
            const pct = s.progress || 0;
            document.getElementById('downloadProgressFill').style.width = pct + '%';
            document.getElementById('downloadProgressText').textContent  = pct + '%';
            const msg = document.getElementById('downloadStatusMsg');
            const btn = document.getElementById('downloadModelBtn');
            if (s.state === 'done' || s.state === 'already_downloaded') {
                clearInterval(downloadPollInterval);
                setStep(msg, '✅ Modelo descargado.', 'success');
                btn.textContent = '✅ Ya descargado';
                document.getElementById('downloadStatusBar').classList.add('hidden');
                setText('infoWeightsReady','✅ Listos');
            } else if (s.state === 'error') {
                clearInterval(downloadPollInterval);
                setStep(msg, '❌ ' + s.message, 'error');
                btn.disabled = false; btn.textContent = 'Reintentar';
            } else { setStep(msg, s.message || 'Descargando...', 'info'); }
        } catch(e) {}
    }, 3000);
}

async function startTraining() {
    const btn    = document.getElementById('startTrainingBtn');
    const stopBtn= document.getElementById('stopTrainingBtn');
    const msg    = document.getElementById('trainingStatusMsg');
    btn.disabled = true; setStep(msg, 'Iniciando entrenamiento...', 'info');
    try {
        const data = await (await fetch('/api/training/start', { method:'POST' })).json();
        if (!data.success) { setStep(msg, data.message, 'error'); btn.disabled = false; return; }
        btn.classList.add('hidden'); stopBtn.classList.remove('hidden');
        document.getElementById('reloadModelBtn').classList.add('hidden');
        setStep(msg, 'Entrenamiento en progreso...', 'info');
        pollTraining();
    } catch(e) { setStep(msg, 'Error de conexión', 'error'); btn.disabled = false; }
}

async function stopTraining() {
    await fetch('/api/training/stop', { method:'POST' });
    setStep(document.getElementById('trainingStatusMsg'), 'Detención solicitada...', 'info');
}

async function reloadModel() {
    const btn = document.getElementById('reloadModelBtn');
    btn.disabled = true; btn.textContent = 'Aplicando...';
    try {
        const data = await (await fetch('/api/training/reload-model', { method:'POST' })).json();
        btn.textContent = data.success ? '✅ Aplicado' : '❌ Error';
        if (data.success) refreshModelInfo();
    } catch(e) { btn.textContent = '❌ Error'; }
    setTimeout(() => { btn.disabled = false; btn.textContent = 'Aplicar modelo'; }, 3000);
}

function pollTraining() {
    if (trainingPollInterval) clearInterval(trainingPollInterval);
    trainingPollInterval = setInterval(async () => {
        try {
            const [sr, lr] = await Promise.all([
                fetch('/api/training/status'), fetch('/api/training/log?lines=60')
            ]);
            const sd = await sr.json(); const ld = await lr.json();
            const s  = sd.status || {};
            const msg = document.getElementById('trainingStatusMsg');

            document.getElementById('trainingProgressFill').style.width = (s.progress || 0) + '%';
            if (s.epoch) setText('epochLabel', `Época ${s.epoch}/${s.total_epochs}`);
            if (s.loss     != null) { setPill('metricLoss',  `Pérdida: ${s.loss}`,    true); }
            if (s.best_loss!= null) { setPill('metricBestLoss',`Mejor: ${s.best_loss}`, false); }
            if (s.step     != null) { setPill('metricStep',  `Paso: ${s.step}`,        false); }
            setStep(msg, s.message || '', s.state === 'error' ? 'error' : 'info');
            if (ld.log) appendLog(ld.log);

            if (!s.is_running || ['done','stopped','error'].includes(s.state)) {
                clearInterval(trainingPollInterval);
                document.getElementById('startTrainingBtn').classList.remove('hidden');
                document.getElementById('startTrainingBtn').disabled = false;
                document.getElementById('stopTrainingBtn').classList.add('hidden');
                if (s.state === 'done') {
                    setStep(msg, '✅ ' + s.message, 'success');
                    document.getElementById('reloadModelBtn').classList.remove('hidden');
                } else if (s.state === 'error') { setStep(msg, '❌ ' + s.message, 'error'); }
                refreshModelInfo();
            }
        } catch(e) {}
    }, 4000);
}

function refreshTrainingStatus() {
    fetch('/api/training/status').then(r => r.json()).then(d => {
        const s = d.status || {};
        if (s.is_running) {
            document.getElementById('startTrainingBtn').classList.add('hidden');
            document.getElementById('stopTrainingBtn').classList.remove('hidden');
            pollTraining();
        }
        if (s.state === 'done') document.getElementById('reloadModelBtn').classList.remove('hidden');
    }).catch(() => {});
}

function appendLog(lines) {
    const el = document.getElementById('trainingLog'); if (!el) return;
    lines.forEach(line => {
        if (trainingLogLines.includes(line)) return;
        trainingLogLines.push(line);
        const div = document.createElement('div');
        div.className = line.includes('Error')||line.includes('❌') ? 'log-line-error' :
                        line.includes('✅')||line.includes('completad') ? 'log-line-success' :
                        line.includes('poca') || line.includes('poch') ? 'log-line-epoch' : 'log-line-info';
        div.textContent = line; el.appendChild(div);
    });
    el.scrollTop = el.scrollHeight;
}

// ── CONTINUOUS LEARNING ───────────────────────────────────────

async function rebuildDatasetSilently() {
    try {
        const data = await (await fetch('/api/training/rebuild-dataset', { method:'POST' })).json();
        if (data.success) {
            setText('trainingPairsCount', data.total_pairs);
            setText('infoDictEntries', data.dict_entries);
        }
        // Show badge on nav item
        const badge = document.getElementById('datasetBadge');
        if (badge) { badge.style.display = 'inline'; setTimeout(() => badge.style.display = 'none', 8000); }
    } catch(e) {}
}

// ── UTILITIES ─────────────────────────────────────────────────

function showError(msg) {
    const el = document.getElementById('errorMessage');
    if (el) { el.textContent = msg; el.classList.remove('hidden'); }
}
function clearError() {
    const el = document.getElementById('errorMessage');
    if (el) { el.textContent = ''; el.classList.add('hidden'); }
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setStep(el, text, type) {
    if (!el) return; el.textContent = text;
    el.className = 'step-status ' + (type || '');
}
function setPill(id, text, active) {
    const el = document.getElementById(id); if (!el) return;
    el.textContent = text;
    el.classList.toggle('active', active);
}

// ── INIT ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadLanguages();
    // Theme
    const saved = localStorage.getItem('theme') || 'dark';
    applyTheme(saved);
    document.getElementById('themeToggle')?.addEventListener('click', () => {
        const cur = document.documentElement.classList.contains('light-mode') ? 'dark' : 'light';
        applyTheme(cur); localStorage.setItem('theme', cur);
    });

    // Navigation
    document.querySelectorAll('.nav-item[data-section]').forEach(btn => {
        btn.addEventListener('click', () => showSection(btn.dataset.section));
    });

    // Mobile menu
    document.getElementById('mobileMenuBtn')?.addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });

    // Show default section
    showSection('textInputSection');

    // Init speech recognition early
    initSpeech();

    // Translate
    document.getElementById('translateText')?.addEventListener('click', doTranslate);

    // Clear input
    document.getElementById('clearInputBtn')?.addEventListener('click', () => {
        document.getElementById('inputText').value = '';
        document.getElementById('translatedText').value = '';
        document.getElementById('charCount').textContent = '0';
        document.getElementById('methodBadge')?.classList.add('hidden');
        document.getElementById('copyTranslationBtn')?.classList.add('hidden');
        document.getElementById('feedbackSection')?.classList.add('hidden');
        clearError();
    });

    // Char count
    document.getElementById('inputText')?.addEventListener('input', e => {
        document.getElementById('charCount').textContent = e.target.value.length;
    });

    // Copy
    document.getElementById('copyTranslationBtn')?.addEventListener('click', async () => {
        const text = document.getElementById('translatedText').value;
        if (text) { await navigator.clipboard.writeText(text); }
    });

    // Swap langs
    document.getElementById('swapLangsBtn')?.addEventListener('click', () => {
        const src = document.getElementById('sourceLanguage');
        const tgt = document.getElementById('targetLanguage');
        const tmp = src.value; src.value = tgt.value; tgt.value = tmp;
    });

    // Voice
    document.getElementById('startRecording')?.addEventListener('click', startRecording);
    document.getElementById('stopRecording')?.addEventListener('click', stopRecording);

    // Add word
    document.getElementById('addWordBtn')?.addEventListener('click', addNewWord);

    // Feedback
    document.querySelector('.feedback-form')?.addEventListener('submit', e => {
        e.preventDefault(); submitFeedback();
    });

    // Media
    document.getElementById('uploadImageBtn')?.addEventListener('click', () => uploadMedia('image'));
    document.getElementById('uploadAudioBtn')?.addEventListener('click', () => uploadMedia('audio'));
    document.getElementById('useTextAsTagBtn')?.addEventListener('click', () => {
        const tag = document.getElementById('inputText')?.value.trim() || '';
        document.getElementById('mediaTag').value = tag;
        refreshMedia('image', tag); refreshMedia('audio', tag);
    });
    document.getElementById('mediaTag')?.addEventListener('input', e => {
        const t = e.target.value.trim();
        refreshMedia('image', t); refreshMedia('audio', t);
    });

    // Training
    document.getElementById('downloadModelBtn')?.addEventListener('click', startModelDownload);
    document.getElementById('startTrainingBtn')?.addEventListener('click', startTraining);
    document.getElementById('stopTrainingBtn')?.addEventListener('click', stopTraining);
    document.getElementById('reloadModelBtn')?.addEventListener('click', reloadModel);
    document.getElementById('clearLogBtn')?.addEventListener('click', () => {
        trainingLogLines = [];
        const el = document.getElementById('trainingLog');
        if (el) el.innerHTML = '';
    });
});