/* ═══════════════════════════════════════════════
   Blue Archive RVC — 前端应用逻辑
   ═══════════════════════════════════════════════ */

var gsvSpeed = 1.0;
var gsvTemp = 1.0;

const state = {
    categories: [],
    characters: [],
    currentChar: null,
    inputMode: 'upload',
    audioPath: null,
    gsvAudioPath: null,
    params: {
        f0: 0, index: 0.7, filter: 3, rms: 1.0,
        protect: 0.5, resample: 0, f0method: 'rmvpe',
    },
    isConverting: false,
    dereverbOverlap: 4,
};

function navigateTo(page) {
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.querySelectorAll('.nav-link').forEach(function(l) { l.classList.remove('active'); });
    document.getElementById('page-' + page).classList.add('active');
    var link = document.querySelector('.nav-link[data-page="' + page + '"]');
    if (link) link.classList.add('active');
    window.location.hash = '#' + page;
}
window.addEventListener('hashchange', function() {
    var page = window.location.hash.replace('#', '') || 'home';
    if (document.getElementById('page-' + page)) navigateTo(page);
});

document.addEventListener('DOMContentLoaded', function() {
    var page = window.location.hash.replace('#', '') || 'home';
    if (document.getElementById('page-' + page)) navigateTo(page); else navigateTo('home');
    loadCharacters();
    loadTTSVoices();
    loadGSVModels();
});

async function loadCharacters() {
    var resp = await fetch('/api/models');
    var data = await resp.json();
    state.categories = data.categories;
    state.characters = [];
    for (var ci = 0; ci < data.categories.length; ci++) {
        var cat = data.categories[ci];
        for (var chi = 0; chi < cat.characters.length; chi++) {
            var ch = cat.characters[chi];
            ch.categoryTitle = cat.title;
            state.characters.push(ch);
        }
    }
    renderCharacterGrid();
    populateCharSelect();
}

function renderCharacterGrid() {
    var grid = document.getElementById('character-grid');
    grid.innerHTML = '';
    // Group by category
    for (var ci = 0; ci < state.categories.length; ci++) {
        var cat = state.categories[ci];
        var chars = cat.characters || [];
        if (chars.length === 0) continue;
        var section = document.createElement('div');
        section.className = 'category-section';
        section.style.cssText = 'margin-bottom:28px';
        var header = document.createElement('div');
        header.className = 'category-header';
        header.style.cssText = 'margin-bottom:12px';
        var h3 = document.createElement('h3');
        h3.style.cssText = 'font-size:18px;font-weight:600;color:#222;margin:0';
        h3.textContent = cat.title;
        header.appendChild(h3);
        if (cat.description) {
            var p = document.createElement('p');
            p.style.cssText = 'margin:2px 0 0;font-size:13px;color:#888';
            p.textContent = cat.description;
            header.appendChild(p);
        }
        section.appendChild(header);
        var gridWrap = document.createElement('div');
        gridWrap.className = 'character-grid';
        for (var i = 0; i < chars.length; i++) {
            var ch = chars[i];
            var card = document.createElement('div');
            card.className = 'character-card';
            card.onclick = (function(n) { return function() { selectCharacter(n); navigateTo('studio'); }; })(ch.name);
            var imgHtml = ch.cover
                ? '<img class="char-card-img" src="/' + ch.cover + '" alt="' + ch.name + '" loading="lazy" />'
                : '<div class="char-card-img-placeholder">' + ch.name.charAt(0) + '</div>';
            var authorHtml = ch.author ? '<p class="char-meta">' + ch.author + '</p>' : '';
            card.innerHTML = imgHtml + '<div class="char-card-body"><h4>' + ch.name + '</h4>' + authorHtml + '<span class="char-version">RVC ' + ch.version + '</span></div>';
            gridWrap.appendChild(card);
        }
        section.appendChild(gridWrap);
        grid.appendChild(section);
    }
}


// ─── RVC 搜索 ───
var _searchTimer = null;
function onSearchChange() {
    clearTimeout(_searchTimer);
    var q = document.getElementById('search-input').value.trim().toLowerCase();
    _searchTimer = setTimeout(function() {
        var grid = document.getElementById('character-grid');
        if (!grid) return;
        var sections = grid.querySelectorAll('.category-section');
        for (var si = 0; si < sections.length; si++) {
            var cards = sections[si].querySelectorAll('.character-card');
            var visibleCount = 0;
            for (var ci = 0; ci < cards.length; ci++) {
                var name = cards[ci].querySelector('h4');
                if (name) {
                    var match = name.textContent.toLowerCase().indexOf(q) > -1;
                    cards[ci].style.display = match || !q ? '' : 'none';
                    if (match || !q) visibleCount++;
                }
            }
            sections[si].style.display = visibleCount > 0 ? '' : 'none';
        }
    }, 200);
}

function populateCharSelect() {
    var select = document.getElementById('studio-char-select');
    select.innerHTML = '<option value="">— 选择角色 —</option>';
    for (var ci = 0; ci < state.categories.length; ci++) {
        var cat = state.categories[ci];
        if (cat.characters.length === 0) continue;
        var group = document.createElement('optgroup');
        group.label = cat.title;
        for (var chi = 0; chi < cat.characters.length; chi++) {
            var opt = document.createElement('option');
            opt.value = cat.characters[chi].name;
            opt.textContent = cat.characters[chi].name;
            group.appendChild(opt);
        }
        select.appendChild(group);
    }
}

function selectCharacter(name) {
    var ch = null;
    for (var i = 0; i < state.characters.length; i++) {
        if (state.characters[i].name === name) { ch = state.characters[i]; break; }
    }
    if (!ch) return;
    state.currentChar = ch;
    var cover = document.getElementById('studio-cover');
    cover.innerHTML = ch.cover ? '<img src="/' + ch.cover + '" alt="' + ch.name + '" />' : '<div class="char-card-placeholder">' + ch.name + '</div>';
    document.getElementById('studio-char-name').textContent = ch.name;
    document.getElementById('studio-char-info').textContent = 'RVC ' + ch.version + ' | ' + ch.categoryTitle + (ch.author ? ' | ' + ch.author : '');
    document.getElementById('studio-char-select').value = ch.name;
    checkConvertReady();
}

function onStudioCharChange() {
    var name = document.getElementById('studio-char-select').value;
    if (name) { selectCharacter(name); }
    else {
        state.currentChar = null;
        document.getElementById('studio-cover').innerHTML = '<div class="char-card-placeholder">选择角色</div>';
        document.getElementById('studio-char-name').textContent = '—';
        document.getElementById('studio-char-info').textContent = '';
        checkConvertReady();
    }
}

function setInputMode(mode) {
    state.inputMode = mode;
    document.querySelectorAll('.input-tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelector('.input-tab[data-mode="' + mode + '"]').classList.add('active');
    document.querySelectorAll('.input-panel').forEach(function(p) { p.classList.remove('active'); });
    document.getElementById('input-' + mode).classList.add('active');
    // If switching to gsv mode but no gsv audio, switch back to upload
    if (mode === 'gsv' && !state.gsvAudioPath) {
        document.getElementById('gsv-input-tab').style.display = 'none';
        setInputMode('upload');
        return;
    }
    checkConvertReady();
}

function onFileSelected(e) {
    var file = e.target.files[0];
    if (!file) return;
    state.gsvAudioPath = null;
    document.getElementById('file-name').textContent = '📁 ' + file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    document.getElementById('file-info').style.display = 'flex';
    uploadFile(file);
}

async function uploadFile(file) {
    var fd = new FormData();
    fd.append('file', file);
    var resp = await fetch('/api/upload', { method: 'POST', body: fd });
    var data = await resp.json();
    state.audioPath = data.path;
    checkConvertReady();
}

function clearFile() {
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').style.display = 'none';
    state.audioPath = null;
    state.gsvAudioPath = null;
    checkConvertReady();
}

function clearGSVAudio() {
    state.gsvAudioPath = null;
    state.audioPath = null;
    document.getElementById('gsv-input-tab').style.display = 'none';
    if (state.inputMode === 'gsv') setInputMode('upload');
    checkConvertReady();
}

async function loadTTSVoices() {
    try {
        var resp = await fetch('/api/voices');
        var data = await resp.json();
        var select = document.getElementById('tts-voice');
        select.innerHTML = '';
        var groups = {};
        for (var i = 0; i < data.voices.length; i++) {
            var v = data.voices[i];
            var loc = v.locale || 'other';
            if (!groups[loc]) groups[loc] = [];
            groups[loc].push(v);
        }
        var keys = Object.keys(groups);
        for (var ki = 0; ki < keys.length; ki++) {
            var g = document.createElement('optgroup');
            g.label = keys[ki];
            var voices = groups[keys[ki]];
            for (var vi = 0; vi < voices.length; vi++) {
                var opt = document.createElement('option');
                opt.value = voices[vi].name;
                opt.textContent = voices[vi].short_name + ' (' + voices[vi].gender + ')';
                g.appendChild(opt);
            }
            select.appendChild(g);
        }
    } catch (e) { console.error('TTS failed', e); }
}

async function generateTTS() {
    var text = document.getElementById('tts-text').value.trim();
    if (!text) { showToast('请输入文本'); return; }
    var voice = document.getElementById('tts-voice').value;
    var btn = document.querySelector('#input-tts .btn-secondary');
    btn.disabled = true;
    btn.textContent = '生成中...';
    try {
        var fd = new FormData();
        fd.append('text', text);
        fd.append('voice', voice);
        var resp = await fetch('/api/tts', { method: 'POST', body: fd });
        var data = await resp.json();
        state.audioPath = data.path;
        showToast('语音生成成功');
        checkConvertReady();
    } catch (e) { showToast('语音生成失败'); }
    finally {
        btn.disabled = false;
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 生成语音';
    }
}

function updateParam(name, value) {
    state.params[name] = value;
    var map = { f0: 'val-f0', index: 'val-index', filter: 'val-filter', rms: 'val-rms', protect: 'val-protect', resample: 'val-resample' };
    var el = document.getElementById(map[name]);
    if (el) el.textContent = value;
}

function checkConvertReady() {
    document.getElementById('btn-convert').disabled = !(state.currentChar && state.audioPath);
}

// ─── 通用队列轮询 ───
async function pollTask(queueId, opts) {
    var statusEl = opts.statusEl;
    var statusText = opts.statusText;
    var progressWrap = opts.progressWrap;
    var progressBar = opts.progressBar;

    statusEl.style.display = 'flex';
    statusText.innerHTML = '⏳ 正在加入队列...';

    while (true) {
        await new Promise(function(r) { setTimeout(r, 1500); });
        var qResp = await fetch('/api/queue/' + queueId);
        var qData = await qResp.json();

        if (qData.status === 'done' || qData.output_path) {
            statusEl.style.display = 'none';
            if (progressWrap) progressWrap.style.display = 'none';
            opts.onDone(qData);
            return;
        }
        if (qData.status === 'error' || qData.error) {
            statusEl.style.display = 'none';
            if (progressWrap) progressWrap.style.display = 'none';
            throw new Error(qData.error || '处理失败');
        }
        if (qData.status === 'processing') {
            if (progressWrap) progressWrap.style.display = 'block';
            var pct = qData.progress || 0;
            if (progressBar) progressBar.style.width = pct + '%';
            var msg = qData.message || '处理中...';
            statusText.innerHTML = '🔄 ' + msg + '<br><small>' + pct + '%</small>';
        } else if (qData.status === 'queued') {
            var pos = qData.position;
            var est = qData.estimated;
            var timeStr = '即将开始';
            if (est > 0) {
                timeStr = est < 60 ? '约 ' + Math.round(est) + ' 秒' : '约 ' + Math.round(est/60) + ' 分 ' + Math.round(est%60) + ' 秒';
            }
            statusText.innerHTML = '⏳ 队列位置: 第 ' + pos + ' 位<br><small>预计等待: ' + timeStr + '</small>';
            if (progressWrap) progressWrap.style.display = 'none';
        }
    }
}

// ─── 翻唱转换 ───
async function startConvert() {
    if (state.isConverting) return;
    if (!state.currentChar || !state.audioPath) return;

    state.isConverting = true;
    var btn = document.getElementById('btn-convert');
    var outputArea = document.getElementById('output-area');
    outputArea.style.display = 'none';
    btn.disabled = true;
    btn.innerHTML = '<div class="status-spinner" style="width:16px;height:16px;border-width:2px"></div> 队列中';

    try {
        var fd = new FormData();
        fd.append('character', state.currentChar.name);
        fd.append('audio_path', state.audioPath);
        fd.append('f0_up_key', state.params.f0);
        fd.append('f0_method', state.params.f0method);
        fd.append('index_rate', state.params.index);
        fd.append('filter_radius', state.params.filter);
        fd.append('resample_sr', state.params.resample);
        fd.append('rms_mix_rate', state.params.rms);
        fd.append('protect', state.params.protect);

        var resp = await fetch('/api/convert', { method: 'POST', body: fd });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || '请求失败');

        await pollTask(data.queue_id, {
            statusEl: document.getElementById('convert-status'),
            statusText: document.getElementById('status-text'),
            progressWrap: document.getElementById('convert-progress-wrap'),
            progressBar: document.getElementById('convert-progress-bar'),
            onDone: function(qd) {
                outputArea.style.display = 'block';
                document.getElementById('output-info').textContent = qd.info || '';
                var fn = qd.output_path.split('/').pop();
                document.getElementById('output-audio').src = '/api/download/' + fn;
                document.getElementById('download-link').href = '/api/download/' + fn;
            }
        });
    } catch (e) {
        showToast('转换失败: ' + e.message);
        console.error(e);
    } finally {
        state.isConverting = false;
        btn.disabled = false;
        btn.innerHTML = '\n            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>\n            开始转换\n        ';
    }
}

// ─── UVR5 ───
function onUVR5FileSelected(e) {
    var file = e.target.files[0];
    if (!file) return;
    document.getElementById('uvr5-file-name').textContent = '📁 ' + file.name;
    document.getElementById('uvr5-file-info').style.display = 'flex';
    var fd = new FormData();
    fd.append('file', file);
    fetch('/api/upload', { method: 'POST', body: fd })
        .then(function(r) { return r.json(); })
        .then(function(d) { window._uvr5Path = d.path; document.getElementById('btn-uvr5').disabled = false; })
        .catch(function(e) { console.error(e); });
}

function clearUVR5File() {
    document.getElementById('uvr5-file-input').value = '';
    document.getElementById('uvr5-file-info').style.display = 'none';
    window._uvr5Path = null;
    document.getElementById('btn-uvr5').disabled = true;
}

async function startUVR5() {
    if (!window._uvr5Path) return;
    var btn = document.getElementById('btn-uvr5');
    btn.disabled = true;
    btn.innerHTML = '<div class="status-spinner" style="width:16px;height:16px;border-width:2px"></div> 提交中...';
    try {
        var fd = new FormData();
        fd.append('audio_path', window._uvr5Path);
        fd.append('model_name', document.getElementById('uvr5-model').value);
        var resp = await fetch('/api/uvr5/separate', { method: 'POST', body: fd });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail);
        await pollTask(data.queue_id, {
            statusEl: document.getElementById('uvr5-status-area'),
            statusText: document.getElementById('uvr5-status-text'),
            progressWrap: document.getElementById('uvr5-progress-wrap'),
            progressBar: document.getElementById('uvr5-progress-bar'),
            onDone: function(qd) {
                document.getElementById('uvr5-results').style.display = 'block';
                document.getElementById('uvr5-placeholder').style.display = 'none';
                document.getElementById('uvr5-vocals').src = '/api/download/' + qd.vocals.split('/').pop();
                document.getElementById('uvr5-instrumental').src = '/api/download/' + qd.instrumental.split('/').pop();
                document.getElementById('uvr5-status').textContent = qd.status || '完成';
            }
        });
    } catch (e) { showToast('分离失败: ' + e.message); console.error(e); }
    finally {
        btn.disabled = false;
        btn.innerHTML = '\n            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>\n            开始分离\n        ';
    }
}

function showToast(msg) {
    var existing = document.querySelector('.toast-notice');
    if (existing) existing.remove();
    var toast = document.createElement('div');
    toast.className = 'toast-notice';
    toast.textContent = msg;
    toast.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#2C3E50;color:white;padding:12px 24px;border-radius:12px;font-size:14px;font-weight:500;z-index:999;box-shadow:0 4px 16px rgba(0,0,0,0.2);font-family:"Noto Sans SC",sans-serif';
    document.body.appendChild(toast);
    setTimeout(function() { toast.remove(); }, 3000);
}

function setUVR5Tab(tab) {
    document.querySelectorAll('[data-uvr5tab]').forEach(function(t) { t.classList.remove('active'); });
    document.querySelector('[data-uvr5tab="' + tab + '"]').classList.add('active');
    document.querySelectorAll('.uvr5-tab-content').forEach(function(c) { c.classList.remove('active'); });
    document.getElementById('uvr5-' + tab).classList.add('active');
}

function onDeReverbFileSelected(e) {
    var file = e.target.files[0];
    if (!file) return;
    document.getElementById('dereverb-file-name').textContent = '📁 ' + file.name;
    document.getElementById('dereverb-file-info').style.display = 'flex';
    var fd = new FormData();
    fd.append('file', file);
    fetch('/api/upload', { method: 'POST', body: fd })
        .then(function(r) { return r.json(); })
        .then(function(d) { window._dereverbPath = d.path; document.getElementById('btn-dereverb').disabled = false; })
        .catch(function(e) { console.error(e); });
}

function clearDeReverbFile() {
    document.getElementById('dereverb-file-input').value = '';
    document.getElementById('dereverb-file-info').style.display = 'none';
    window._dereverbPath = null;
    document.getElementById('btn-dereverb').disabled = true;
}

async function startDeReverb() {
    if (!window._dereverbPath) return;
    var btn = document.getElementById('btn-dereverb');
    btn.disabled = true;
    btn.innerHTML = '<div class="status-spinner" style="width:16px;height:16px;border-width:2px"></div> 提交中...';
    try {
        var fd = new FormData();
        fd.append('audio_path', window._dereverbPath);
        fd.append('overlap', state.dereverbOverlap || 4);
        var resp = await fetch('/api/uvr5/dereverb', { method: 'POST', body: fd });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.detail);
        await pollTask(data.queue_id, {
            statusEl: document.getElementById('dereverb-status-area'),
            statusText: document.getElementById('dereverb-status-text'),
            progressWrap: document.getElementById('dereverb-progress-wrap'),
            progressBar: document.getElementById('dereverb-progress-bar'),
            onDone: function(qd) {
                document.getElementById('dereverb-results').style.display = 'block';
                document.getElementById('dereverb-placeholder').style.display = 'none';
                document.getElementById('dereverb-dry').src = '/api/download/' + qd.dry.split('/').pop();
                document.getElementById('dereverb-reverb').src = '/api/download/' + qd.reverb.split('/').pop();
                document.getElementById('dereverb-status').textContent = qd.status || '完成';
            }
        });
    } catch (e) { showToast('去混响失败: ' + e.message); console.error(e); }
    finally {
        btn.disabled = false;
        btn.innerHTML = '\n            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>\n            消除混响\n        ';
    }
}

// ─── GSV TTS ───
var _gsvLastAudioUrl = null;
var _gsvModels = [];
var _gsvOnline = false;
var _gsvCurrentChar = null;


// ─── GSV 搜索 ───
var _gsvSearchTimer = null;
function onGSVSearchChange() {
    clearTimeout(_gsvSearchTimer);
    _gsvSearchTimer = setTimeout(function() {
        var q = document.getElementById('gsv-search-input').value.trim().toLowerCase();
        var grid = document.getElementById('gsv-character-grid');
        if (!grid) return;
        var cards = grid.querySelectorAll('.character-card');
        for (var i = 0; i < cards.length; i++) {
            var name = cards[i].querySelector('h4');
            if (name) {
                var match = name.textContent.toLowerCase().indexOf(q) > -1;
                cards[i].style.display = match || !q ? '' : 'none';
            }
        }
    }, 200);
}

async function loadGSVModels() {
    var statusEl = document.getElementById('gsv-connection-status');
    try {
        var resp = await fetch('/api/gsv/models');
        var data = await resp.json();
        _gsvModels = data.models || [];
        renderGSVGrid();
    } catch(e) {
        if (statusEl) statusEl.innerHTML = '❌ 模型加载失败';
        return;
    }
    try {
        var checkResp = await fetch('/api/gsv/check');
        var checkData = await checkResp.json();
        _gsvOnline = checkData.ok;
        if (checkData.ok) {
            if (statusEl) statusEl.innerHTML = '✅ GSV 服务在线<br><small>' + _gsvModels.length + ' 个模型可用</small>';
        } else {
            _gsvOnline = false;
            if (statusEl) statusEl.innerHTML = '⚠️ GSV 服务不可达';
        }
    } catch(e) {
        _gsvOnline = false;
        if (statusEl) statusEl.innerHTML = '⚠️ GSV 服务不可达';
    }
}

function renderGSVGrid() {
    var grid = document.getElementById('gsv-character-grid');
    if (!grid) return;
    grid.innerHTML = '';
    if (!_gsvModels || _gsvModels.length === 0) {
        var msg = document.createElement('div');
        msg.style.cssText = 'text-align:center;padding:60px;color:#999';
        msg.innerHTML = '暂无可用 GSV 模型<br><small>请先在 <a href="/admin" style="color:#4A90D9">后台</a> 配置</small>';
        grid.appendChild(msg);
        return;
    }
    // Group by section
    var groups = {};
    for (var i = 0; i < _gsvModels.length; i++) {
        var m = _gsvModels[i];
        var sec = m.section || '未分类';
        if (!groups[sec]) groups[sec] = [];
        groups[sec].push(m);
    }
    var sectionOrder = Object.keys(groups).sort(function(a, b) {
        if (a === '未分类') return 1;
        if (b === '未分类') return -1;
        return a.localeCompare(b);
    });

    for (var si = 0; si < sectionOrder.length; si++) {
        var sec = sectionOrder[si];
        var models = groups[sec];

        var section = document.createElement('div');
        section.className = 'category-section';
        section.style.cssText = 'margin-bottom:28px';

        var header = document.createElement('div');
        header.className = 'category-header';
        header.style.cssText = 'margin-bottom:12px';
        var h3 = document.createElement('h3');
        h3.style.cssText = 'font-size:16px;font-weight:600;color:#222;margin:0';
        h3.textContent = sec;
        header.appendChild(h3);
        section.appendChild(header);

        var gridWrap = document.createElement('div');
        gridWrap.className = 'character-grid';

        for (var mi = 0; mi < models.length; mi++) {
            var ch = models[mi];
            var card = document.createElement('div');
            card.className = 'character-card';
            card.onclick = (function(name) { return function() { selectGSVCharacter(name); }; })(ch.name);

            var coverUrl = '';
            if (ch.cover) {
                var cv = ch.cover;
                if (cv.indexOf('gsv_covers') > -1 || cv.indexOf('temp/gsv_covers') > -1) {
                    var parts = cv.split('/');
                    coverUrl = '/api/download/gsv_covers/' + parts[parts.length - 1];
                } else {
                    coverUrl = cv;
                }
            }

            var img = document.createElement('img');
            img.className = 'char-card-img';
            img.alt = ch.name;
            img.loading = 'lazy';
            if (coverUrl) {
                img.src = coverUrl;
            } else {
                img.style.display = 'none';
            }
            img.onerror = function() {
                this.style.display = 'none';
            };

            var body = document.createElement('div');
            body.className = 'char-card-body';
            var nameEl = document.createElement('h4');
            nameEl.textContent = ch.name;
            var tag = document.createElement('span');
            tag.className = 'char-version';
            tag.textContent = 'GSV';
            body.appendChild(nameEl);
            body.appendChild(tag);

            card.appendChild(img);
            card.appendChild(body);
            gridWrap.appendChild(card);
        }

        section.appendChild(gridWrap);
        grid.appendChild(section);
    }
}

function selectGSVCharacter(name) {
    _gsvCurrentChar = name;
    document.getElementById('gsv-studio-panel').style.display = 'block';
    document.getElementById('gsv-studio-panel').scrollIntoView({ behavior: 'smooth' });
    var coverEl = document.getElementById('gsv-cover');
    var display = document.getElementById('gsv-model-display');
    var desc = document.getElementById('gsv-model-desc');
    display.textContent = name;
    desc.textContent = '已选择';
    var coverUrl = '';
    for (var i = 0; i < _gsvModels.length; i++) {
        if (_gsvModels[i].name === name && _gsvModels[i].cover) {
            var cv = _gsvModels[i].cover;
            if (cv.indexOf('gsv_covers') > -1 || cv.indexOf('temp/gsv_covers') > -1) {
                var parts = cv.split('/');
                coverUrl = '/api/download/gsv_covers/' + parts[parts.length - 1];
            } else {
                coverUrl = cv;
            }
            break;
        }
    }
    if (coverUrl) {
        coverEl.innerHTML = '';
        var img = document.createElement('img');
        img.src = coverUrl;
        img.alt = name;
        img.style.cssText = 'width:100%;height:100%;object-fit:contain;padding:8px';
        img.onerror = function() { coverEl.innerHTML = '<div class="char-card-placeholder" style="font-size:48px">🎤</div>'; };
        coverEl.appendChild(img);
    } else {
        coverEl.innerHTML = '<div class="char-card-placeholder" style="font-size:48px">🎤</div>';
    }
    var btn = document.getElementById('btn-gsv-generate');
    if (btn) btn.disabled = !_gsvOnline;
}

function deselectGSV() {
    _gsvCurrentChar = null;
    document.getElementById('gsv-studio-panel').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function generateGSVTTS() {
    var text = document.getElementById('gsv-text').value.trim();
    if (!text) { showToast('请输入文本'); return; }
    if (text.length > 500) { showToast('文本过长（最多 500 字符）'); return; }
    if (!_gsvCurrentChar) { showToast('请选择 GSV 角色'); return; }
    var statusEl = document.getElementById('gsv-convert-status');
    var statusText = document.getElementById('gsv-status-text');
    var outputArea = document.getElementById('gsv-output-area');
    var noOutput = document.getElementById('gsv-no-output');
    if (!_gsvOnline) { showToast('GSV 服务不可用'); return; }
    statusEl.style.display = 'flex';
    statusText.innerHTML = '⏳ 正在请求 GSV TTS...';
    outputArea.style.display = 'none';
    if (noOutput) noOutput.style.display = 'none';
    try {
        var fd = new FormData();
        fd.append('text', text);
        fd.append('model_name', _gsvCurrentChar);
        fd.append('speed_factor', gsvSpeed || 1.0);
        fd.append('temperature', gsvTemp || 1.0);
        fd.append('top_k', (document.getElementById('gsv-topk') ? document.getElementById('gsv-topk').value : 5));
        var resp = await fetch('/api/gsv/tts', { method: 'POST', body: fd });
        if (!resp.ok) { var ed = await resp.json().catch(function(){}); throw new Error(ed && ed.detail || 'GSV 失败'); }
        var blob = await resp.blob();
        var url = URL.createObjectURL(blob);
        _gsvLastAudioUrl = url;
        window._gsvLastBlob = blob;
        statusEl.style.display = 'none';
        outputArea.style.display = 'block';
        document.getElementById('gsv-output-audio').src = url;
        document.getElementById('gsv-output-info').textContent = '已生成 ' + (blob.size / 1024).toFixed(0) + ' KB';
        showToast('GSV 语音生成成功');
    } catch (e) {
        // Don't mark GSV offline if it's a client-side validation error
        if (e.message.indexOf('文本过长') === -1) {
            _gsvOnline = false;
        }
        statusEl.style.display = 'none';
        if (noOutput) noOutput.style.display = 'block';
        showToast('GSV 失败: ' + e.message);
    }
}

async function sendToStudio() {
    if (!window._gsvLastBlob) { showToast('请先生成语音'); return; }
    var fd = new FormData();
    fd.append('file', new File([window._gsvLastBlob], 'gsv.wav', { type: 'audio/wav' }));
    try {
        var r = await fetch('/api/upload', { method: 'POST', body: fd });
        var d = await r.json();
        state.gsvAudioPath = d.path;
        state.audioPath = d.path;
        checkConvertReady();
        showToast('已发送到变声工作室（GSV 音频）');
        navigateTo('studio');
        // After navigation, show and select GSV tab
        setTimeout(function() {
            var gsvTab = document.getElementById('gsv-input-tab');
            if (gsvTab) gsvTab.style.display = '';
            setInputMode('gsv');
        }, 100);
    } catch (e) { showToast('发送失败: ' + e.message); }
}
