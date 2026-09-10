/**
 * VoxEmotion AI - Speech Emotion Recognition Frontend Application Controller
 * Architecture: Vercel Frontend CDN <-> Render ML Backend <-> Supabase Cloud Database
 */

// Default Production Render Backend
const DEFAULT_RENDER_URL = "https://emotion-recognition-speech.onrender.com";

// Global App State
const AppState = {
    apiBaseUrl: localStorage.getItem("vox_api_url") || (window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1") ? window.location.origin : (window.VOX_API_URL || DEFAULT_RENDER_URL)),
    currentPredictionId: null,
    currentEmotion: null,
    backendOnline: false,
    supabaseOnline: false,
    selectedRating: 5
};

// Chart References
let circumplexChart = null;
let emotionProbsChart = null;
let featImpChart = null;

document.addEventListener("DOMContentLoaded", () => {
    initBackendConfig();
    initModeTabs();
    initMicrophoneRecorder();
    initAudioUpload();
    initBatchProcessing();
    initApiTabs();
    initModals();
    
    // Test backend connection & load initial data
    testBackendConnection(false).then(() => {
        loadBenchmarks();
        loadSupabaseHistory();
        loadSupabaseAnalytics();
        loadPresetSample("happy");
    });
});

// ============================================================================
// BACKEND URL & HEALTH MANAGEMENT
// ============================================================================
function getApiUrl(endpoint) {
    let base = AppState.apiBaseUrl.replace(/\/+$/, "");
    if (!endpoint.startsWith("/")) endpoint = "/" + endpoint;
    return `${base}${endpoint}`;
}

async function testBackendConnection(showToast = true) {
    const statusPill = document.getElementById("backend-status-pill");
    const statusText = document.getElementById("backend-status-text");
    const dbStatusPill = document.getElementById("db-status-pill");
    const testResultMsg = document.getElementById("test-result-msg");
    const startTime = performance.now();

    if (statusPill) {
        statusPill.className = "status-pill warning";
        statusPill.querySelector(".pulse-dot").className = "pulse-dot warning";
        if (statusText) statusText.textContent = "Connecting to Backend...";
    }

    try {
        const response = await fetch(getApiUrl("/api/health"), { cache: "no-store" });
        const latency = Math.round(performance.now() - startTime);

        if (response.ok) {
            const data = await response.json();
            AppState.backendOnline = true;
            AppState.supabaseOnline = data.database && data.database.connected;

            if (statusPill) {
                statusPill.className = "status-pill";
                statusPill.querySelector(".pulse-dot").className = "pulse-dot";
                if (statusText) statusText.innerHTML = `AI Core: <strong>Online</strong>`;
            }

            if (dbStatusPill) {
                if (AppState.supabaseOnline) {
                    dbStatusPill.className = "status-pill";
                    dbStatusPill.innerHTML = `<span class="pulse-dot"></span><span>Database: <strong>Connected</strong></span>`;
                } else {
                    dbStatusPill.className = "status-pill warning";
                    dbStatusPill.innerHTML = `<span class="pulse-dot warning"></span><span>Database: <strong>Standby</strong></span>`;
                }
            }

            if (testResultMsg) {
                testResultMsg.innerHTML = `<span style="color: #34d399;">✓ Connected (${latency}ms)</span>`;
            }

            updateApiSnippets();
            return true;
        } else {
            throw new Error(`Server returned HTTP ${response.status}`);
        }
    } catch (err) {
        AppState.backendOnline = false;
        AppState.supabaseOnline = false;

        if (statusPill) {
            statusPill.className = "status-pill danger";
            statusPill.querySelector(".pulse-dot").className = "pulse-dot danger";
            if (statusText) statusText.innerHTML = `AI Core: <strong>Offline</strong>`;
        }

        if (testResultMsg) {
            testResultMsg.innerHTML = `<span style="color: #fb7185;">✗ Could not reach backend server.</span>`;
        }
        return false;
    }
}

function initBackendConfig() {
    const inputUrl = document.getElementById("backend-url-input");
    const btnSaveUrl = document.getElementById("btn-save-backend-url");
    const btnTestUrl = document.getElementById("btn-test-backend-url");
    const btnResetUrl = document.getElementById("btn-reset-backend-url");

    if (inputUrl) inputUrl.value = AppState.apiBaseUrl;

    if (btnSaveUrl) {
        btnSaveUrl.addEventListener("click", () => {
            const val = inputUrl.value.trim();
            if (val) {
                AppState.apiBaseUrl = val;
                localStorage.setItem("vox_api_url", val);
                testBackendConnection(true).then(() => {
                    loadBenchmarks();
                    loadSupabaseHistory();
                    closeModal("modal-settings");
                });
            }
        });
    }

    if (btnTestUrl) {
        btnTestUrl.addEventListener("click", () => {
            AppState.apiBaseUrl = inputUrl.value.trim();
            testBackendConnection(true);
        });
    }

    if (btnResetUrl) {
        btnResetUrl.addEventListener("click", () => {
            const defaultUrl = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1") ? window.location.origin : DEFAULT_RENDER_URL;
            inputUrl.value = defaultUrl;
            AppState.apiBaseUrl = defaultUrl;
            localStorage.removeItem("vox_api_url");
            testBackendConnection(true);
        });
    }
}

// ============================================================================
// MODE TABS & NAVIGATION
// ============================================================================
function initModeTabs() {
    const tabBtns = document.querySelectorAll(".mode-tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const mode = btn.dataset.mode;
            document.querySelectorAll(".mode-content").forEach(c => c.classList.add("hidden"));
            const target = document.getElementById(`mode-${mode}`);
            if (target) target.classList.remove("hidden");
        });
    });
}

// ============================================================================
// PRESET SAMPLE LOADER
// ============================================================================
window.loadPresetSample = async function(emotionId) {
    const player = document.getElementById("main-audio-player");
    const playerTitle = document.getElementById("player-title");
    const filename = `${emotionId}_sample.wav`;

    if (playerTitle) playerTitle.textContent = filename;
    if (player) {
        player.src = `/static/samples/${filename}`;
        // Fallback for Vercel static root
        player.onerror = () => { player.src = `samples/${filename}`; };
        player.play().catch(() => {});
    }

    try {
        const res = await fetch(getApiUrl(`/api/sample/${emotionId}`));
        const data = await res.json();
        if (data.status === "success") {
            AppState.currentPredictionId = data.db_id || null;
            renderEmotionResults(data.result);
            loadSupabaseHistory();
        }
    } catch (e) {
        console.error("Failed to load preset sample from API:", e);
    }
};

// ============================================================================
// LIVE MICROPHONE RECORDING & WEBAUDIO WAVEFORM
// ============================================================================
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animFrame = null;
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;

function initMicrophoneRecorder() {
    const recordBtn = document.getElementById("btn-record-toggle");
    if (!recordBtn) return;

    recordBtn.addEventListener("click", () => {
        if (!isRecording) startRecording();
        else stopRecording();
    });
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const player = document.getElementById("main-audio-player");
            const playerTitle = document.getElementById("player-title");
            
            if (player) player.src = URL.createObjectURL(audioBlob);
            if (playerTitle) playerTitle.textContent = "Live_Microphone_Stream.wav";

            sendAudioForPrediction(audioBlob, "Live_Microphone_Stream.wav");
            stream.getTracks().forEach(t => t.stop());
            if (audioContext) audioContext.close();
            cancelAnimationFrame(animFrame);
        };

        mediaRecorder.start();
        isRecording = true;
        recordStartTime = Date.now();

        // UI state
        const recBtn = document.getElementById("btn-record-toggle");
        const recLabel = document.getElementById("rec-label");
        if (recBtn) recBtn.classList.add("recording");
        if (recLabel) recLabel.textContent = "Stop & Analyze";

        timerInterval = setInterval(updateTimer, 100);
        drawWaveform();
    } catch (err) {
        alert("Microphone access denied or unsupported browser.");
        console.error(err);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        clearInterval(timerInterval);

        const recBtn = document.getElementById("btn-record-toggle");
        const recLabel = document.getElementById("rec-label");
        if (recBtn) recBtn.classList.remove("recording");
        if (recLabel) recLabel.textContent = "Start Recording";
    }
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    const timerElem = document.getElementById("mic-timer");
    if (timerElem) timerElem.textContent = `${mins}:${secs}`;
}

function drawWaveform() {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d");
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function render() {
        if (!isRecording) return;
        animFrame = requestAnimationFrame(render);
        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "rgba(8, 7, 19, 0.4)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#06b6d4";
        ctx.beginPath();

        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    render();
}

// ============================================================================
// AUDIO FILE UPLOAD (DRAG & DROP)
// ============================================================================
function initAudioUpload() {
    const dropzone = document.getElementById("audio-dropzone");
    const fileInput = document.getElementById("audio-file-input");
    if (!dropzone || !fileInput) return;

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

function handleFileUpload(file) {
    const player = document.getElementById("main-audio-player");
    const playerTitle = document.getElementById("player-title");

    if (player) player.src = URL.createObjectURL(file);
    if (playerTitle) playerTitle.textContent = file.name;

    sendAudioForPrediction(file, file.name);
}

// ============================================================================
// API INFERENCE DISPATCHER
// ============================================================================
async function sendAudioForPrediction(audioBlobOrFile, filename) {
    const formData = new FormData();
    formData.append("file", audioBlobOrFile, filename);

    try {
        const response = await fetch(getApiUrl("/api/predict"), {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if (data.status === "success") {
            AppState.currentPredictionId = data.prediction_id || null;
            renderEmotionResults(data.result);
            loadSupabaseHistory();
            loadSupabaseAnalytics();
        } else {
            alert(`Prediction failed: ${data.message}`);
        }
    } catch (err) {
        console.error("API error during predict:", err);
        alert("Failed to connect to AI engine. Please verify the server is active.");
    }
}

// ============================================================================
// RESULTS RENDERING & CHARTS
// ============================================================================
function renderEmotionResults(result) {
    if (!result || !result.consensus) return;
    const c = result.consensus;
    AppState.currentEmotion = c.top_emotion;

    // Verdict Elements
    const emojiElem = document.getElementById("verdict-emoji");
    const nameElem = document.getElementById("verdict-emotion");
    const confElem = document.getElementById("consensus-confidence");
    const descElem = document.getElementById("verdict-desc");
    const secElem = document.getElementById("secondary-tag");

    if (emojiElem) emojiElem.textContent = c.emoji;
    if (nameElem) {
        nameElem.textContent = (c.label || c.top_emotion).toUpperCase();
        nameElem.style.color = c.color || "#ffffff";
    }
    if (confElem) confElem.textContent = `${c.confidence}% Confidence`;
    if (descElem) descElem.textContent = c.description || "";
    if (secElem) {
        secElem.innerHTML = `Secondary: <strong>${c.secondary_emotion} (${c.secondary_confidence}%)</strong>`;
    }

    // Biometrics
    const feats = result.acoustic_features || {};
    const pitchElem = document.getElementById("bio-pitch");
    const rmsElem = document.getElementById("bio-rms");
    const centroidElem = document.getElementById("bio-centroid");
    const zcrElem = document.getElementById("bio-zcr");

    if (pitchElem) pitchElem.textContent = feats.pitch_f0_mean ? `${Math.round(feats.pitch_f0_mean)} Hz` : "--";
    if (rmsElem) rmsElem.textContent = feats.rms_energy ? feats.rms_energy.toFixed(3) : "--";
    if (centroidElem) centroidElem.textContent = feats.spectral_centroid ? `${Math.round(feats.spectral_centroid)} Hz` : "--";
    if (zcrElem) zcrElem.textContent = feats.zcr_mean ? feats.zcr_mean.toFixed(3) : "--";

    // 2D Circumplex Plot
    renderCircumplexPlot(c.valence, c.arousal, c.label, c.color);

    // Probability Distribution Chart
    renderEmotionProbsChart(c.probabilities);
}

function renderCircumplexPlot(valence, arousal, label, color) {
    const coordsElem = document.getElementById("affect-coords");
    if (coordsElem) {
        coordsElem.textContent = `Valence: ${valence >= 0 ? '+' : ''}${valence.toFixed(2)} | Arousal: ${arousal >= 0 ? '+' : ''}${arousal.toFixed(2)}`;
    }

    const ctx = document.getElementById("chart-circumplex");
    if (!ctx) return;

    const dataPoints = [
        { x: valence, y: arousal, r: 9 }
    ];

    if (circumplexChart) circumplexChart.destroy();

    circumplexChart = new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: label || 'Current Voice Sample',
                data: dataPoints,
                backgroundColor: color || '#8b5cf6',
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    min: -1.0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    title: { display: true, text: '← Unpleasant (Valence) Pleasant →', color: '#94a3b8', font: { size: 11 } }
                },
                y: {
                    min: -1.0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    title: { display: true, text: '← Subdued (Arousal) Intense →', color: '#94a3b8', font: { size: 11 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => `${label}: Valence ${item.raw.x}, Arousal ${item.raw.y}`
                    }
                }
            }
        }
    });
}

function renderEmotionProbsChart(probDict) {
    const ctx = document.getElementById("chart-emotion-probs");
    if (!ctx || !probDict) return;

    const labels = Object.keys(probDict).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const values = Object.values(probDict);

    const colors = [
        '#94a3b8', '#38bdf8', '#10b981', '#6366f1',
        '#f43f5e', '#a855f7', '#f59e0b', '#ec4899'
    ];

    if (emotionProbsChart) emotionProbsChart.destroy();

    emotionProbsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 10 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// ============================================================================
// SUPABASE LIVE HISTORY & CLOUD ANALYTICS
// ============================================================================
async function loadSupabaseHistory() {
    const tbody = document.getElementById("history-table-body");
    if (!tbody) return;

    try {
        const response = await fetch(getApiUrl("/api/history?limit=15"));
        const data = await response.json();

        if (data.status === "success" && data.history) {
            if (data.history.length === 0) {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">No speech predictions stored yet. Run an analysis above!</td></tr>`;
                return;
            }

            tbody.innerHTML = data.history.map(item => {
                const date = new Date(item.created_at);
                const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const emo = (item.predicted_emotion || "neutral").toLowerCase();
                const conf = parseFloat(item.confidence || 0).toFixed(1);
                const val = parseFloat(item.valence || 0).toFixed(2);
                const aro = parseFloat(item.arousal || 0).toFixed(2);

                return `
                    <tr>
                        <td style="font-family: var(--font-mono); color: var(--text-subtle);">${timeStr}</td>
                        <td style="font-weight: 500;">${item.filename || 'recording.wav'}</td>
                        <td><span class="emotion-pill ${emo}">${emo}</span></td>
                        <td>
                            <div class="progress-track"><div class="progress-fill" style="width: ${conf}%;"></div></div>
                            <span style="font-family: var(--font-mono); font-size: 0.8rem;">${conf}%</span>
                        </td>
                        <td><span class="affect-chip">V: ${val >= 0 ? '+' : ''}${val}</span></td>
                        <td><span class="affect-chip">A: ${aro >= 0 ? '+' : ''}${aro}</span></td>
                        <td>
                            <button class="btn btn-secondary btn-sm" onclick="openFeedbackModal('${item.id}', '${emo}')" style="padding: 0.25rem 0.6rem; font-size: 0.74rem;">
                                ★ Rate
                            </button>
                        </td>
                    </tr>
                `;
            }).join("");
        }
    } catch (err) {
        console.warn("Could not load history from Supabase:", err);
    }
}

async function loadSupabaseAnalytics() {
    try {
        const response = await fetch(getApiUrl("/api/analytics"));
        const data = await response.json();
        if (data.status === "success" && data.analytics) {
            const a = data.analytics;
            const totalElem = document.getElementById("analytics-total");
            const confElem = document.getElementById("analytics-conf");
            const valElem = document.getElementById("analytics-val");

            if (totalElem) totalElem.textContent = a.total_inferences;
            if (confElem) confElem.textContent = `${a.average_confidence}%`;
            if (valElem) valElem.textContent = `${a.average_valence >= 0 ? '+' : ''}${a.average_valence}`;
        }
    } catch (e) {
        console.warn("Could not load analytics summary:", e);
    }
}

// ============================================================================
// USER FEEDBACK & SUPABASE RATING MODAL
// ============================================================================
window.openFeedbackModal = function(predictionId, emotionName) {
    AppState.currentPredictionId = predictionId || AppState.currentPredictionId;
    const targetEmotionSelect = document.getElementById("feedback-correct-emotion");
    if (targetEmotionSelect && emotionName) {
        targetEmotionSelect.value = emotionName;
    }
    openModal("modal-feedback");
};

function initModals() {
    // Star rating picker
    const starSpans = document.querySelectorAll("#star-picker span");
    starSpans.forEach(star => {
        star.addEventListener("click", () => {
            const rating = parseInt(star.dataset.star);
            AppState.selectedRating = rating;
            starSpans.forEach((s, idx) => {
                if (idx < rating) s.classList.add("active");
                else s.classList.remove("active");
            });
        });
    });

    // Submit Feedback button
    const btnSubmitFeedback = document.getElementById("btn-submit-feedback");
    if (btnSubmitFeedback) {
        btnSubmitFeedback.addEventListener("click", async () => {
            if (!AppState.currentPredictionId) {
                alert("No active prediction ID selected.");
                return;
            }
            const correctEmo = document.getElementById("feedback-correct-emotion")?.value;
            const comments = document.getElementById("feedback-comments")?.value;

            try {
                const res = await fetch(getApiUrl("/api/feedback"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        prediction_id: AppState.currentPredictionId,
                        rating: AppState.selectedRating,
                        user_corrected_emotion: correctEmo,
                        user_feedback: comments
                    })
                });
                const data = await res.json();
                if (data.status === "success") {
                    alert("✓ Feedback recorded successfully!");
                    closeModal("modal-feedback");
                    loadSupabaseHistory();
                } else {
                    alert(`Failed: ${data.message}`);
                }
            } catch (err) {
                alert("Network error submitting feedback.");
            }
        });
    }
}

window.openModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add("active");
};

window.closeModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove("active");
};

// ============================================================================
// MODEL BENCHMARKS & CONFUSION MATRIX
// ============================================================================
async function loadBenchmarks() {
    try {
        const response = await fetch(getApiUrl("/api/metrics"));
        const data = await response.json();
        if (data.status === "success") {
            renderBenchmarkMetrics(data.metrics, data.best_model);
            renderConfusionMatrix(data.confusion_matrices[data.best_model]);
            renderFeatureImportance(data.feature_importances["RandomForest"]);
        }
    } catch (e) {
        console.warn("Could not load benchmarks:", e);
    }
}

function renderBenchmarkMetrics(metrics, bestModel) {
    if (!metrics) return;
    for (const [modelName, m] of Object.entries(metrics)) {
        const accElem = document.getElementById(`metric-${modelName.toLowerCase()}-acc`);
        const f1Elem = document.getElementById(`metric-${modelName.toLowerCase()}-f1`);
        if (accElem) accElem.textContent = `${(m.accuracy * 100).toFixed(2)}%`;
        if (f1Elem) f1Elem.textContent = `${(m.f1_score * 100).toFixed(2)}%`;
    }
}

function renderConfusionMatrix(cmData) {
    const table = document.getElementById("confusion-matrix-table");
    if (!table || !cmData) return;

    const labels = cmData.labels;
    const matrix = cmData.matrix;

    let headerHtml = `<thead><tr><th>Actual \\ Pred</th>${labels.map(l => `<th>${l.slice(0,3).toUpperCase()}</th>`).join("")}</tr></thead>`;
    let bodyHtml = `<tbody>${matrix.map((row, rIdx) => `
        <tr>
            <th style="text-align:left;">${labels[rIdx].slice(0,3).toUpperCase()}</th>
            ${row.map((val, cIdx) => {
                const isDiag = rIdx === cIdx;
                const bg = isDiag ? 'rgba(139, 92, 246, 0.45)' : (val > 0 ? 'rgba(244, 63, 94, 0.2)' : 'transparent');
                return `<td style="background:${bg}; font-weight:${isDiag ? '700' : '400'}; color:${isDiag ? '#ffffff' : 'var(--text-muted)'}">${val}</td>`;
            }).join("")}
        </tr>
    `).join("")}</tbody>`;

    table.innerHTML = headerHtml + bodyHtml;
}

function renderFeatureImportance(feats) {
    const ctx = document.getElementById("chart-feat-imp");
    if (!ctx || !feats) return;

    const labels = feats.map(f => f.feature.replace("_mean", ""));
    const values = feats.map(f => f.importance * 100);

    if (featImpChart) featImpChart.destroy();

    featImpChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: 'rgba(6, 182, 212, 0.7)',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 9 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 9 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================
function initBatchProcessing() {
    const btnRunBatch = document.getElementById("btn-run-batch-demo");
    const tbody = document.getElementById("batch-results-body");
    if (!btnRunBatch) return;

    btnRunBatch.addEventListener("click", async () => {
        btnRunBatch.textContent = "Processing Batch...";
        btnRunBatch.disabled = true;

        try {
            const res = await fetch(getApiUrl("/api/batch-predict"), { method: "POST" });
            const data = await res.json();

            if (data.status === "success" && data.results && tbody) {
                tbody.innerHTML = data.results.map(r => `
                    <tr>
                        <td>${r.id}</td>
                        <td>${r.filename}</td>
                        <td><span class="emotion-pill ${r.top_emotion.toLowerCase()}">${r.emoji} ${r.top_emotion}</span></td>
                        <td>${r.confidence}%</td>
                        <td>${r.valence >= 0 ? '+' : ''}${r.valence}</td>
                        <td>${r.arousal >= 0 ? '+' : ''}${r.arousal}</td>
                    </tr>
                `).join("");
                loadSupabaseHistory();
            }
        } catch (e) {
            alert("Failed to execute batch test.");
        } finally {
            btnRunBatch.textContent = "Run 8-Class Benchmark Batch";
            btnRunBatch.disabled = false;
        }
    });
}

// ============================================================================
// API PLAYGROUND SNIPPETS
// ============================================================================
function initApiTabs() {
    const tabs = document.querySelectorAll(".api-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const lang = tab.dataset.lang;
            document.querySelectorAll(".code-block").forEach(b => b.classList.remove("active"));
            const target = document.getElementById(`snippet-${lang}`);
            if (target) target.classList.add("active");
        });
    });
}

function updateApiSnippets() {
    const base = AppState.apiBaseUrl;
    const curlElem = document.getElementById("snippet-curl");
    const jsElem = document.getElementById("snippet-js");
    const pyElem = document.getElementById("snippet-py");

    if (curlElem) {
        curlElem.textContent = `curl -X POST "${base}/api/predict" \\\n  -F "file=@/path/to/speech.wav" \\\n  -H "Accept: application/json"`;
    }

    if (jsElem) {
        jsElem.textContent = `const formData = new FormData();\nformData.append("file", audioBlob, "speech.wav");\n\nconst response = await fetch("${base}/api/predict", {\n  method: "POST",\n  body: formData\n});\nconst data = await response.json();\nconsole.log(data.result.consensus.top_emotion);`;
    }

    if (pyElem) {
        pyElem.textContent = `import requests\n\nurl = "${base}/api/predict"\nwith open("speech.wav", "rb") as f:\n    files = {"file": f}\n    response = requests.post(url, files=files)\n\nprint(response.json())`;
    }
}
