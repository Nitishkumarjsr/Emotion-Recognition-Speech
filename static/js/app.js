/**
 * VoxEmotion AI - Speech Emotion Recognition Frontend Controller
 * Manages live microphone recording with animated waveform, audio playback,
 * 2D Circumplex Affect mapping, and multi-model emotion inference.
 */

document.addEventListener("DOMContentLoaded", () => {
    initModeTabs();
    initMicrophoneRecorder();
    initAudioUpload();
    initBatchProcessing();
    initApiTabs();
    loadBenchmarks();
    
    // Load initial sample (Happy)
    loadSample("happy");
});

// Chart references
let circumplexChart = null;
let emotionProbsChart = null;
let modelCompChart = null;
let featImpChart = null;

// ==========================================
// Mode Tabs
// ==========================================
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

// ==========================================
// Sample Loader
// ==========================================
window.loadSample = async function(emotionId) {
    const player = document.getElementById("main-audio-player");
    const playerTitle = document.getElementById("player-title");

    if (playerTitle) playerTitle.textContent = `${emotionId}_sample.wav`;
    if (player) {
        player.src = `/static/samples/${emotionId}_sample.wav`;
        player.play().catch(() => {}); // Autoplay if permitted by browser
    }

    try {
        const res = await fetch(`/api/sample/${emotionId}`);
        const data = await res.json();
        if (data.status === "success") {
            renderEmotionResults(data.result);
        }
    } catch (e) {
        console.error("Failed to load sample:", e);
    }
};

// ==========================================
// Live Microphone Recording & Web Audio Waveform
// ==========================================
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animationFrameId = null;
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;

function initMicrophoneRecorder() {
    const recordBtn = document.getElementById("btn-record-toggle");
    if (!recordBtn) return;

    recordBtn.addEventListener("click", () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
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

        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioUrl = URL.createObjectURL(audioBlob);

            const player = document.getElementById("main-audio-player");
            const playerTitle = document.getElementById("player-title");
            if (player) player.src = audioUrl;
            if (playerTitle) playerTitle.textContent = "microphone_recording.wav";

            // Send to API
            const formData = new FormData();
            formData.append("file", audioBlob, "recording.wav");
            try {
                const res = await fetch("/api/predict", { method: "POST", body: formData });
                const data = await res.json();
                if (data.status === "success") {
                    renderEmotionResults(data.result);
                }
            } catch (e) {
                console.error("Inference failed:", e);
            }

            // Stop audio tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        recordStartTime = Date.now();

        const btn = document.getElementById("btn-record-toggle");
        const lbl = document.getElementById("rec-label");
        if (btn) btn.classList.add("recording");
        if (lbl) lbl.textContent = "Stop & Analyze Recording";

        timerInterval = setInterval(updateTimer, 100);
        drawWaveform();
    } catch (e) {
        alert("Microphone access denied or not available. Please allow mic permissions in your browser.");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        const btn = document.getElementById("btn-record-toggle");
        const lbl = document.getElementById("rec-label");
        if (btn) btn.classList.remove("recording");
        if (lbl) lbl.textContent = "Start Recording";

        clearInterval(timerInterval);
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    }
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    const timerEl = document.getElementById("mic-timer");
    if (timerEl) timerEl.textContent = `${mins}:${secs}`;
}

function drawWaveform() {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
        if (!isRecording) return;
        animationFrameId = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "#04030a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#8b5cf6";
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    };

    draw();
}

// ==========================================
// Audio Upload Dropzone
// ==========================================
function initAudioUpload() {
    const fileInput = document.getElementById("audio-file-input");
    const dropzone = document.getElementById("audio-dropzone");

    if (fileInput) {
        fileInput.addEventListener("change", e => {
            if (e.target.files.length > 0) processAudioFile(e.target.files[0]);
        });
    }

    if (dropzone) {
        dropzone.addEventListener("dragover", e => {
            e.preventDefault();
            dropzone.style.borderColor = "#8b5cf6";
        });
        dropzone.addEventListener("dragleave", () => {
            dropzone.style.borderColor = "rgba(255,255,255,0.15)";
        });
        dropzone.addEventListener("drop", e => {
            e.preventDefault();
            dropzone.style.borderColor = "rgba(255,255,255,0.15)";
            if (e.dataTransfer.files.length > 0) processAudioFile(e.dataTransfer.files[0]);
        });
    }
}

async function processAudioFile(file) {
    const player = document.getElementById("main-audio-player");
    const playerTitle = document.getElementById("player-title");
    if (player) player.src = URL.createObjectURL(file);
    if (playerTitle) playerTitle.textContent = file.name;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (data.status === "success") {
            renderEmotionResults(data.result);
        } else {
            alert("Audio processing error: " + data.message);
        }
    } catch (e) {
        console.error("Audio upload error:", e);
    }
}

// ==========================================
// Render Emotion Inference Results
// ==========================================
function renderEmotionResults(result) {
    const { consensus, acoustic_features } = result;

    // 1. Verdict card
    const emojiEl = document.getElementById("verdict-emoji");
    const emotionEl = document.getElementById("verdict-emotion");
    const descEl = document.getElementById("verdict-desc");
    const confEl = document.getElementById("consensus-confidence");
    const secTag = document.getElementById("secondary-tag");

    if (emojiEl) emojiEl.textContent = consensus.emoji;
    if (emotionEl) {
        emotionEl.textContent = consensus.label.toUpperCase();
        emotionEl.style.color = consensus.color;
    }
    if (descEl) descEl.textContent = consensus.description;
    if (confEl) confEl.textContent = `${consensus.confidence}% Confidence`;
    if (secTag) {
        secTag.innerHTML = `Secondary: <strong>${consensus.secondary_emotion} (${consensus.secondary_confidence}%)</strong>`;
    }

    // 2. Biometrics
    if (acoustic_features) {
        const setBio = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        setBio("bio-pitch", `${Math.round(acoustic_features.pitch_f0_mean || 140)} Hz`);
        setBio("bio-rms", Number(acoustic_features.rms_energy || 0).toFixed(3));
        setBio("bio-centroid", `${Number(acoustic_features.spectral_centroid || 0).toLocaleString()} Hz`);
        setBio("bio-zcr", Number(acoustic_features.zcr_mean || 0).toFixed(3));
    }

    // 3. 2D Circumplex Plot
    renderCircumplexChart(consensus.valence, consensus.arousal, consensus.label, consensus.color);

    // 4. Probability Spectrum Bar / Radar
    renderEmotionProbsChart(consensus.probabilities);
}

function renderCircumplexChart(val, aro, label, color) {
    const ctx = document.getElementById("chart-circumplex");
    const coordsEl = document.getElementById("affect-coords");
    if (!ctx) return;

    if (coordsEl) {
        coordsEl.textContent = `Valence: ${val >= 0 ? '+' : ''}${val.toFixed(2)} | Arousal: ${aro >= 0 ? '+' : ''}${aro.toFixed(2)}`;
    }

    // Fixed Emotion Anchor Landmarks
    const anchors = [
        { x: 0.85, y: 0.65, label: "Happy", color: "#10b981" },
        { x: -0.65, y: 0.9, label: "Angry", color: "#f43f5e" },
        { x: -0.7, y: -0.6, label: "Sad", color: "#6366f1" },
        { x: 0.45, y: -0.4, label: "Calm", color: "#38bdf8" },
        { x: 0.0, y: 0.0, label: "Neutral", color: "#94a3b8" },
        { x: -0.75, y: 0.7, label: "Fearful", color: "#a855f7" },
        { x: 0.35, y: 0.8, label: "Surprised", color: "#ec4899" },
        { x: -0.8, y: -0.1, label: "Disgust", color: "#f59e0b" }
    ];

    if (circumplexChart) circumplexChart.destroy();

    circumplexChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Current Voice Sample",
                    data: [{ x: val, y: aro }],
                    backgroundColor: color || "#8b5cf6",
                    borderColor: "#ffffff",
                    borderWidth: 2.5,
                    pointRadius: 9,
                    pointHoverRadius: 11
                },
                {
                    label: "Reference Affect Anchors",
                    data: anchors.map(a => ({ x: a.x, y: a.y })),
                    backgroundColor: "rgba(255, 255, 255, 0.2)",
                    borderColor: "rgba(255, 255, 255, 0.4)",
                    borderWidth: 1,
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.datasetIndex === 0) return `${label} (V: ${val}, A: ${aro})`;
                            return anchors[ctx.dataIndex].label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    min: -1.0,
                    max: 1.0,
                    title: { display: true, text: "← Negative Valence | Positive Valence →", color: "#64748b", font: { size: 10 } },
                    grid: { color: "rgba(255, 255, 255, 0.06)" },
                    ticks: { color: "#64748b" }
                },
                y: {
                    min: -1.0,
                    max: 1.0,
                    title: { display: true, text: "← Low Arousal | High Arousal →", color: "#64748b", font: { size: 10 } },
                    grid: { color: "rgba(255, 255, 255, 0.06)" },
                    ticks: { color: "#64748b" }
                }
            }
        }
    });
}

function renderEmotionProbsChart(probs) {
    const ctx = document.getElementById("chart-emotion-probs");
    if (!ctx || !probs) return;

    const labels = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"];
    const keys = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];
    const values = keys.map(k => probs[k] || 0.0);
    const colors = ["#94a3b8", "#38bdf8", "#10b981", "#6366f1", "#f43f5e", "#a855f7", "#f59e0b", "#ec4899"];

    if (emotionProbsChart) emotionProbsChart.destroy();

    emotionProbsChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Probability (%)",
                data: values,
                backgroundColor: colors.map(c => c + "cc"),
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { weight: "600" } }
                }
            }
        }
    });
}

// ==========================================
// Benchmarks & Metrics
// ==========================================
async function loadBenchmarks() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        if (data.status === "success") {
            renderModelComparisonChart(data.metrics);
            renderFeatureImportanceChart(data.feature_importances);
            renderConfusionMatrixTable(data.confusion_matrices[data.best_model]);
            renderMetricsTable(data.metrics);
        }
    } catch (e) {
        console.error("Failed to load benchmarks:", e);
    }
}

function renderModelComparisonChart(metrics) {
    const ctx = document.getElementById("chart-model-comparison");
    if (!ctx) return;

    const labels = Object.keys(metrics);
    const accs = labels.map(l => (metrics[l].accuracy * 100).toFixed(2));
    const f1s = labels.map(l => (metrics[l].f1_score * 100).toFixed(2));

    if (modelCompChart) modelCompChart.destroy();

    modelCompChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Accuracy (%)",
                    data: accs,
                    backgroundColor: "rgba(139, 92, 246, 0.75)",
                    borderColor: "#8b5cf6",
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: "Weighted F1 (%)",
                    data: f1s,
                    backgroundColor: "rgba(6, 182, 212, 0.75)",
                    borderColor: "#06b6d4",
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#94a3b8" } }
            },
            scales: {
                y: {
                    min: 70,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { weight: "600" } }
                }
            }
        }
    });
}

function renderFeatureImportanceChart(featureImportances) {
    const ctx = document.getElementById("chart-feature-importance");
    if (!ctx) return;

    const list = featureImportances.RandomForest || [];
    const top10 = list.slice(0, 10);
    const labels = top10.map(item => item.feature);
    const vals = top10.map(item => item.importance);

    if (featImpChart) featImpChart.destroy();

    featImpChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Acoustic Importance",
                data: vals,
                backgroundColor: "rgba(139, 92, 246, 0.75)",
                borderColor: "#8b5cf6",
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { size: 10, family: "JetBrains Mono" } }
                }
            }
        }
    });
}

function renderConfusionMatrixTable(cmData) {
    const thead = document.getElementById("cm-thead");
    const tbody = document.getElementById("cm-tbody");
    if (!thead || !tbody || !cmData) return;

    const labels = cmData.labels;
    const matrix = cmData.matrix;

    // Header row
    thead.innerHTML = `<tr><th>True \\ Pred</th>${labels.map(l => `<th>${l.slice(0, 4).toUpperCase()}</th>`).join("")}</tr>`;

    // Body rows
    tbody.innerHTML = "";
    matrix.forEach((row, i) => {
        const tr = document.createElement("tr");
        const cells = row.map((val, j) => `<td class="${i === j ? 'cm-cell-diag' : ''}">${val}</td>`).join("");
        tr.innerHTML = `<th>${labels[i].slice(0, 4).toUpperCase()}</th>${cells}`;
        tbody.appendChild(tr);
    });
}

function renderMetricsTable(metrics) {
    const tbody = document.getElementById("metrics-tbody");
    if (!tbody) return;

    tbody.innerHTML = "";
    Object.keys(metrics).forEach(name => {
        const m = metrics[name];
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${name}</strong></td>
            <td><strong>${(m.accuracy * 100).toFixed(2)}%</strong></td>
            <td>${(m.precision * 100).toFixed(2)}%</td>
            <td>${(m.recall * 100).toFixed(2)}%</td>
            <td><strong style="color:#a78bfa;">${(m.f1_score * 100).toFixed(2)}%</strong></td>
            <td>${m.test_samples}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ==========================================
// Batch Voice Processing
// ==========================================
function initBatchProcessing() {
    const batchBtn = document.getElementById("btn-run-batch-demo");
    if (!batchBtn) return;

    batchBtn.addEventListener("click", async () => {
        batchBtn.style.opacity = "0.6";
        try {
            const res = await fetch("/api/batch-predict", { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                renderBatchResults(data);
            }
        } catch (e) {
            console.error("Batch processing failed:", e);
        } finally {
            batchBtn.style.opacity = "1";
        }
    });
}

function renderBatchResults(data) {
    document.getElementById("batch-total").textContent = data.total_processed;
    document.getElementById("batch-top-emo").textContent = "Varied Spectrum (8 Classes)";

    const wrapper = document.getElementById("batch-results-wrapper");
    const tbody = document.getElementById("batch-results-tbody");
    if (wrapper) wrapper.style.display = "block";
    if (tbody) {
        tbody.innerHTML = "";
        data.results.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><code>#${r.id}</code></td>
                <td>${r.filename}</td>
                <td><strong>${r.emoji} ${r.top_emotion}</strong></td>
                <td>${r.confidence}%</td>
                <td>${r.valence}</td>
                <td>${r.arousal}</td>
            `;
            tbody.appendChild(tr);
        });
    }
}

// ==========================================
// API Playground
// ==========================================
function initApiTabs() {
    const tabs = document.querySelectorAll(".api-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const lang = tab.dataset.lang;
            document.querySelectorAll(".code-block").forEach(b => b.classList.remove("active"));
            const target = document.getElementById(`code-${lang}`);
            if (target) target.classList.add("active");
        });
    });
}
