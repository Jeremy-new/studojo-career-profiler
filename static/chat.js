/**
 * CandidateProfiler — Chat JavaScript (Dynamic Mode v4)
 * - MCQ chips render INLINE in chat messages (centered)
 * - Handles ||| separator for split bubbles (ack + question)
 * - "Other" option shows inline text input
 * - CTC/salary shows min/max boxes
 * - Auto-scroll to CENTER, not bottom
 * - Strips em dashes from all messages
 * - Enter key confirms multi-select
 */

// ============================================================================
// State
// ============================================================================

const state = {
    sessionId: '',
    currentView: 'upload',
    isProcessing: false,
    payload: null,
    selectedChips: new Set(),
    questionNumber: 0,
    totalQuestions: 10,
};

// ============================================================================
// DOM
// ============================================================================

const $ = (id) => document.getElementById(id);

const els = {
    sessionId: $('session-id'),
    uploadSection: $('upload-section'),
    chatSection: $('chat-section'),
    payloadSection: $('payload-section'),
    uploadZone: $('upload-zone'),
    fileInput: $('file-input'),
    uploadProgress: $('upload-progress'),
    progressFill: $('progress-fill'),
    progressText: $('progress-text'),
    resumePreview: $('resume-preview'),
    previewBody: $('preview-body'),
    startChatBtn: $('start-chat-btn'),
    skipBtn: $('skip-btn'),
    chatMessages: $('chat-messages'),
    typingIndicator: $('typing-indicator'),
    chatInputArea: $('chat-input-area'),
    chatInput: $('chat-input'),
    sendBtn: $('send-btn'),
    chatProgress: $('chat-progress'),
    progressLabel: $('progress-label'),
    payloadContent: $('payload-content'),
    downloadJsonBtn: $('download-json-btn'),
    startOverBtn: $('start-over-btn'),
};

// ============================================================================
// Init
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    state.sessionId = els.sessionId.value;
    initUpload();
    initChat();
    initPayload();
});

// ============================================================================
// View Switching
// ============================================================================

function switchView(view) {
    state.currentView = view;
    els.uploadSection.classList.remove('active');
    els.chatSection.classList.remove('active');
    els.payloadSection.classList.remove('active');

    switch (view) {
        case 'upload':
            els.uploadSection.classList.add('active');
            break;
        case 'chat':
            els.chatSection.classList.add('active');
            if (els.chatMessages.querySelectorAll('.message').length === 0) {
                sendMessage('', true);
            }
            break;
        case 'payload':
            els.payloadSection.classList.add('active');
            break;
    }
}

// ============================================================================
// Progress Stepper
// ============================================================================

function updateProgress(questionNum) {
    state.questionNumber = questionNum;
    if (!els.chatProgress) return;

    const dots = els.chatProgress.querySelectorAll('.progress-dot');
    dots.forEach((dot, i) => {
        dot.classList.remove('active', 'completed');
        if (i < questionNum - 1) dot.classList.add('completed');
        else if (i === questionNum - 1) dot.classList.add('active');
    });

    if (els.progressLabel) {
        els.progressLabel.textContent = questionNum > state.totalQuestions
            ? 'Done ✓' : `${questionNum}/${state.totalQuestions}`;
    }
}

// ============================================================================
// Scroll — center the latest content in the chat frame
// ============================================================================

function scrollToCenter() {
    setTimeout(() => {
        // Find the last message or interactive block
        const allItems = els.chatMessages.querySelectorAll('.message, .mcq-inline-block, .ctc-inline-block');
        const lastItem = allItems[allItems.length - 1];
        if (!lastItem) return;

        lastItem.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
        });
    }, 100);
}

// ============================================================================
// Text Cleanup — strip em dashes
// ============================================================================

function cleanText(str) {
    if (!str) return '';
    // Replace em dashes with comma-space
    return str.replace(/\u2014/g, ',').replace(/—/g, ',');
}

// ============================================================================
// Upload
// ============================================================================

function initUpload() {
    els.uploadZone.addEventListener('click', () => els.fileInput.click());

    els.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
    });

    els.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.uploadZone.classList.add('dragover');
    });
    els.uploadZone.addEventListener('dragleave', () => {
        els.uploadZone.classList.remove('dragover');
    });
    els.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    });

    els.skipBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/skip-resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.sessionId }),
            });
            switchView('chat');
        } catch (err) {
            console.error('Skip error:', err);
            switchView('chat');
        }
    });

    els.startChatBtn.addEventListener('click', () => switchView('chat'));
}

async function handleFileUpload(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'doc'].includes(ext)) {
        alert('Please upload a PDF or DOCX file.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Max 10MB.');
        return;
    }

    console.log('[UPLOAD] Starting upload for:', file.name);
    els.uploadZone.classList.add('hidden');
    els.uploadProgress.classList.remove('hidden');
    els.progressFill.style.width = '30%';
    els.progressText.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('file', file);

    // Safety timeout: if upload takes > 20s, go to chat anyway
    const safetyTimeout = setTimeout(() => {
        console.warn('[UPLOAD] Safety timeout reached, transitioning to chat');
        els.uploadProgress.classList.add('hidden');
        switchView('chat');
    }, 20000);

    try {
        els.progressFill.style.width = '60%';
        els.progressText.textContent = 'Parsing resume...';

        const res = await fetch(`/api/upload-resume?session_id=${state.sessionId}`, {
            method: 'POST',
            body: formData,
        });

        console.log('[UPLOAD] Server responded:', res.status);

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await res.json();
        console.log('[UPLOAD] Response data:', JSON.stringify(data).substring(0, 200));

        clearTimeout(safetyTimeout);
        els.progressFill.style.width = '100%';
        els.progressText.textContent = 'Done!';

        // Show preview, then auto-transition to chat
        setTimeout(() => {
            els.uploadProgress.classList.add('hidden');
            try {
                if (data.summary) showResumePreview(data.summary);
            } catch (previewErr) {
                console.error('[UPLOAD] Preview rendering failed:', previewErr);
            }
            // Auto-start chat after 2 seconds (user doesn't have to click)
            setTimeout(() => switchView('chat'), 2000);
        }, 300);

    } catch (err) {
        clearTimeout(safetyTimeout);
        console.error('[UPLOAD] Error:', err);
        els.uploadProgress.classList.add('hidden');
        els.uploadZone.classList.remove('hidden');
        alert('Failed to upload resume: ' + err.message);
    }
}

function showResumePreview(summary) {
    let html = '';
    if (summary.name) html += `<p><strong>${escapeHtml(summary.name)}</strong></p>`;
    if (summary.email) html += `<p style="color:var(--text-secondary)">${escapeHtml(summary.email)}</p>`;
    if (summary.skills && summary.skills.length > 0) {
        const tags = summary.skills.slice(0, 12).map(s =>
            `<span class="preview-tag">${escapeHtml(s)}</span>`
        ).join('');
        html += `<p style="margin-top:10px">🛠️ ${tags}</p>`;
    }
    if (summary.summary_text) {
        html += `<p style="margin-top:10px;color:var(--text-secondary);font-size:13px">${escapeHtml(summary.summary_text)}</p>`;
    }
    els.previewBody.innerHTML = html;
    els.resumePreview.classList.remove('hidden');
}

// ============================================================================
// Chat
// ============================================================================

function initChat() {
    els.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const msg = els.chatInput.value.trim();
            if (msg && !state.isProcessing) sendMessage(msg);
        }
    });

    els.sendBtn.addEventListener('click', () => {
        const msg = els.chatInput.value.trim();
        if (msg && !state.isProcessing) sendMessage(msg);
    });

    els.chatInput.addEventListener('input', () => {
        els.sendBtn.disabled = !els.chatInput.value.trim() || state.isProcessing;
    });

    // Global Enter key for confirming multi-select
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            const confirmBtn = els.chatMessages.querySelector('.mcq-confirm-btn.visible');
            if (confirmBtn && !confirmBtn.disabled) {
                e.preventDefault();
                confirmBtn.click();
            }
        }
    });
}

function addMessage(content, role = 'agent') {
    const cleaned = cleanText(content);
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    const avatar = role === 'agent' ? '🎯' : '👤';
    msgDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-bubble">${escapeHtml(cleaned)}</div>
    `;
    els.chatMessages.appendChild(msgDiv);
    scrollToCenter();
}

let _typingTimer = null;

function showTyping() {
    // Move indicator to the end of chat messages so it appears BELOW the latest content
    els.chatMessages.appendChild(els.typingIndicator);
    els.typingIndicator.classList.remove('hidden');

    // Add a visible countdown so user knows it's working
    let seconds = 0;
    const dotsEl = els.typingIndicator.querySelector('.typing-dots');
    if (dotsEl) {
        _typingTimer = setInterval(() => {
            seconds++;
            dotsEl.innerHTML = `<span style="font-size:13px;color:#7c3aed">Thinking... ${seconds}s</span>`;
        }, 1000);
    }
    scrollToCenter();
}

function hideTyping() {
    els.typingIndicator.classList.add('hidden');
    if (_typingTimer) {
        clearInterval(_typingTimer);
        _typingTimer = null;
    }
    // Restore original dots
    const dotsEl = els.typingIndicator.querySelector('.typing-dots');
    if (dotsEl) {
        dotsEl.innerHTML = '<span></span><span></span><span></span>';
    }
}

// ============================================================================
// MCQ Inline — with "Other" option text input
// ============================================================================

function showMCQInline(mcq) {
    state.selectedChips.clear();
    const allowMultiple = mcq.allow_multiple || false;

    const mcqBlock = document.createElement('div');
    mcqBlock.className = 'mcq-inline-block';

    // Multi-select note
    if (allowMultiple) {
        const note = document.createElement('div');
        note.className = 'mcq-note';
        note.textContent = '* You can select multiple options';
        mcqBlock.appendChild(note);
    }

    const chipsWrapper = document.createElement('div');
    chipsWrapper.className = 'mcq-chips';

    let otherInput = null; // reference to the "Other" text input

    mcq.options.forEach(opt => {
        const isOther = opt.text.toLowerCase().startsWith('other');
        const chip = document.createElement('button');
        chip.className = 'mcq-chip';
        chip.dataset.label = opt.label;
        chip.dataset.text = opt.text;

        const checkSpan = document.createElement('span');
        checkSpan.className = 'chip-check';

        const textSpan = document.createElement('span');
        textSpan.textContent = opt.text;

        chip.appendChild(checkSpan);
        chip.appendChild(textSpan);

        chip.addEventListener('click', () => {
            if (isOther) {
                // Toggle "Other" chip and show/hide text input
                chip.classList.toggle('selected');
                if (chip.classList.contains('selected')) {
                    checkSpan.textContent = '✓';
                    state.selectedChips.add('Other');
                    if (!otherInput) {
                        otherInput = document.createElement('div');
                        otherInput.className = 'other-input-wrap';
                        otherInput.innerHTML = `
                            <input type="text" class="other-text-input" placeholder="Please specify..." autocomplete="off">
                        `;
                        mcqBlock.insertBefore(otherInput, mcqBlock.querySelector('.mcq-confirm-btn'));
                        const inp = otherInput.querySelector('input');
                        inp.focus();
                        inp.addEventListener('keydown', (e) => {
                            if (e.key === 'Enter') {
                                e.preventDefault();
                                const confirmBtn = mcqBlock.querySelector('.mcq-confirm-btn');
                                if (confirmBtn) confirmBtn.click();
                            }
                        });
                    }
                    otherInput.classList.remove('hidden');
                } else {
                    checkSpan.textContent = '';
                    state.selectedChips.delete('Other');
                    if (otherInput) otherInput.classList.add('hidden');
                }
                updateConfirmVisibility(mcqBlock);
            } else if (allowMultiple) {
                chip.classList.toggle('selected');
                if (chip.classList.contains('selected')) {
                    state.selectedChips.add(opt.label);
                    checkSpan.textContent = '✓';
                } else {
                    state.selectedChips.delete(opt.label);
                    checkSpan.textContent = '';
                }
                updateConfirmVisibility(mcqBlock);
            } else {
                // Single select: disable all, send immediately
                chipsWrapper.querySelectorAll('.mcq-chip').forEach(c => {
                    c.disabled = true;
                    c.style.pointerEvents = 'none';
                    c.style.opacity = '0.5';
                });
                chip.style.opacity = '1';
                chip.classList.add('selected');
                checkSpan.textContent = '✓';

                if (isOther && otherInput) {
                    const customText = otherInput.querySelector('input').value.trim();
                    sendMessage(customText || 'Other');
                } else {
                    sendMessage(opt.text);
                }
            }
        });

        chipsWrapper.appendChild(chip);
    });

    mcqBlock.appendChild(chipsWrapper);

    // Confirm button for multi-select
    if (allowMultiple) {
        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'mcq-confirm-btn';
        confirmBtn.disabled = true;
        confirmBtn.innerHTML = 'Confirm Selection →';

        confirmBtn.addEventListener('click', () => {
            if (state.selectedChips.size === 0) return;

            chipsWrapper.querySelectorAll('.mcq-chip').forEach(c => {
                c.disabled = true;
                c.style.pointerEvents = 'none';
            });
            confirmBtn.disabled = true;
            confirmBtn.style.pointerEvents = 'none';
            confirmBtn.style.opacity = '0.5';

            const selectedTexts = [];
            chipsWrapper.querySelectorAll('.mcq-chip.selected').forEach(chip => {
                if (chip.dataset.text.toLowerCase().startsWith('other') && otherInput) {
                    const customText = otherInput.querySelector('input').value.trim();
                    if (customText) selectedTexts.push(customText);
                    else selectedTexts.push('Other');
                } else {
                    selectedTexts.push(chip.dataset.text);
                }
            });
            if (otherInput) {
                otherInput.querySelector('input').disabled = true;
            }
            sendMessage(selectedTexts.join(', '));
        });

        mcqBlock.appendChild(confirmBtn);
    }

    els.chatMessages.appendChild(mcqBlock);
    scrollToCenter();
}

function updateConfirmVisibility(mcqBlock) {
    const confirmBtn = mcqBlock.querySelector('.mcq-confirm-btn');
    if (confirmBtn) {
        confirmBtn.classList.toggle('visible', state.selectedChips.size > 0);
        confirmBtn.disabled = state.selectedChips.size === 0;
    }
}

// ============================================================================
// CTC / Salary Text Input
// ============================================================================

function showCTCInput() {
    const ctcBlock = document.createElement('div');
    ctcBlock.className = 'ctc-inline-block';

    ctcBlock.innerHTML = `
        <div class="ctc-label">Enter your expected range (in LPA)</div>
        <div class="ctc-inputs">
            <div class="ctc-field">
                <label>Min</label>
                <div class="ctc-input-wrap">
                    <span class="ctc-prefix">₹</span>
                    <input type="number" class="ctc-input" id="ctc-min" placeholder="e.g. 6" min="0" max="200">
                    <span class="ctc-suffix">LPA</span>
                </div>
            </div>
            <span class="ctc-dash">,</span>
            <div class="ctc-field">
                <label>Max</label>
                <div class="ctc-input-wrap">
                    <span class="ctc-prefix">₹</span>
                    <input type="number" class="ctc-input" id="ctc-max" placeholder="e.g. 15" min="0" max="200">
                    <span class="ctc-suffix">LPA</span>
                </div>
            </div>
        </div>
        <button class="mcq-confirm-btn visible" id="ctc-submit-btn">Submit →</button>
    `;

    els.chatMessages.appendChild(ctcBlock);

    const minInput = ctcBlock.querySelector('#ctc-min');
    const maxInput = ctcBlock.querySelector('#ctc-max');
    const submitBtn = ctcBlock.querySelector('#ctc-submit-btn');

    const handleEnter = (e) => {
        if (e.key === 'Enter') { e.preventDefault(); submitBtn.click(); }
    };
    minInput.addEventListener('keydown', handleEnter);
    maxInput.addEventListener('keydown', handleEnter);
    setTimeout(() => minInput.focus(), 100);

    submitBtn.addEventListener('click', () => {
        const min = minInput.value.trim();
        const max = maxInput.value.trim();
        if (!min && !max) { minInput.focus(); return; }
        submitBtn.disabled = true;
        submitBtn.style.pointerEvents = 'none';
        submitBtn.style.opacity = '0.5';
        minInput.disabled = true;
        maxInput.disabled = true;
        sendMessage(`₹${min || '0'} - ₹${max || min} LPA`);
    });

    scrollToCenter();
}

// ============================================================================
// Send Message & Handle Response
// ============================================================================

async function sendMessage(message, isInitial = false, retryCount = 0) {
    if (state.isProcessing && retryCount === 0) return;

    if (retryCount === 0) {
        state.isProcessing = true;
        els.sendBtn.disabled = true;

        if (!isInitial && message) {
            addMessage(message, 'user');
            els.chatInput.value = '';
        }

        showTyping();
    }

    // Client-side timeout: abort fetch after 60s
    const controller = new AbortController();
    const fetchTimeout = setTimeout(() => controller.abort(), 60000);

    try {
        console.log('[CHAT] Sending:', isInitial ? '(initial)' : message, 'retry:', retryCount);
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                message: isInitial ? '' : message,
            }),
            signal: controller.signal,
        });

        clearTimeout(fetchTimeout);
        console.log('[CHAT] Response status:', res.status);
        if (!res.ok) {
            throw new Error(`HTTP Error: ${res.status}`);
        }

        const data = await res.json();
        hideTyping();

        // Handle ||| separator: split into separate bubbles
        const rawMessage = cleanText(data.message || '');
        const parts = rawMessage.split('|||').map(p => p.trim()).filter(p => p);

        if (parts.length > 1) {
            // First part is acknowledgment, rest are questions
            parts.forEach(part => addMessage(part, 'agent'));
        } else {
            addMessage(rawMessage, 'agent');
        }

        // Update progress
        if (data.questions_asked !== undefined) {
            updateProgress(data.questions_asked + 1);
        }

        // Show CTC text input OR MCQ chips inline
        if (data.text_input) {
            showCTCInput();
        } else if (data.mcq && data.mcq.options && data.mcq.options.length > 0) {
            showMCQInline(data.mcq);
        }

        if (data.is_complete) {
            setTimeout(() => generatePayload(), 1000);
        }

        state.isProcessing = false;
        els.sendBtn.disabled = !els.chatInput.value.trim();

    } catch (err) {
        clearTimeout(fetchTimeout);

        // Auto-retry once silently before showing an error
        if (retryCount < 1) {
            console.warn(`[CHAT] Request failed (attempt ${retryCount + 1}), retrying silently...`, err);
            setTimeout(() => sendMessage(message, isInitial, retryCount + 1), 1000);
            return;
        }

        hideTyping();
        state.isProcessing = false;
        els.sendBtn.disabled = !els.chatInput.value.trim();

        if (err.name === 'AbortError' || (err.message && err.message.includes('504'))) {
            console.error('[CHAT] Request timed out completely after retries');
            addMessage('The AI is taking very long to respond. Please try submitting your answer again.', 'agent');
        } else {
            console.error('[CHAT] Error:', err);
            addMessage('Sorry, something went wrong. Please try submitting your answer again.', 'agent');
        }
    }
}

async function generatePayload() {
    showTyping();
    addMessage("Generating your career intelligence report... 📊", 'agent');
    updateProgress(state.totalQuestions + 1);

    try {
        const res = await fetch('/api/generate-payload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.sessionId }),
        });

        if (!res.ok) throw new Error('Failed to generate payload');

        const data = await res.json();
        // Response shape: { payload: {...}, saved_to: "filename.md" }
        state.payload = data.payload || data;
        hideTyping();

        if (data.saved_to) {
            addMessage(`Report saved to Outputs/${data.saved_to} ✅`, 'agent');
        }

        renderPayload(state.payload);
        setTimeout(() => switchView('payload'), 500);

    } catch (err) {
        hideTyping();
        addMessage('Had trouble generating the report. You can keep chatting and try again!', 'agent');
        console.error('Payload error:', err);
    }
}

// ============================================================================
// Payload Display
// ============================================================================

function initPayload() {
    els.downloadJsonBtn.addEventListener('click', () => {
        if (!state.payload) return;
        const blob = new Blob([JSON.stringify(state.payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `career-profile-${state.payload.candidate_id || 'report'}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });

    els.startOverBtn.addEventListener('click', () => window.location.reload());
}

function renderPayload(p) {
    let html = '';

    html += `
        <div class="payload-card">
            <h3>📋 Profile Summary</h3>
            <p>${escapeHtml(cleanText(p.profile_summary))}</p>
        </div>
    `;

    if (p.personal_info) {
        const pi = p.personal_info;
        let piHtml = '';
        if (pi.name) piHtml += `<p><strong>Name:</strong> <span class="value">${escapeHtml(pi.name)}</span></p>`;
        if (pi.education && pi.education.length) {
            piHtml += `<p><strong>Education:</strong> <span class="value">${pi.education.map(e => `${e.degree} in ${e.field}`).join(', ')}</span></p>`;
        }
        if (pi.skills_detected && pi.skills_detected.length) {
            piHtml += `<p style="margin-top:8px">${pi.skills_detected.map(s => `<span class="tag">${escapeHtml(s)}</span>`).join('')}</p>`;
        }
        html += `<div class="payload-card"><h3>👤 Personal Info</h3>${piHtml}</div>`;
    }

    if (p.preferences) {
        const pr = p.preferences;
        html += `
            <div class="payload-card">
                <h3>⚙️ Preferences</h3>
                <p><strong>Locations:</strong> <span class="value">${pr.locations ? pr.locations.join(', ') : 'N/A'}</span></p>
                <p><strong>Work Mode:</strong> <span class="value">${pr.work_mode || 'N/A'}</span></p>
                <p><strong>Company Size:</strong> <span class="value">${pr.company_size || 'N/A'}</span></p>
                <p><strong>Industries:</strong> <span class="value">${pr.industry_interests ? pr.industry_interests.join(', ') : 'N/A'}</span></p>
                <p><strong>Salary Range:</strong> <span class="value">${formatSalary(pr.salary_expectations)}</span></p>
                <p><strong>Timeline:</strong> <span class="value">${pr.timeline || 'N/A'}</span></p>
            </div>
        `;
    }

    if (p.career_analysis) {
        const ca = p.career_analysis;
        html += `
            <div class="payload-card">
                <h3>🎯 Career Analysis</h3>
                <p><strong>Primary Cluster:</strong> <span class="tag">${escapeHtml(ca.primary_cluster)}</span></p>
                ${ca.secondary_cluster ? `<p><strong>Secondary Cluster:</strong> <span class="tag">${escapeHtml(ca.secondary_cluster)}</span></p>` : ''}
            </div>
        `;

        if (ca.recommended_roles && ca.recommended_roles.length) {
            let rolesHtml = ca.recommended_roles.map(r => {
                const scoreClass = r.fit_score >= 0.8 ? 'high' : r.fit_score >= 0.6 ? 'medium' : 'low';
                const scorePercent = Math.round(r.fit_score * 100);
                return `
                    <div class="role-card">
                        <div class="role-title">${escapeHtml(cleanText(r.title))}</div>
                        <div class="role-meta">
                            <span class="fit-score ${scoreClass}">${scorePercent}% fit</span>
                            <span>${r.seniority || 'entry'}-level</span>
                        </div>
                        <div class="role-reasoning">${escapeHtml(cleanText(r.reasoning))}</div>
                    </div>
                `;
            }).join('');
            html += `<div class="payload-card"><h3>💼 Recommended Roles</h3>${rolesHtml}</div>`;
        }
    }

    els.payloadContent.innerHTML = html;
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatSalary(sal) {
    if (!sal) return 'N/A';
    const format = (n) => {
        if (n >= 10000000) return `₹${(n / 10000000).toFixed(1)} Cr`;
        if (n >= 100000) return `₹${(n / 100000).toFixed(1)} LPA`;
        if (n >= 1000) return `₹${(n / 1000).toFixed(0)}K`;
        return `₹${n}`;
    };
    return `${format(sal.min_annual_ctc)} to ${format(sal.max_annual_ctc)} ${sal.currency || ''}`;
}
