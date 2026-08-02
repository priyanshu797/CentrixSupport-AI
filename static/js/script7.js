// ===== DOM Elements =====
const chatBox = document.getElementById("chatBox");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const fileList = document.getElementById("fileList");
const historyList = document.getElementById("historyList");
const historySidebar = document.getElementById("historySidebar");
const chatMain = document.getElementById("chatMain");
const micBtn = document.getElementById("micBtn");

// ===== State Management =====
let uploadedFilePaths = [];
let activeDocumentId = null;
let isSearching = false;
let currentSessionId = Date.now().toString();
let sessions = JSON.parse(localStorage.getItem("chatSessions") || "{}");

// ===== Emotion Configuration =====
const EMOTION_CONFIG = {
  overwhelmed: { emoji: "😰", label: "Overwhelmed", class: "emotion-overwhelmed" },
  sad: { emoji: "😢", label: "Sad", class: "emotion-sad" },
  angry: { emoji: "😠", label: "Angry", class: "emotion-angry" },
  anxious: { emoji: "😨", label: "Anxious", class: "emotion-anxious" },
  neutral: { emoji: "😌", label: "Neutral", class: "emotion-neutral" },
  happy: { emoji: "😊", label: "Happy", class: "emotion-happy" }
};

// ===== Speech Synthesis Setup =====
let currentUtterance = null;
let isSpeaking = false;
let femaleVoice = null;
let currentSpeakBtn = null;

function waitForVoices(timeoutMs = 2000) {
  return new Promise((resolve) => {
    const start = performance.now();
    const check = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices && voices.length) return resolve(voices);
      if (performance.now() - start > timeoutMs) return resolve(voices || []);
      setTimeout(check, 50);
    };
    check();
  });
}

function pickFemaleVoice(voices) {
  const byName = (s) => s ? s.toLowerCase() : "";
  const candidates = voices.filter(v =>
    byName(v.name).includes("female") ||
    byName(v.name).includes("woman") ||
    (v.lang && v.lang.toLowerCase().startsWith("en") && byName(v.name).includes("google"))
  );
  if (candidates.length) return candidates[0];
  const en = voices.find(v => v.lang && v.lang.toLowerCase().startsWith("en"));
  return en || voices[0] || null;
}

async function initVoices() {
  const voices = await waitForVoices();
  femaleVoice = pickFemaleVoice(voices);
}

if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => initVoices();
  initVoices();
}

function resetSpeakUI() {
  isSpeaking = false;
  if (currentSpeakBtn) currentSpeakBtn.innerHTML = "🔊";
  currentSpeakBtn = null;
  currentUtterance = null;
}

function stopAllSpeech() {
  try { 
    window.speechSynthesis.cancel(); 
  } catch (e) {
    console.error("Error stopping speech:", e);
  }
  resetSpeakUI();
}

function toggleSpeech(text, btn) {
  if (isSpeaking) { 
    stopAllSpeech(); 
    return; 
  }
  
  stopAllSpeech();
  currentUtterance = new SpeechSynthesisUtterance(text);
  currentUtterance.voice = femaleVoice || null;
  currentUtterance.pitch = 1.1;
  currentUtterance.rate = 1;
  currentUtterance.onend = resetSpeakUI;
  currentUtterance.onerror = resetSpeakUI;
  
  try {
    window.speechSynthesis.speak(currentUtterance);
    isSpeaking = true;
    currentSpeakBtn = btn || null;
    if (currentSpeakBtn) currentSpeakBtn.innerHTML = "⏹️";
  } catch (e) {
    console.error("Error speaking:", e);
    resetSpeakUI();
  }
}

window.addEventListener("beforeunload", stopAllSpeech);

// ===== Emotion Display Functions =====
function showEmotionBadge(emotion) {
  const config = EMOTION_CONFIG[emotion];
  if (!config || emotion === 'neutral') return;

  const existing = document.querySelector('.emotion-float');
  if (existing) existing.remove();

  const badge = document.createElement('div');
  badge.className = `emotion-badge ${config.class} emotion-float`;
  badge.innerHTML = `
    <span style="font-size: 20px;">${config.emoji}</span>
    <span>Emotion detected: ${config.label}</span>
  `;
  document.body.appendChild(badge);

  setTimeout(() => {
    badge.style.opacity = '0';
    badge.style.transition = 'opacity 0.3s ease-out';
    setTimeout(() => badge.remove(), 300);
  }, 6000);
}

function createEmotionBadge(emotion) {
  const config = EMOTION_CONFIG[emotion];
  if (!config || emotion === 'neutral') return null;

  const badge = document.createElement('div');
  badge.className = `emotion-badge ${config.class}`;
  badge.innerHTML = `
    <span>${config.emoji}</span>
    <span>${config.label}</span>
  `;
  return badge;
}

// ===== Utility Functions =====
function addTimestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function focusUserInput() {
  if (!userInput || userInput.disabled) return;
  window.setTimeout(() => {
    userInput.focus({ preventScroll: true });
    const end = userInput.value.length;
    userInput.setSelectionRange(end, end);
  }, 0);
}

function saveMessage(sessionId, role, text, emotion = null) {
  if (!sessions[sessionId]) {
    sessions[sessionId] = { messages: [], created: new Date().toISOString() };
  }
  sessions[sessionId].messages.push({ 
    role, 
    text, 
    time: addTimestamp(), 
    emotion 
  });
  localStorage.setItem("chatSessions", JSON.stringify(sessions));
  localStorage.setItem("currentSessionId", sessionId);
  renderHistory();
}

function renderHistory() {
  if (!historyList) return;
  
  historyList.innerHTML = "";
  const sortedSessions = Object.entries(sessions).sort((a, b) => {
    const timeA = a[1].created || 0;
    const timeB = b[1].created || 0;
    return new Date(timeB) - new Date(timeA);
  });
  
  sortedSessions.forEach(([sessionId, session]) => {
    const container = document.createElement("div");
    container.className = "relative group";
    
    const firstMessage = session.messages.find(m => m.role === "user");
    const label = firstMessage ? `${firstMessage.text.slice(0, 30)}...` : "New Chat";
    
    const div = document.createElement("div");
    div.className = "text-sm bg-teal-700 p-2 rounded hover:bg-teal-600 cursor-pointer pr-8";
    div.style.color = "white";
    div.style.marginBottom = "8px";
    div.style.transition = "background 0.2s";
    
    const startedAt = session.messages[0]?.time || "now";
    div.textContent = `${label} [${startedAt}]`;
    div.onclick = () => loadSession(sessionId);
    
    const delBtn = document.createElement("button");
    delBtn.innerHTML = "🗑️";
    delBtn.className = "absolute right-2 top-1/2 transform -translate-y-1/2 opacity-0 group-hover:opacity-100";
    delBtn.style.background = "rgba(239, 68, 68, 0.9)";
    delBtn.style.border = "none";
    delBtn.style.borderRadius = "4px";
    delBtn.style.padding = "4px 8px";
    delBtn.style.cursor = "pointer";
    delBtn.style.transition = "opacity 0.2s";
    
    delBtn.onclick = (e) => {
      e.stopPropagation();
      if (confirm("Delete this chat session?")) {
        delete sessions[sessionId];
        localStorage.setItem("chatSessions", JSON.stringify(sessions));
        
        if (currentSessionId === sessionId) {
          const remainingSessions = Object.keys(sessions);
          if (remainingSessions.length > 0) {
            currentSessionId = remainingSessions[0];
            loadSession(currentSessionId);
          } else {
            currentSessionId = Date.now().toString();
            sessions[currentSessionId] = { messages: [], created: new Date().toISOString() };
            localStorage.setItem("chatSessions", JSON.stringify(sessions));
            chatBox.innerHTML = "";
            addMessage("Hi there! 👋 How can I support you today?");
          }
          localStorage.setItem("currentSessionId", currentSessionId);
        }
        renderHistory();
      }
    };
    
    container.appendChild(div);
    container.appendChild(delBtn);
    historyList.appendChild(container);
  });
}

function loadSession(sessionId) {
  currentSessionId = sessionId;
  localStorage.setItem("currentSessionId", sessionId);
  chatBox.innerHTML = "";
  
  const session = sessions[sessionId];
  if (session?.messages && session.messages.length > 0) {
    session.messages.forEach(msg => {
      addMessage(msg.text, msg.role === "user", msg.time, msg.emotion);
    });
  } else {
    addMessage("Hi there! 👋 How can I support you today?");
  }
  focusUserInput();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function compactFileName(value, maxLength = 38) {
  const fullName = String(value || "Document").split(/[\\/]/).pop();
  const cleanName = fullName.replace(/_\d{13,}(?=\.[^.]+$)/, "");
  if (cleanName.length <= maxLength) return cleanName;
  const dotIndex = cleanName.lastIndexOf(".");
  const extension = dotIndex > 0 ? cleanName.slice(dotIndex) : "";
  const baseName = dotIndex > 0 ? cleanName.slice(0, dotIndex) : cleanName;
  const available = Math.max(12, maxLength - extension.length - 1);
  return `${baseName.slice(0, available)}…${extension}`;
}

function formatInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,!?:;])/g, "$1<em>$2</em>");
}

/**
 * Render the small Markdown subset used by LLM and RAG responses.
 * Parsing block-by-block keeps headings, paragraphs and lists on their own
 * lines and escapes model-provided HTML before inserting it into the page.
 */
function formatBotResponse(text) {
  const lines = String(text ?? "")
    .replace(/\\n/g, "\n")
    .replace(/\r\n?/g, "\n")
    .trim()
    .split("\n");
  const html = [];
  let paragraph = [];
  let listType = null;
  let inCodeBlock = false;
  let codeLines = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(formatInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  };

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  };

  const openList = (type) => {
    flushParagraph();
    if (listType === type) return;
    closeList();
    listType = type;
    html.push(`<${type}>`);
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (/^```/.test(trimmed)) {
      flushParagraph();
      closeList();
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
      }
      inCodeBlock = !inCodeBlock;
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(rawLine);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      closeList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${formatInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    // A short label ending in a colon is treated as a section label.
    if (/^[^.!?]{2,64}:$/.test(trimmed) && !/^https?:/i.test(trimmed)) {
      flushParagraph();
      closeList();
      html.push(`<h3>${formatInlineMarkdown(trimmed.slice(0, -1))}</h3>`);
      continue;
    }

    const unordered = trimmed.match(/^[-*•]\s+(.+)$/);
    if (unordered) {
      openList("ul");
      html.push(`<li>${formatInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (ordered) {
      openList("ol");
      html.push(`<li>${formatInlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    const quote = trimmed.match(/^>\s?(.+)$/);
    if (quote) {
      flushParagraph();
      closeList();
      html.push(`<blockquote>${formatInlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    closeList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  closeList();
  if (inCodeBlock && codeLines.length) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }

  return html.join("");
}

function addMessage(text, isUser = false, time = null, emotion = null, hasFiles = false) {
  const wrapper = document.createElement("div");
  wrapper.className = isUser ? "text-right" : "text-left";
  
  const bubble = document.createElement("div");
  const isCrisis = text.includes("🚨") || text.toLowerCase().includes("crisis");
  bubble.className = `inline-block ${isUser ? "user-message" : "bot-message"} ${isCrisis ? "crisis-alert" : ""} animate-fade-in`;
  
  // Add emotion badge for bot messages
  if (!isUser && emotion) {
    const emotionBadge = createEmotionBadge(emotion);
    if (emotionBadge) {
      wrapper.appendChild(emotionBadge);
      showEmotionBadge(emotion);
    }
  }
  
  // Message content
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  
  if (isUser) {
    contentDiv.textContent = text;
  } else {
    contentDiv.innerHTML = formatBotResponse(text);
  }
  
  bubble.appendChild(contentDiv);
  
  // File indicator
  if (hasFiles && !isUser) {
    const fileInd = document.createElement('div');
    fileInd.className = 'file-indicator';
    fileInd.innerHTML = '🔍 Response based on uploaded files';
    bubble.appendChild(fileInd);
  }
  
  // Message footer
  const actionRow = document.createElement("div");
  actionRow.className = "message-footer";
  
  const timestamp = document.createElement("span");
  timestamp.className = "timestamp";
  timestamp.textContent = time || addTimestamp();
  
  const copyBtn = document.createElement("button");
  copyBtn.className = "icon-btn";
  copyBtn.innerHTML = "📋";
  copyBtn.title = "Copy message";
  copyBtn.onclick = () => {
    const plainText = text.replace(/<[^>]*>/g, '');
    navigator.clipboard.writeText(plainText).then(() => {
      copyBtn.innerHTML = "✅";
      setTimeout(() => (copyBtn.innerHTML = "📋"), 1000);
    }).catch(err => {
      console.error("Failed to copy:", err);
    });
  };
  
  const speakBtn = document.createElement("button");
  speakBtn.className = "icon-btn";
  speakBtn.innerHTML = "🔊";
  speakBtn.title = "Read aloud";
  speakBtn.onclick = () => {
    const plainText = text.replace(/<[^>]*>/g, '');
    toggleSpeech(plainText, speakBtn);
  };
  
  actionRow.appendChild(timestamp);
  actionRow.appendChild(copyBtn);
  actionRow.appendChild(speakBtn);
  bubble.appendChild(actionRow);
  
  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addTypingBubble() {
  const wrapper = document.createElement("div");
  wrapper.className = "text-left";
  wrapper.id = "typingIndicator";
  
  const bubble = document.createElement("div");
  bubble.className = "inline-block bot-message animate-fade-in";
  
  const loading = document.createElement("div");
  loading.className = "typing-loader mt-1";
  loading.innerHTML = "<span></span><span></span><span></span>";
  
  bubble.appendChild(loading);
  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
  
  return wrapper;
}

function removeTypingBubble() {
  const typingIndicator = document.getElementById("typingIndicator");
  if (typingIndicator) {
    typingIndicator.remove();
  }
}

// ===== New Chat Button =====
const newChatBtn = document.getElementById("newChat");
if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    currentSessionId = Date.now().toString();
    sessions[currentSessionId] = { messages: [], created: new Date().toISOString() };
    localStorage.setItem("chatSessions", JSON.stringify(sessions));
    localStorage.setItem("currentSessionId", currentSessionId);
    
    chatBox.innerHTML = "";
    addMessage("Hi there! 👋 How can I support you today?");
    renderHistory();
    
    // Reset file state
    uploadedFilePaths = [];
    activeDocumentId = null;
    fileInput.value = "";
    uploadStatus.innerHTML = "";
    fileList.innerHTML = "";
    focusUserInput();
  });
}

// ===== File Upload Handler =====
if (uploadForm) {
  uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const files = fileInput.files;
    if (!files.length) {
      alert("Please select a file to upload.");
      return;
    }
    
    uploadStatus.innerHTML = `
      <div class="upload-progress" role="progressbar" aria-label="Uploading files" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <span>0%</span>
      </div>
      <span class="upload-progress-label">Uploading…</span>`;
    const formData = new FormData();
    for (const f of files) formData.append("file", f);
    
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);
    
    const submitBtn = chatForm.querySelector("button[type='submit']");
    const uploadBtn = uploadForm.querySelector("button[type='submit']");
    if (submitBtn) submitBtn.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;
    userInput.disabled = true;
    
    const progressBar = uploadStatus.querySelector(".upload-progress");
    const progressText = progressBar.querySelector("span");
    
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        progressBar.style.setProperty("--upload-progress", `${percent * 3.6}deg`);
        progressBar.setAttribute("aria-valuenow", String(percent));
        progressText.textContent = `${percent}%`;
      }
    };

    xhr.upload.onload = () => {
      progressBar.style.setProperty("--upload-progress", "360deg");
      progressBar.setAttribute("aria-valuenow", "100");
      progressText.textContent = "…";
      const label = uploadStatus.querySelector(".upload-progress-label");
      if (label) label.textContent = "Processing document and preparing RAG…";
    };
    
    const setControlsDisabled = (disabled) => {
      if (submitBtn) submitBtn.disabled = disabled;
      if (uploadBtn) uploadBtn.disabled = disabled;
      userInput.disabled = disabled;
      if (!disabled) focusUserInput();
    };

    const showFailure = (message) => {
      setControlsDisabled(false);
      uploadStatus.innerHTML = `<p style='font-size: 14px; color: #dc2626;'>❌ ${escapeHtml(message)}</p>`;
    };

    xhr.onload = function () {
      if (xhr.status === 202) {
        let result = {};
        try {
          result = JSON.parse(xhr.responseText || "{}");
        } catch (error) {
          showFailure("Invalid upload response from server.");
          return;
        }

        const pendingPaths = result.filepaths || [];
        const pendingFiles = result.files || [];
        const pendingDocumentId = result.document_id || "";
        if (!pendingDocumentId) {
          showFailure("The server did not return an indexing job ID.");
          return;
        }

        const pollStatus = async () => {
          try {
            const response = await fetch(
              `/documents/${encodeURIComponent(pendingDocumentId)}/status`,
              { cache: "no-store" }
            );
            const statusResult = await response.json();
            if (!response.ok || !statusResult.success) {
              throw new Error(statusResult.error || "Unable to read indexing status");
            }

            const percent = Math.max(0, Math.min(100, Number(statusResult.progress) || 0));
            progressBar.style.setProperty("--upload-progress", `${percent * 3.6}deg`);
            progressBar.setAttribute("aria-valuenow", String(percent));
            progressText.textContent = `${percent}%`;
            const label = uploadStatus.querySelector(".upload-progress-label");
            if (label) label.textContent = `${statusResult.status} document…`;

            if (statusResult.status === "Indexed") {
              uploadedFilePaths = pendingPaths;
              activeDocumentId = pendingDocumentId;
              setControlsDisabled(false);
              fileInput.value = "";
              uploadStatus.innerHTML = `<div class="upload-ready"><span class="upload-complete-ring" aria-hidden="true">✓</span><span><strong>Upload and indexing complete.</strong><br>${pendingPaths.length} file(s) are ready for fast retrieval.</span></div>`;
              const activeFiles = pendingFiles.length
                ? pendingFiles
                : pendingPaths.map((path) => ({ name: path, url: "" }));
              fileList.innerHTML = activeFiles.map((file) => {
                const fullName = String(file.original_name || file.name || "Document")
                  .split(/[\\/]/)
                  .pop();
                const shortName = compactFileName(fullName);
                const label = `<span aria-hidden="true">📎</span><span><strong>Current:</strong> ${escapeHtml(shortName)}</span>`;
                const content = file.url
                  ? `<a href="${escapeHtml(file.url)}" target="_blank" rel="noopener noreferrer" aria-label="Open ${escapeHtml(fullName)}">${label}</a>`
                  : label;
                return `<li class="active-document-item" title="Open ${escapeHtml(fullName)}">${content}</li>`;
              }).join("");
              return;
            }
            if (statusResult.status === "Failed") {
              showFailure(statusResult.error || "Document indexing failed.");
              return;
            }
            window.setTimeout(pollStatus, 500);
          } catch (error) {
            showFailure(error.message || "Indexing status check failed.");
          }
        };
        pollStatus();
      } else {
        let errorMessage = `Upload failed (${xhr.status}).`;
        try {
          const errorResult = JSON.parse(xhr.responseText || "{}");
          if (errorResult.error) errorMessage = errorResult.error;
        } catch (error) {
          console.error("Failed to parse upload error:", error);
        }
        showFailure(errorMessage);
      }
    };

    xhr.onerror = () => {
      if (submitBtn) submitBtn.disabled = false;
      if (uploadBtn) uploadBtn.disabled = false;
      userInput.disabled = false;
      uploadStatus.innerHTML = "<p style='font-size: 14px; color: #dc2626;'>❌ Network error during upload.</p>";
    };
    
    xhr.send(formData);
  });
}

// ===== Buffered response streaming =====

/**
 * Creates a live bot bubble that response chunks are buffered into.
 * Returns an object with:
 *   - wrapper  : the outer div appended to chatBox
 *   - contentEl: the div where raw text is being built
 *   - cursor   : the animated cursor span
 *   - finalize(fullText, sources, emotion, hasFiles) — call when stream ends
 */
function responseTypeLabel(toolUsed, ragUsed) {
  const labels = {
    calculator: "Calculated answer",
    web_search: "Web-assisted answer",
    "llm+web": "Web-assisted answer",
    docs: "Document answer",
    "docs+web": "Document + web answer",
    cache: "Saved answer",
    crisis: "Priority support",
  };
  return labels[toolUsed] || (ragUsed ? "Document answer" : "AI answer");
}

function createStreamingBubble(toolUsed, ragUsed) {
  const wrapper = document.createElement("div");
  wrapper.className = "text-left streaming-wrapper";

  const bubble = document.createElement("div");
  bubble.className = "inline-block bot-message animate-fade-in streaming-active";

  const contentEl = document.createElement("div");
  contentEl.className = "message-content stream-content";

  const responseLabel = document.createElement("div");
  responseLabel.className = "response-label";
  responseLabel.textContent = responseTypeLabel(toolUsed, ragUsed);

  const cursor = document.createElement("span");
  cursor.className = "stream-cursor";
  cursor.setAttribute("aria-hidden", "true");

  bubble.appendChild(responseLabel);
  bubble.appendChild(contentEl);
  bubble.appendChild(cursor);
  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;

  // Native LLM chunks and completed RAG answers share one fast token renderer.
  // Large completed answers use small bursts so they remain quick and progressive.
  let _plain = "";
  let _displayed = "";
  let tokenQueue = [];
  let pumpTimer = null;
  let drainResolvers = [];

  function renderBufferedText() {
    contentEl.innerHTML = formatBotResponse(_displayed);
    bubble.appendChild(cursor);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function resolveDrain() {
    if (tokenQueue.length || pumpTimer) return;
    const resolvers = drainResolvers;
    drainResolvers = [];
    resolvers.forEach((resolve) => resolve());
  }

  function pumpTokens() {
    pumpTimer = null;
    const burstSize = tokenQueue.length > 120 ? 8 : tokenQueue.length > 40 ? 4 : 1;
    for (let index = 0; index < burstSize && tokenQueue.length; index += 1) {
      _displayed += tokenQueue.shift();
    }
    renderBufferedText();
    if (tokenQueue.length) {
      pumpTimer = window.setTimeout(pumpTokens, 18);
    } else {
      resolveDrain();
    }
  }

  function appendChunk(text) {
    if (!text) return;
    _plain += text;
    tokenQueue.push(...(text.match(/\s+|[^\s]+\s*/g) || [text]));
    if (!pumpTimer) pumpTimer = window.setTimeout(pumpTokens, 0);
  }

  function waitForDrain() {
    if (!tokenQueue.length && !pumpTimer) return Promise.resolve();
    return new Promise((resolve) => drainResolvers.push(resolve));
  }

  async function finalize(sources, emotion, hasFiles) {
    await waitForDrain();
    _displayed = _plain;
    renderBufferedText();

    // Remove cursor
    cursor.remove();

    // Remove streaming highlight
    bubble.classList.remove("streaming-active");

    // Emotion badge (injected above the bubble)
    if (emotion) {
      const badge = createEmotionBadge(emotion);
      if (badge) {
        wrapper.insertBefore(badge, bubble);
        showEmotionBadge(emotion);
      }
    }

    // File indicator
    if (hasFiles) {
      const fileInd = document.createElement("div");
      fileInd.className = "file-indicator";
      fileInd.innerHTML = "🔍 Response based on uploaded files";
      bubble.appendChild(fileInd);
    }

    // Footer (timestamp + copy + speak)
    const footer = document.createElement("div");
    footer.className = "message-footer";

    const ts = document.createElement("span");
    ts.className = "timestamp";
    ts.textContent = addTimestamp();

    const copyBtn = document.createElement("button");
    copyBtn.className = "icon-btn";
    copyBtn.innerHTML = "📋";
    copyBtn.title = "Copy message";
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(_plain).then(() => {
        copyBtn.innerHTML = "✅";
        setTimeout(() => (copyBtn.innerHTML = "📋"), 1000);
      });
    };

    const speakBtn = document.createElement("button");
    speakBtn.className = "icon-btn";
    speakBtn.innerHTML = "🔊";
    speakBtn.title = "Read aloud";
    speakBtn.onclick = () => toggleSpeech(_plain, speakBtn);

    footer.appendChild(ts);
    footer.appendChild(copyBtn);
    footer.appendChild(speakBtn);
    bubble.appendChild(footer);

    chatBox.scrollTop = chatBox.scrollHeight;
    return _plain;
  }

  return { wrapper, contentEl, cursor, appendChunk, finalize };
}

/**
 * Opens an EventSource to /search/stream and drives the streaming bubble.
 */
function startStream(query, shouldAttachFiles) {
  isSearching = true;
  setInputDisabled(true);

  // Show typing dots while waiting for first token
  const typingBubble = addTypingBubble();
  let firstToken = true;
  let streamBubble = null;
  let metaEmotion = null;
  let metaRagUsed = false;
  let metaToolUsed = "llm";
  let finished = false;

  // Build query string
  const params = new URLSearchParams({
    q:            query,
    session_name: currentSessionId,
  });
  if (activeDocumentId) params.set("document_id", activeDocumentId);

  const es = new EventSource(`/search/stream?${params.toString()}`);

  // Close + cleanup helper
  function cleanup() {
    if (finished) return;
    finished = true;
    es.close();
    isSearching = false;
    setInputDisabled(false);
  }

  es.onmessage = async (event) => {
    let frame;
    try {
      frame = JSON.parse(event.data);
    } catch {
      return;
    }

    if (frame.type === "meta") {
      metaEmotion = frame.emotion_detected || null;
      metaRagUsed = frame.rag_used || false;
      metaToolUsed = frame.tool_used || "llm";
      return;
    }

    if (frame.type === "chunk" || frame.type === "token") {
      // First content — swap typing dots for the buffered response bubble.
      if (firstToken) {
        removeTypingBubble();
        streamBubble = createStreamingBubble(metaToolUsed, metaRagUsed);
        firstToken = false;
      }
      streamBubble.appendChunk(frame.text || "");
      return;
    }

    if (frame.type === "done") {
      es.close();
      removeTypingBubble(); // safety — in case no tokens came through

      if (streamBubble) {
        const finalText = await streamBubble.finalize(
          frame.sources || [],
          metaEmotion,
          shouldAttachFiles && metaRagUsed
        );
        saveMessage(currentSessionId, "bot", finalText, metaEmotion);
      } else {
        addMessage("⚠️ No response was received. Please try again.", false);
      }

      cleanup();
      return;
    }

    if (frame.type === "error") {
      es.close();
      removeTypingBubble();
      if (streamBubble) {
        const partialText = await streamBubble.finalize([], metaEmotion, false);
        if (partialText) saveMessage(currentSessionId, "bot", partialText, metaEmotion);
        addMessage("⚠️ The response was interrupted. Please try again.", false);
      } else {
        addMessage(`❌ ${frame.message || "An error occurred"}`, false);
      }
      cleanup();
    }
  };

  es.onerror = async () => {
    if (finished) return;
    removeTypingBubble();
    if (!streamBubble) {
      addMessage("❌ Connection lost. Please try again.", false);
    } else {
      const partialText = await streamBubble.finalize([], metaEmotion, false);
      if (partialText) saveMessage(currentSessionId, "bot", partialText, metaEmotion);
      addMessage("⚠️ Connection lost after a partial response. Please try again.", false);
    }
    cleanup();
  };
}

// Tiny helper — enable/disable the send button + input together
function setInputDisabled(disabled) {
  const submitBtn = chatForm ? chatForm.querySelector("button[type='submit']") : null;
  if (submitBtn) submitBtn.disabled = disabled;
  if (userInput)  userInput.disabled  = disabled;
  if (!disabled) focusUserInput();
}

// ===== Chat Form Submit =====
if (chatForm) {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();

    if (isSearching) return;

    const query = userInput.value.trim();
    if (!query) return;

    addMessage(query, true);
    saveMessage(currentSessionId, "user", query);
    userInput.value = "";

    const shouldAttachFiles = Boolean(activeDocumentId);
    startStream(query, shouldAttachFiles);
  });
}

// ===== Sidebar Toggle =====
const toggleButtons = document.querySelectorAll("#toggleHistory, #toggleHistoryBtn");
toggleButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    if (historySidebar.classList.contains("hidden")) {
      historySidebar.classList.remove("hidden");
      chatMain.classList.add("with-sidebar");
    } else {
      historySidebar.classList.add("hidden");
      chatMain.classList.remove("with-sidebar");
    }
  });
});

// ===== Speech Recognition =====
let recognition;
let isListening = false;

if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = "en-US";   
  recognition.continuous = false;
  recognition.interimResults = false;

  const micIcon = document.getElementById("micIcon");

  if (micBtn) {
    micBtn.addEventListener("click", () => {
      if (!isListening) {
        try {
          recognition.start();
          if (micIcon) {
            micIcon.src = "https://img.icons8.com/fluency/24/stop-squared.png";
          }
          micBtn.style.background = "#dc2626";
          isListening = true;
        } catch (e) {
          console.error("Error starting recognition:", e);
        }
      } else {
        recognition.stop();
        if (micIcon) {
          micIcon.src = "https://img.icons8.com/material-sharp/24/microphone--v1.png";
        }
        micBtn.style.background = "transparent";
        isListening = false;
      }
    });
  }

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
  };

  recognition.onend = () => {
    const micIcon = document.getElementById("micIcon");
    if (micIcon) {
      micIcon.src = "https://img.icons8.com/material-sharp/24/microphone--v1.png";
    }
    if (micBtn) {
      micBtn.style.background = "transparent";
    }
    isListening = false;
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    const micIcon = document.getElementById("micIcon");
    if (micIcon) {
      micIcon.src = "https://img.icons8.com/material-sharp/24/microphone--v1.png";
    }
    if (micBtn) {
      micBtn.style.background = "transparent";
    }
    isListening = false;
  };
} else {
  console.warn("Speech Recognition not supported in this browser.");
  if (micBtn) micBtn.style.display = "none";
}

// ===== Global handlers for inline onclick in static HTML messages =====
// These are referenced by onclick="copyMessage(this)" and onclick="speakMessage(this)"
// in the initial bot message rendered directly in index.html.
window.copyMessage = function(btn) {
  const bubble = btn.closest('.bot-message, .user-message');
  const contentEl = bubble ? bubble.querySelector('.message-content') : null;
  const text = contentEl ? contentEl.innerText : (bubble ? bubble.innerText : '');
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = '✅';
    setTimeout(() => (btn.innerHTML = '📋'), 1000);
  }).catch(err => console.error('Copy failed:', err));
};

window.speakMessage = function(btn) {
  const bubble = btn.closest('.bot-message, .user-message');
  const contentEl = bubble ? bubble.querySelector('.message-content') : null;
  const text = contentEl ? contentEl.innerText : (bubble ? bubble.innerText : '');
  toggleSpeech(text, btn);
};

// ===== Page Load Initialization =====
window.addEventListener("DOMContentLoaded", () => {
  const seenDisclaimer = localStorage.getItem("seenDisclaimer");
  if (!seenDisclaimer) {
    localStorage.setItem("seenDisclaimer", "yes");
    window.location.href = "/disclaimer";
    return;
  }
  
  // Always open on a fresh conversation. Previous chats remain available
  // through the history sidebar and are only restored when the user selects one.
  currentSessionId = Date.now().toString();
  localStorage.setItem("currentSessionId", currentSessionId);
  uploadedFilePaths = [];
  activeDocumentId = null;
  chatBox.innerHTML = "";
  addMessage("Hi there! 👋 How can I support you today?");
  renderHistory();
  focusUserInput();
});
