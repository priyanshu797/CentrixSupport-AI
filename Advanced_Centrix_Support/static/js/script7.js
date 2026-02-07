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
let fileUsedOnce = false;
let pendingSearchAbort = null;
let isSearching = false;
let currentSessionId = localStorage.getItem("currentSessionId") || Date.now().toString();
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
}

function formatBotResponse(text) {
  let formatted = text;
  
  // Convert markdown bold
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Convert line breaks to paragraphs
  const paragraphs = formatted.split('\n\n').filter(p => p.trim());
  if (paragraphs.length > 1) {
    formatted = paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
  } else {
    formatted = formatted.replace(/\n/g, '<br>');
  }
  
  // Convert lists
  formatted = formatted.replace(/^- (.+)$/gm, '<li>$1</li>');
  formatted = formatted.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');
  
  return formatted;
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

function markFilesConsumed() {
  fileUsedOnce = true;
  const tempPaths = [...uploadedFilePaths];
  uploadedFilePaths = [];
  fileInput.value = "";
  uploadStatus.innerHTML = `<p style='font-size: 14px; color: #854d0e;'>🔍 Files used in last response. Upload again if needed.</p>`;
  fileList.innerHTML = "";
  return tempPaths;
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
    fileUsedOnce = false;
    uploadStatus.innerHTML = "";
    fileList.innerHTML = "";
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
    
    uploadStatus.innerHTML = "<p style='font-size: 14px; color: #0f766e;'>⏳ Uploading...</p>";
    fileUsedOnce = false;
    uploadedFilePaths = [];
    
    const formData = new FormData();
    for (const f of files) formData.append("file", f);
    
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);
    
    const submitBtn = chatForm.querySelector("button[type='submit']");
    if (submitBtn) submitBtn.disabled = true;
    userInput.disabled = true;
    
    const progressBar = document.createElement("div");
    progressBar.style.cssText = "background: #0f766e; height: 10px; border-radius: 999px; width: 0; transition: width 0.3s;";
    
    const container = document.createElement("div");
    container.style.cssText = "width: 100%; background: #e5e7eb; border-radius: 999px; margin-top: 8px;";
    container.appendChild(progressBar);
    uploadStatus.appendChild(container);
    
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = percent + "%";
      }
    };
    
    xhr.onload = function () {
      if (submitBtn) submitBtn.disabled = false;
      userInput.disabled = false;
      
      if (xhr.status === 200) {
        let result = {};
        try { 
          result = JSON.parse(xhr.responseText || "{}"); 
        } catch (e) {
          console.error("Failed to parse upload response:", e);
        }
        
        uploadedFilePaths = result.filepaths || [];
        
        if (uploadedFilePaths.length) {
          uploadStatus.innerHTML = `<p style='font-size: 14px; color: #15803d; font-weight: 600;'>✅ ${uploadedFilePaths.length} file(s) uploaded successfully.</p><p style='font-size: 14px; color: #0f766e;'>💬 Your next question will use these files.</p>`;
          
          fileList.innerHTML = uploadedFilePaths.map(p => {
            const name = p.split(/[\\/]/).pop();
            return `<li style="display: flex; align-items: center; gap: 8px; padding: 4px 0;"><span style="font-size: 18px;">📎</span><span>${name}</span></li>`;
          }).join("");
        } else {
          uploadStatus.innerHTML = `<p style='font-size: 14px; color: #dc2626;'>⚠️ Upload succeeded but no file paths returned.</p>`;
        }
      } else {
        uploadStatus.innerHTML = `<p style='font-size: 14px; color: #dc2626;'>❌ Upload failed (${xhr.status}).</p>`;
      }
    };
    
    xhr.onerror = () => {
      if (submitBtn) submitBtn.disabled = false;
      userInput.disabled = false;
      uploadStatus.innerHTML = "<p style='font-size: 14px; color: #dc2626;'>❌ Network error during upload.</p>";
    };
    
    xhr.send(formData);
  });
}

// ===== Chat Form Submit =====
if (chatForm) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    if (isSearching) return;
    
    const query = userInput.value.trim();
    if (!query) return;
    
    // Add user message
    addMessage(query, true);
    saveMessage(currentSessionId, "user", query);
    userInput.value = "";
    
    // Show typing indicator
    const typingBubble = addTypingBubble();
    
    // Determine file attachment
    const shouldAttachFiles = uploadedFilePaths.length > 0 && !fileUsedOnce;
    const filesToSend = shouldAttachFiles ? [...uploadedFilePaths] : [];
    
    const payload = { 
      query, 
      filepaths: filesToSend,
      session_name: currentSessionId 
    };
    
    try { 
      pendingSearchAbort?.abort(); 
    } catch (e) {
      console.error("Error aborting previous request:", e);
    }
    
    pendingSearchAbort = new AbortController();
    isSearching = true;
    
    try {
      const res = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: pendingSearchAbort.signal  //cancelling the request if needed (like user stops typing)
      });
      
      removeTypingBubble();
      
      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        const responseText = (data.response || "").trim() || "⚠️ No response received.";
        const emotion = data.emotion_detected || null;
        
        addMessage(responseText, false, null, emotion, shouldAttachFiles);
        saveMessage(currentSessionId, "bot", responseText, emotion);
        
        if (shouldAttachFiles) {
          markFilesConsumed();
        }
      } else {
        const errorMsg = data.error || data.response || "An error occurred";
        addMessage(`❌ Error: ${errorMsg}`, false);
      }
      
    } catch (err) {
      removeTypingBubble();
      
      if (err.name === 'AbortError') {
        console.log('Request was aborted');
      } else {
        const errorMsg = `❌ Failed to get response from server.\n${err.message}`;
        addMessage(errorMsg, false);
        console.error("Chat error:", err);
      }
    } finally {
      isSearching = false;
    }
  });
}

// ===== Sidebar Toggle =====
const toggleButtons = document.querySelectorAll("#toggleHistory");
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

// ===== Page Load Initialization =====
window.addEventListener("DOMContentLoaded", () => {
  const seenDisclaimer = localStorage.getItem("seenDisclaimer");
  if (!seenDisclaimer) {
    localStorage.setItem("seenDisclaimer", "yes");
    window.location.href = "/disclaimer";
    return;
  }
  
  renderHistory();
  
  if (!sessions[currentSessionId]) {
    sessions[currentSessionId] = { messages: [], created: new Date().toISOString() };
    localStorage.setItem("chatSessions", JSON.stringify(sessions));
  }
  
  loadSession(currentSessionId);
});