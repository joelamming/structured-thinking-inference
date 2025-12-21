/**
 * Minimal JS helpers for HTMX chat interface
 * HTMX handles: WebSocket connection, form submission, HTML swapping
 * JS handles: Keyboard shortcuts, thinking toggle, status updates, clear button
 */

// HTMX WebSocket event handlers
document.addEventListener('htmx:wsOpen', function() {
    updateStatus('connected', 'Connected');
});

document.addEventListener('htmx:wsClose', function() {
    updateStatus('disconnected', 'Disconnected');
});

document.addEventListener('htmx:wsError', function() {
    updateStatus('disconnected', 'Connection Error');
});

// Auto-scroll when messages arrive
document.addEventListener('htmx:wsAfterMessage', function() {
    scrollToBottom();
});

// Intercept form submission to add effort/format to payload
document.addEventListener('htmx:wsConfigSend', function(event) {
    const effort = document.getElementById('effort').value;
    const format = document.getElementById('format').value;
    
    // Add to parameters that HTMX will serialize
    event.detail.parameters.effort = effort;
    event.detail.parameters.format = format;
    
    // Show processing state
    updateButtonState(true);
});

// Reset button state after send
document.addEventListener('htmx:wsAfterSend', function() {
    const messageInput = document.getElementById('message-input');
    messageInput.value = '';
    
    // Reset button after short delay
    setTimeout(() => updateButtonState(false), 1000);
});

// Keyboard shortcut: Ctrl/Cmd+Enter to submit
document.getElementById('message-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        const form = document.getElementById('chat-form');
        // Trigger HTMX form submission
        htmx.trigger(form, 'submit');
    }
});

// Clear chat history
document.getElementById('clear-btn').addEventListener('click', async function() {
    if (!confirm('Clear all conversation history? This cannot be undone.')) {
        return;
    }
    
    // Clear UI
    document.getElementById('messages').innerHTML = '';
    
    // Clear server-side session
    await fetch('/clear', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `session_id=${SESSION_ID}`
    });
});

// Toggle thinking sections (called from template onclick)
function toggleThinking(id) {
    const element = document.getElementById(id);
    const button = element.previousElementSibling.querySelector('.thinking-toggle');
    
    if (element.classList.contains('collapsed')) {
        element.classList.remove('collapsed');
        button.textContent = 'Collapse ▲';
    } else {
        element.classList.add('collapsed');
        button.textContent = 'Expand ▼';
    }
}

// UI helper functions
function updateStatus(className, text) {
    const statusDiv = document.getElementById('status');
    statusDiv.className = `status ${className}`;
    statusDiv.textContent = text;
}

function updateButtonState(processing) {
    const sendBtn = document.getElementById('send-btn');
    const btnText = sendBtn.querySelector('.btn-text');
    const btnSpinner = sendBtn.querySelector('.btn-spinner');
    
    sendBtn.disabled = processing;
    btnText.style.display = processing ? 'none' : 'inline';
    btnSpinner.style.display = processing ? 'inline' : 'none';
}

function scrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Expose to global scope for template onclick handlers
window.toggleThinking = toggleThinking;
