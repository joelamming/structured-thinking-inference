/**
 * This is shitty Claude slop and will be replaced with as much HTMX as possible
 */

class ChatClient {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.processing = false;
        this.conversationHistory = [];  // Array of {role, content} messages
        
        // DOM elements
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('message-input');
        this.chatForm = document.getElementById('chat-form');
        this.sendBtn = document.getElementById('send-btn');
        this.effortSelect = document.getElementById('effort');
        this.formatSelect = document.getElementById('format');
        this.statusDiv = document.getElementById('status');
        this.clearBtn = document.getElementById('clear-btn');
        
        // Bind events
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.clearBtn.addEventListener('click', () => this.clearHistory());
        
        // Allow Ctrl/Cmd+Enter to submit
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.handleSubmit(e);
            }
        });
        
        // Load saved conversation from localStorage
        this.loadFromStorage();
        
        // Connect on load
        this.connect();
    }
    
    connect() {
        try {
            this.updateStatus('disconnected', 'Connecting...');
            // Connect to local frontend WebSocket (not orchestrator directly)
            const wsUrl = `ws://${window.location.host}/ws`;
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.connected = true;
                this.updateStatus('connected', 'Connected');
                console.log('WebSocket connected to frontend server');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    this.handleMessage(JSON.parse(event.data));
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('disconnected', 'Connection Error');
            };
            
            this.ws.onclose = (event) => {
                this.connected = false;
                this.processing = false;
                this.sendBtn.disabled = false;
                
                if (event.wasClean) {
                    console.log(`WebSocket closed cleanly, code=${event.code}`);
                    this.updateStatus('disconnected', 'Disconnected');
                } else {
                    console.log('WebSocket connection died');
                    this.updateStatus('disconnected', 'Connection Lost');
                }
                
                // Reconnect after 3 seconds
                setTimeout(() => {
                    if (!this.connected) {
                        console.log('Attempting to reconnect...');
                        this.connect();
                    }
                }, 3000);
            };
        } catch (error) {
            console.error('Connection error:', error);
            this.updateStatus('disconnected', 'Failed to connect');
            setTimeout(() => this.connect(), 3000);
        }
    }
    
    handleMessage(data) {
        if (data.type === 'ping') {
            // Respond to keepalive
            this.ws.send(JSON.stringify({ type: 'pong' }));
            return;
        }
        
        if (data.status === 'completed') {
            this.processing = false;
            this.updateStatus('connected', 'Connected');
            this.sendBtn.disabled = false;
            this.updateButtonState(false);
            
            const result = data.result;
            const choice = result.choices[0];
            
            // Debug: Log the full response
            console.log('Full response:', data);
            console.log('Choice:', choice);
            console.log('Thinking:', choice.thinking);
            
            // Content is already decrypted by frontend server
            let content = choice.message.content;
            const thinking = choice.thinking || null;
            
            // If it's JSON-wrapped text, extract the response field
            if (this.formatSelect.value === 'text') {
                try {
                    const parsed = JSON.parse(content);
                    if (parsed.response) {
                        content = parsed.response;
                    }
                } catch (e) {
                    // If parsing fails, use content as-is
                    console.warn('Failed to parse JSON-wrapped response:', e);
                }
            }
            
            // Add to conversation history
            this.conversationHistory.push({
                role: 'assistant',
                content: content
            });
            this.saveToStorage();
            
            this.addAssistantMessage(content, thinking);
            
        } else if (data.status === 'error' || data.status === 'timeout') {
            this.processing = false;
            this.updateStatus('connected', 'Connected');
            this.sendBtn.disabled = false;
            this.updateButtonState(false);
            
            this.addErrorMessage(data.error || 'Request failed');
        }
    }
    
    handleSubmit(e) {
        e.preventDefault();
        
        const message = this.messageInput.value.trim();
        if (!message || !this.connected || this.processing) {
            return;
        }
        
        // Add user message to UI
        this.addUserMessage(message);
        
        // Add to conversation history
        this.conversationHistory.push({
            role: 'user',
            content: message
        });
        this.saveToStorage();
        
        // Send conversation history - frontend server handles encryption
        const effort = this.effortSelect.value;
        const format = this.formatSelect.value;
        
        // Build messages array with full conversation context
        const messages = this.conversationHistory.map(msg => ({
            role: msg.role,
            content: msg.content
        }));
        
        // Build request based on format
        const requestData = {
            model: window.CONFIG.modelName,
            messages: messages,  // Send full conversation history
            thinking_effort: effort
        };
        
        if (format === 'text') {
            // Wrap text in minimal JSON structure for guided decoding
            // Send schema in extra_body.structured_outputs (vLLM format)
            requestData.extra_body = {
                structured_outputs: {
                    json: {
                        type: "object",
                        properties: {
                            response: {
                                type: "string"
                            }
                        },
                        required: ["response"]
                    }
                }
            };
        } else {
            // JSON object (free-form JSON)
            requestData.response_format = { type: "json_object" };
        }
        
        const request = {
            request: requestData,
            client_request_id: Date.now().toString(),
            timeout_seconds: this.getTimeout(effort)
        };
        
        this.ws.send(JSON.stringify(request));
        
        // Update UI
        this.messageInput.value = '';
        this.processing = true;
        this.sendBtn.disabled = true;
        this.updateButtonState(true);
        this.updateStatus('processing', 'Thinking...');
    }
    
    getTimeout(effort) {
        // Dynamic timeout based on effort level
        const timeouts = {
            'none': 30,
            'low': 60,
            'medium': 180,
            'high': 600  // 10 minutes for high effort
        };
        return timeouts[effort] || 180;
    }
    
    addUserMessage(content, shouldScroll = true) {
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `
            <div class="message-label">You</div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;
        this.messagesContainer.appendChild(div);
        if (shouldScroll) {
            this.scrollToBottom();
        }
    }
    
    addAssistantMessage(content, thinking) {
        const div = document.createElement('div');
        div.className = 'message assistant';
        
        // Format content (preserve newlines, detect code blocks)
        const formattedContent = this.formatContent(content);
        
        let html = `
            <div class="message-label">Assistant</div>
            <div class="message-content">${formattedContent}</div>
        `;
        
        if (thinking) {
            const thinkingId = 'thinking-' + Date.now();
            html += `
                <div class="thinking">
                    <div class="thinking-header">
                        <span class="thinking-label">Thinking Process</span>
                        <button class="thinking-toggle" onclick="chatClient.toggleThinking('${thinkingId}')">
                            Expand ▼
                        </button>
                    </div>
                    <div class="thinking-content collapsed" id="${thinkingId}">
                        ${this.escapeHtml(thinking)}
                    </div>
                </div>
            `;
        }
        
        div.innerHTML = html;
        this.messagesContainer.appendChild(div);
        this.scrollToBottom();
    }
    
    toggleThinking(id) {
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
    
    formatContent(text) {
        // Simple markdown-like formatting
        let formatted = this.escapeHtml(text);
        
        // Convert ```code blocks``` to <pre><code>
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        
        // Convert `inline code` to <code>
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert newlines to <br>
        formatted = formatted.replace(/\n/g, '<br>');
        
        return formatted;
    }
    
    addErrorMessage(error) {
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.innerHTML = `
            <div class="message-label">Error</div>
            <div class="message-content error">${this.escapeHtml(error)}</div>
        `;
        this.messagesContainer.appendChild(div);
        this.scrollToBottom();
    }
    
    updateStatus(className, text) {
        this.statusDiv.className = `status ${className}`;
        this.statusDiv.textContent = text;
    }
    
    updateButtonState(processing) {
        const btnText = this.sendBtn.querySelector('.btn-text');
        const btnSpinner = this.sendBtn.querySelector('.btn-spinner');
        
        if (processing) {
            btnText.style.display = 'none';
            btnSpinner.style.display = 'inline';
        } else {
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }
    
    clearHistory() {
        if (!confirm('Clear all conversation history? This cannot be undone.')) {
            return;
        }
        
        this.conversationHistory = [];
        localStorage.removeItem('chatHistory');
        
        // Clear UI except welcome message
        const welcomeMessage = this.messagesContainer.firstElementChild;
        this.messagesContainer.innerHTML = '';
        if (welcomeMessage) {
            this.messagesContainer.appendChild(welcomeMessage);
        }
    }
    
    saveToStorage() {
        try {
            localStorage.setItem('chatHistory', JSON.stringify(this.conversationHistory));
        } catch (e) {
            console.warn('Failed to save to localStorage:', e);
        }
    }
    
    loadFromStorage() {
        try {
            const saved = localStorage.getItem('chatHistory');
            if (saved) {
                this.conversationHistory = JSON.parse(saved);
                
                // Restore messages to UI
                this.conversationHistory.forEach((msg, idx) => {
                    if (msg.role === 'user') {
                        this.addUserMessage(msg.content, false);  // Don't scroll for each
                    } else if (msg.role === 'assistant') {
                        // Note: We don't have thinking saved, only content
                        this.addAssistantMessage(msg.content, null);
                    }
                });
                
                if (this.conversationHistory.length > 0) {
                    this.scrollToBottom();
                }
            }
        } catch (e) {
            console.warn('Failed to load from localStorage:', e);
            localStorage.removeItem('chatHistory');  // Clear corrupted data
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.parentElement.scrollTop = 
            this.messagesContainer.parentElement.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
}

// Initialize client when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatClient = new ChatClient();
});

