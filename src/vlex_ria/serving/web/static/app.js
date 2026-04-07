class ChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chat-messages');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.cancelBtn = document.getElementById('cancel-btn');
        this.modelSelect = document.getElementById('model-select');
        this.modelLoading = document.getElementById('model-loading');
        this.modelInfo = document.getElementById('model-info');
        this.createConvBtn = document.getElementById('create-conv');
        this.renameConvBtn = document.getElementById('rename-conv');
        this.deleteConvBtn = document.getElementById('delete-conv');
        this.conversationList = document.getElementById('conversation-list');
        this.errorToast = document.getElementById('error-toast');
        this.statusText = document.getElementById('status-text');
        this.promptChips = document.querySelectorAll('.prompt-chip');
        
        this.temperatureInput = document.getElementById('temperature');
        this.maxTokensInput = document.getElementById('max-tokens');
        this.topPInput = document.getElementById('top-p');
        this.topKInput = document.getElementById('top-k');
        
        this.isGenerating = false;
        this.currentModel = null;
        this.conversationHistory = [];
        this.conversations = [];
        this.activeConversationId = null;
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        this.setupSettingsListeners();
        this.initConversations();
        await this.loadCheckpoints();
    }
    
    setupEventListeners() {
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        if (this.userInput) {
            this.userInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }
        
        if (this.userInput) {
            this.userInput.addEventListener('input', () => {
                this.userInput.style.height = 'auto';
                this.userInput.style.height = Math.min(this.userInput.scrollHeight, 200) + 'px';
            });
        }
        
        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.cancelGeneration());
        }
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', () => this.loadSelectedModel());
        }
        if (this.createConvBtn) {
            this.createConvBtn.addEventListener('click', () => this.createConversationAndSwitch());
        }
        if (this.renameConvBtn) {
            this.renameConvBtn.addEventListener('click', () => this.renameConversation());
        }
        if (this.deleteConvBtn) {
            this.deleteConvBtn.addEventListener('click', () => this.deleteConversation());
        }
        this.promptChips.forEach((chip) => {
            chip.addEventListener('click', () => {
                if (this.isGenerating || this.userInput.disabled) {
                    return;
                }
                const prompt = chip.dataset.prompt || '';
                this.userInput.value = prompt;
                this.userInput.dispatchEvent(new Event('input'));
                this.userInput.focus();
            });
        });
    }
    
    setupSettingsListeners() {
        this.temperatureInput.addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });
        
        this.maxTokensInput.addEventListener('input', (e) => {
            document.getElementById('tokens-value').textContent = e.target.value;
        });
        
        this.topPInput.addEventListener('input', (e) => {
            document.getElementById('topp-value').textContent = e.target.value;
        });
        
        this.topKInput.addEventListener('input', (e) => {
            document.getElementById('topk-value').textContent = e.target.value;
        });
    }

    initConversations() {
        this.createConversationAndSwitch();
    }

    createConversationAndSwitch() {
        const id = Date.now().toString();
        const conversation = {
            id,
            title: `Hội thoại ${this.conversations.length + 1}`,
            messages: [],
            createdAt: Date.now(),
        };
        this.conversations.push(conversation);
        this.switchConversation(id);
    }

    switchConversation(conversationId) {
        if (this.activeConversationId) {
            this.syncActiveConversation();
        }
        this.activeConversationId = conversationId;
        const active = this.getActiveConversation();
        this.conversationHistory = active ? [...active.messages] : [];
        this.renderConversationList();
        this.renderCurrentConversation();
        if (active) {
            this.updateStatus(`Đang mở ${active.title}`);
        }
    }

    getActiveConversation() {
        return this.conversations.find((c) => c.id === this.activeConversationId) || null;
    }

    syncActiveConversation() {
        const active = this.getActiveConversation();
        if (active) {
            active.messages = [...this.conversationHistory];
        }
    }

    renderConversationList() {
        if (!this.conversationList) {
            return;
        }
        this.conversationList.innerHTML = '';
        this.conversations.forEach((conversation) => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = `conversation-item ${conversation.id === this.activeConversationId ? 'active' : ''}`;
            item.textContent = conversation.title;
            item.title = conversation.title;
            item.addEventListener('click', () => {
                if (this.isGenerating) {
                    return;
                }
                this.switchConversation(conversation.id);
            });
            this.conversationList.appendChild(item);
        });
    }

    renderCurrentConversation() {
        this.chatMessages.innerHTML = '';
        if (this.conversationHistory.length === 0) {
            this.chatMessages.innerHTML = `
                <div class="welcome-message">
                    <h2>Chào mừng bạn đến với VLex-RIA</h2>
                    <p>Hệ thống hội thoại hỗ trợ phân tích tác động chính sách theo RIA. Chọn checkpoint phù hợp và bắt đầu trao đổi chuyên sâu.</p>
                </div>
            `;
            return;
        }
        this.conversationHistory.forEach((message) => {
            this.addMessage(message.role, message.content, false);
        });
    }

    renameConversation() {
        if (this.isGenerating) {
            return;
        }
        const active = this.getActiveConversation();
        if (!active) {
            return;
        }
        const nextName = window.prompt('Nhập tên hội thoại mới:', active.title);
        if (!nextName) {
            return;
        }
        const trimmed = nextName.trim();
        if (!trimmed) {
            return;
        }
        active.title = trimmed;
        this.renderConversationList();
        this.updateStatus(`Đã đổi tên thành ${trimmed}`);
    }

    deleteConversation() {
        if (this.isGenerating) {
            return;
        }
        if (this.conversations.length === 0) {
            return;
        }
        const active = this.getActiveConversation();
        if (!active) {
            return;
        }
        const confirmed = window.confirm(`Bạn có chắc muốn xóa "${active.title}"?`);
        if (!confirmed) {
            return;
        }
        this.conversations = this.conversations.filter((c) => c.id !== active.id);
        if (this.conversations.length === 0) {
            this.createConversationAndSwitch();
            return;
        }
        this.switchConversation(this.conversations[0].id);
        this.updateStatus('Đã xóa hội thoại');
    }

    autoRenameFromFirstMessage() {
        const active = this.getActiveConversation();
        if (!active || active.messages.length !== 1) {
            return;
        }
        const content = active.messages[0].content || '';
        const normalized = content.replace(/\s+/g, ' ').trim();
        if (!normalized) {
            return;
        }
        active.title = normalized.slice(0, 36) + (normalized.length > 36 ? '...' : '');
        this.renderConversationList();
    }
    
    async loadCheckpoints() {
        try {
            const response = await fetch('/api/checkpoints');
            const data = await response.json();
            
            this.modelSelect.innerHTML = '';
            
            if (data.checkpoints.length === 0) {
                this.modelSelect.innerHTML = '<option value="">Không tìm thấy mô hình</option>';
                this.showError('Không tìm thấy file checkpoint trong thư mục checkpoints/');
                return;
            }
            
            const grouped = {};
            data.checkpoints.forEach(ckpt => {
                if (!grouped[ckpt.category]) {
                    grouped[ckpt.category] = [];
                }
                grouped[ckpt.category].push(ckpt);
            });
            
            for (const [category, checkpoints] of Object.entries(grouped)) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = category;
                
                checkpoints.forEach(ckpt => {
                    const option = document.createElement('option');
                    option.value = ckpt.path;
                    option.textContent = `${ckpt.name} (${ckpt.size_mb} MB)`;
                    if (ckpt.path === data.default || ckpt.path === data.current) {
                        option.selected = true;
                    }
                    optgroup.appendChild(option);
                });
                
                this.modelSelect.appendChild(optgroup);
            }
            
            this.modelSelect.disabled = false;
            this.currentModel = data.current || data.default;
            
            if (this.currentModel) {
                const ckpt = data.checkpoints.find(c => c.path === this.currentModel);
                if (ckpt) {
                    this.modelInfo.textContent = `Đã tải: ${ckpt.name}`;
                    this.updateStatus(`Đang dùng checkpoint ${ckpt.name}`);
                }
            }
            
            this.userInput.disabled = false;
            this.sendBtn.disabled = false;
            
        } catch (error) {
            console.error('Failed to load checkpoints:', error);
            this.showError('Không tải được danh sách mô hình');
        }
    }
    
    async loadSelectedModel() {
        const checkpointPath = this.modelSelect.value;
        if (!checkpointPath || checkpointPath === this.currentModel) {
            return;
        }
        
        if (this.isGenerating) {
            await this.cancelGeneration();
        }
        
        this.modelSelect.disabled = true;
        this.modelLoading.style.display = 'block';
        this.modelInfo.textContent = 'Đang tải mô hình...';
        this.updateStatus('Đang chuyển đổi checkpoint');
        
        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checkpoint_path: checkpointPath })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentModel = checkpointPath;
                const name = checkpointPath.split('/').slice(-2).join('/');
                this.modelInfo.textContent = `Đã tải: ${name}`;
                this.updateStatus(`Sẵn sàng hội thoại với ${name}`);
            } else {
                this.showError(data.error || 'Không tải được mô hình');
                this.modelInfo.textContent = 'Tải mô hình thất bại';
                this.updateStatus('Lỗi khi tải checkpoint');
            }
        } catch (error) {
            console.error('Failed to load model:', error);
            this.showError('Không tải được mô hình');
            this.modelInfo.textContent = 'Tải mô hình thất bại';
            this.updateStatus('Lỗi khi tải checkpoint');
        } finally {
            this.modelSelect.disabled = false;
            this.modelLoading.style.display = 'none';
        }
    }
    
    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message || this.isGenerating) {
            return;
        }
        
        const welcome = this.chatMessages.querySelector('.welcome-message');
        if (welcome) {
            welcome.remove();
        }
        
        this.addMessage('user', message);
        this.conversationHistory.push({ role: 'user', content: message });
        this.syncActiveConversation();
        this.autoRenameFromFirstMessage();
        
        this.userInput.value = '';
        this.userInput.style.height = 'auto';
        
        const prompt = this.buildPrompt(message);
        await this.generateResponse(prompt);
    }
    
    buildPrompt(userMessage) {
        let prompt = '';
        
        for (const msg of this.conversationHistory.slice(0, -1)) {
            if (msg.role === 'user') {
                prompt += `Human: ${msg.content}\n\n`;
            } else {
                prompt += `Assistant: ${msg.content}\n\n`;
            }
        }
        
        prompt += `Human: ${userMessage}\n\nAssistant:`;
        
        return prompt;
    }
    
    async generateResponse(prompt) {
        this.isGenerating = true;
        this.sendBtn.style.display = 'none';
        this.cancelBtn.style.display = 'flex';
        this.userInput.disabled = true;
        this.updateStatus('Đang sinh phản hồi...');
        
        const messageDiv = this.addMessage('assistant', '', true);
        const contentDiv = messageDiv.querySelector('.message-content');
        
        let fullResponse = '';
        
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    checkpoint_path: this.currentModel,
                    max_new_tokens: parseInt(this.maxTokensInput.value),
                    temperature: parseFloat(this.temperatureInput.value),
                    top_p: parseFloat(this.topPInput.value),
                    top_k: parseInt(this.topKInput.value),
                })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'token') {
                                fullResponse += data.content;
                                contentDiv.textContent = fullResponse;
                                this.scrollToBottom();
                            } else if (data.type === 'cancelled') {
                                contentDiv.classList.remove('streaming');
                                const cancelledNote = document.createElement('div');
                                cancelledNote.className = 'message-cancelled';
                                cancelledNote.textContent = '[Đã dừng sinh phản hồi]';
                                contentDiv.appendChild(cancelledNote);
                            } else if (data.type === 'error') {
                                this.showError(data.message);
                            } else if (data.type === 'done') {
                                contentDiv.classList.remove('streaming');
                            }
                        } catch (e) {
                            // Ignore parse errors for incomplete chunks
                        }
                    }
                }
            }
            
            if (fullResponse) {
                this.conversationHistory.push({ role: 'assistant', content: fullResponse });
                this.syncActiveConversation();
            }
            
        } catch (error) {
            console.error('Generation error:', error);
            if (error.name !== 'AbortError') {
                this.showError('Sinh phản hồi thất bại: ' + error.message);
            }
            contentDiv.classList.remove('streaming');
        } finally {
            this.isGenerating = false;
            this.sendBtn.style.display = 'flex';
            this.cancelBtn.style.display = 'none';
            this.userInput.disabled = false;
            this.userInput.focus();
            this.updateStatus('Sẵn sàng hội thoại');
        }
    }
    
    async cancelGeneration() {
        try {
            this.updateStatus('Đang dừng sinh phản hồi...');
            await fetch('/api/cancel', { method: 'POST' });
        } catch (error) {
            console.error('Cancel error:', error);
        }
    }
    
    addMessage(role, content, streaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        const avatarImg = document.createElement('img');
        avatarImg.src = role === 'user' ? '/static/user.png' : '/static/logo.png';
        avatarImg.alt = role === 'user' ? 'Người dùng' : 'VLex-RIA';
        avatarImg.onerror = () => {
            avatar.textContent = role === 'user' ? 'U' : 'AI';
        };
        avatar.appendChild(avatarImg);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        if (streaming) {
            contentDiv.classList.add('streaming');
        }
        contentDiv.textContent = content;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    clearChat() {
        if (this.isGenerating) {
            return;
        }
        this.conversationHistory = [];
        this.syncActiveConversation();
        this.renderCurrentConversation();
        this.updateStatus('Đã làm mới cuộc hội thoại');
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showError(message) {
        this.errorToast.textContent = message;
        this.errorToast.style.display = 'block';
        
        setTimeout(() => {
            this.errorToast.style.display = 'none';
        }, 5000);
    }

    updateStatus(message) {
        if (this.statusText) {
            this.statusText.textContent = message;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
