/* eslint-disable semi */
/* eslint-disable array-callback-return */
/* eslint-disable no-undef */
/* eslint-disable no-console */
/* eslint-disable no-debugger */
/* eslint-disable no-unused-expressions */
/* eslint-disable no-unused-labels */
/* eslint-disable no-unused-imports */
/* eslint-disable no-trailing-spaces */
/* eslint-disable indent */


<template>
    <div class="chat-container" v-show="isVisible" :style="{ width: width + 'px', height: height + 'px' }">
        <div class="chat-header">
            <h3>Flight Data Assistant</h3>
            <button class="close-button" @click="closeChat">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="chat-messages" ref="messageContainer">
            <div v-for="(message, index) in messages" :key="index"
                 :class="['message', message.role === 'user' ? 'user-message' : 'system-message']">
                <div class="message-content">
                    <span class="sender">{{ message.role === 'user' ? 'You' : 'Assistant' }}</span>
                    <div class="message-text" v-html="formatMessage(message.content)"></div>
                    <div v-if="message.plotData" :id="'plot-' + index" class="plot-container"></div>
                    <span class="timestamp">{{ formatTime(message.timestamp) }}</span>
                </div>
            </div>
        </div>
        <div class="chat-input">
            <input
                type="text"
                v-model="newMessage"
                @keyup.enter="sendMessage"
                placeholder="Ask about your flight data..."
                :disabled="isLoading"
            >
            <button @click="sendMessage" :disabled="isLoading">
                <i class="fas" :class="isLoading ? 'fa-spinner fa-spin' : 'fa-paper-plane'"></i>
            </button>
        </div>
        <div class="resize-handle" @mousedown="startResize"></div>
    </div>
</template>

<script>
import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'
import Plotly from 'plotly.js-dist'

const md = new MarkdownIt({
    html: true,
    breaks: true,
    linkify: true
})

export default {
    name: 'ChatComponent',
    data () {
        return {
            isVisible: false,
            messages: [],
            newMessage: '',
            isLoading: false,
            width: 350,
            height: 500,
            isResizing: false,
            backendUrl: process.env.NODE_ENV === 'production' 
                ? 'http://localhost:5001'  // Production URL
                : 'http://localhost:5001'  // Development URL
        }
    },
    created () {
        this.$eventHub.$on('loadType', this.loadType)
        this.$eventHub.$on('trimFile', this.trimFile)
        this.$eventHub.$on('flightDataLoaded', () => {
            this.messages.push({
                content: 'Flight data loaded. You can now ask questions about your flight.',
                sender: 'system',
                timestamp: new Date().toISOString()
            })
        })
        // Add resize event listeners
        window.addEventListener('mousemove', this.handleResize)
        window.addEventListener('mouseup', this.stopResize)
    },
    beforeDestroy () {
        this.$eventHub.$off('open-sample')
        // Remove resize event listeners
        window.removeEventListener('mousemove', this.handleResize)
        window.removeEventListener('mouseup', this.stopResize)
    },
    methods: {
        showChat () {
            this.isVisible = true
        },
        closeChat () {
            this.isVisible = false
        },
        formatMessage (content) {
            // Convert markdown to HTML and sanitize
            const html = md.render(content)
            return DOMPurify.sanitize(html, {
                ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'code', 'pre', 'ul',
                 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'span'],
                ALLOWED_ATTR: ['class']
            })
        },
        async sendMessage () {
            if (!this.newMessage.trim()) return

            const userMessage = {
                role: 'user',
                content: this.newMessage,
                timestamp: new Date().toISOString()
            }

            this.messages.push(userMessage)
            this.newMessage = ''
            this.isLoading = true

            try {
                console.log('Sending chat request to backend:', { query: userMessage.content })
                const response = await fetch(`${this.backendUrl}/api/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: userMessage.content })
                })

                if (!response.ok) {
                    throw new Error('Network response was not ok')
                }

                const data = await response.json()
                console.log('Received response from backend:', data)
                
                const assistantMessage = {
                    role: 'assistant',
                    content: data.response,
                    timestamp: new Date().toISOString(),
                    plotData: data.plot_data
                }

                this.messages.push(assistantMessage)
                this.scrollToBottom()

                // Render plot if available
                if (data.plot_data) {
                    console.log('Plot data received, attempting to render...')
                    this.$nextTick(() => {
                        const plotContainer = document.getElementById(`plot-${this.messages.length - 1}`)
                        console.log('Plot container found:', plotContainer)
                        if (plotContainer) {
                            try {
                                const plotData = JSON.parse(data.plot_data)
                                console.log('Parsed plot data:', plotData)
                                Plotly.newPlot(plotContainer, plotData)
                                console.log('Plot rendered successfully')
                            } catch (error) {
                                console.error('Error rendering plot:', error)
                            }
                        }
                    })
                } else {
                    console.log('No plot data received in response')
                }
            } catch (error) {
                console.error('Error:', error)
                this.messages.push({
                    content: 'Sorry, I encountered an error. Please try again.',
                    sender: 'assistant',
                    timestamp: new Date().toISOString()
                })
            } finally {
                this.isLoading = false
            }
        },
        formatTime (timestamp) {
            return new Date(timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            })
        },
        scrollToBottom () {
            this.$nextTick(() => {
                const container = this.$refs.messageContainer
                container.scrollTop = container.scrollHeight
            })
        },
        async updateFlightData (flightData) {
            try {
                await fetch(`${this.backendUrl}/api/set-flight-data`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(flightData)
                })
                this.messages.push({
                    content: 'Flight data loaded. You can now ask questions about your flight.',
                    sender: 'system',
                    timestamp: new Date().toISOString()
                })
            } catch (error) {
                console.error('Error updating flight data:', error)
                this.messages.push({
                    content: 'Error loading flight data. Please try again.',
                    sender: 'system',
                    timestamp: new Date().toISOString()
                })
            }
        },
        startResize (e) {
            this.isResizing = true
            this.startX = e.clientX
            this.startY = e.clientY
            this.startWidth = this.width
            this.startHeight = this.height
        },
        handleResize (e) {
            if (!this.isResizing) return
            
            const deltaX = this.startX - e.clientX
            const deltaY = this.startY - e.clientY
            
            this.width = Math.max(300, this.startWidth + deltaX)
            this.height = Math.max(400, this.startHeight + deltaY)
        },
        stopResize () {
            this.isResizing = false
        }
    }
}
</script>

<style scoped>
.chat-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    min-width: 300px;
    min-height: 400px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    z-index: 1000;
    resize: none;
}

.chat-header {
    padding: 10px 15px;
    background-color: #303336;
    color: white;
    border-radius: 8px 8px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-header h3 {
    margin: 0;
    font-size: 16px;
}

.close-button {
    background: none;
    border: none;
    color: white;
    cursor: pointer;
    padding: 5px;
}

.chat-messages {
    flex-grow: 1;
    padding: 15px;
    overflow-y: auto;
    background-color: #f5f5f5;
}

.message {
    margin-bottom: 10px;
    display: flex;
    flex-direction: column;
}

.message-content {
    max-width: 80%;
    padding: 8px 12px;
    border-radius: 12px;
    position: relative;
}

.user-message {
    align-items: flex-end;
}

.user-message .message-content {
    background-color: #303336;
    color: white;
}

.system-message {
    align-items: flex-start;
}

.system-message .message-content {
    background-color: #e9ecef;
    color: #212529;
}

.sender {
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 4px;
    display: block;
}

.timestamp {
    font-size: 10px;
    color: #6c757d;
    margin-top: 4px;
    display: block;
}

/* Markdown Styling */
.message-text {
    line-height: 1.5;
}

.message-text h1,
.message-text h2,
.message-text h3 {
    margin: 10px 0;
    font-weight: 600;
}

.message-text strong {
    font-weight: 600;
    color: #0056b3;
}

.message-text code {
    background-color: rgba(0, 0, 0, 0.05);
    padding: 2px 4px;
    border-radius: 3px;
    font-family: monospace;
    font-size: 0.9em;
}

.message-text ul,
.message-text ol {
    margin: 8px 0;
    padding-left: 20px;
}

.message-text li {
    margin: 4px 0;
}

.message-text hr {
    margin: 15px 0;
    border: none;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
}

/* Emoji styling */
.message-text .emoji {
    font-size: 1.2em;
    margin-right: 4px;
    vertical-align: middle;
}

.chat-input {
    padding: 10px;
    background-color: white;
    border-top: 1px solid #dee2e6;
    display: flex;
    gap: 10px;
}

.chat-input input {
    flex-grow: 1;
    padding: 8px 12px;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    outline: none;
}

.chat-input input:disabled {
    background-color: #f8f9fa;
    cursor: not-allowed;
}

.chat-input button {
    background-color: #303336;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 12px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.chat-input button:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
}

.chat-input button:hover:not(:disabled) {
    background-color: #404346;
}

/* Scrollbar styling */
.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: #555;
}

.resize-handle {
    position: absolute;
    top: 0;
    left: 0;
    width: 20px;
    height: 20px;
    cursor: nwse-resize;
    background: linear-gradient(135deg, #303336 50%, transparent 50%);
    border-radius: 8px 0 0 0;
}

.resize-handle:hover {
    background: linear-gradient(135deg, #404346 50%, transparent 50%);
}

.plot-container {
    margin: 10px 0;
    width: 100%;
    min-height: 200px;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>
