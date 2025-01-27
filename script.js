document.addEventListener('DOMContentLoaded', function() {
    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
        });
    });

    // Generate Button Click Handler
    const generateBtn = document.querySelector('.generate-btn');
    generateBtn.addEventListener('click', () => {
        const prompt = document.querySelector('textarea').value;
        const style = document.querySelector('.style-select').value;
        const ratio = document.querySelector('.ratio-select').value;
        
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }
        
        // Add loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        
        // Simulate API call (replace with actual API integration)
        setTimeout(() => {
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-magic"></i> Generate';
            alert('Generation complete! (This is a demo)');
        }, 2000);
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });


    // Constants for API
const API_KEY = 'your_api_key_here'; // Replace with your actual API key
const IMAGE_API_URL = 'https://api.openai.com/v1/images/generations'; // Example using OpenAI's API
const VIDEO_API_URL = 'your_video_api_endpoint'; // Replace with your video generation API

class AIGenerator {
    constructor() {
        this.initializeUI();
        this.bindEvents();
    }

    initializeUI() {
        this.prompt = document.querySelector('textarea');
        this.styleSelect = document.querySelector('.style-select');
        this.ratioSelect = document.querySelector('.ratio-select');
        this.generateBtn = document.querySelector('.generate-btn');
        this.resultContainer = document.querySelector('.generation-result');
        this.tabBtns = document.querySelectorAll('.tab-btn');
        this.currentTab = 'image'; // Default tab
    }

    bindEvents() {
        this.generateBtn.addEventListener('click', () => this.generate());
        this.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.currentTab = btn.dataset.tab;
                this.tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.clearResult();
            });
        });
    }

    async generate() {
        if (!this.prompt.value.trim()) {
            this.showError('Please enter a prompt');
            return;
        }

        this.setLoadingState(true);

        try {
            const result = await (this.currentTab === 'image' 
                ? this.generateImage() 
                : this.generateVideo());
            
            this.displayResult(result);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.setLoadingState(false);
        }
    }

    async generateImage() {
        const response = await fetch(IMAGE_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                prompt: this.prompt.value,
                n: 1,
                size: this.ratioSelect.value,
                style: this.styleSelect.value
            })
        });

        if (!response.ok) {
            throw new Error('Failed to generate image');
        }

        const data = await response.json();
        return data.data[0].url; // Adjust based on your API response structure
    }

    async generateVideo() {
        // Implement video generation API call
        const response = await fetch(VIDEO_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                prompt: this.prompt.value,
                style: this.styleSelect.value,
                aspect_ratio: this.ratioSelect.value
            })
        });

        if (!response.ok) {
            throw new Error('Failed to generate video');
        }

        const data = await response.json();
        return data.url; // Adjust based on your API response structure
    }

    displayResult(result) {
        this.clearResult();
        
        const resultElement = document.createElement(this.currentTab === 'image' ? 'img' : 'video');
        resultElement.src = result;
        if (this.currentTab === 'video') {
            resultElement.controls = true;
        }
        
        const downloadBtn = this.createDownloadButton(result);
        const saveBtn = this.createSaveButton(result);

        const resultWrapper = document.createElement('div');
        resultWrapper.className = 'result-wrapper';
        resultWrapper.appendChild(resultElement);
        
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'result-buttons';
        buttonContainer.appendChild(downloadBtn);
        buttonContainer.appendChild(saveBtn);
        
        resultWrapper.appendChild(buttonContainer);
        this.resultContainer.appendChild(resultWrapper);
    }

    createDownloadButton(url) {
        const btn = document.createElement('a');
        btn.href = url;
        btn.download = `generated-${this.currentTab}-${Date.now()}`;
        btn.className = 'result-btn download-btn';
        btn.innerHTML = '<i class="fas fa-download"></i> Download';
        return btn;
    }

    createSaveButton(url) {
        const btn = document.createElement('button');
        btn.className = 'result-btn save-btn';
        btn.innerHTML = '<i class="fas fa-heart"></i> Save to Gallery';
        btn.onclick = () => this.saveToGallery(url);
        return btn;
    }

    async saveToGallery(url) {
        // Implement save to gallery functionality
        try {
            // Add to gallery section
            const galleryGrid = document.querySelector('.gallery-grid');
            const galleryItem = document.createElement('div');
            galleryItem.className = 'gallery-item';
            galleryItem.innerHTML = `
                <${this.currentTab === 'image' ? 'img' : 'video'} src="${url}" ${this.currentTab === 'video' ? 'controls' : ''}>
                <div class="gallery-item-overlay">
                    <p>${this.prompt.value}</p>
                    <button class="delete-btn"><i class="fas fa-trash"></i></button>
                </div>
            `;
            galleryGrid.prepend(galleryItem);
        } catch (error) {
            this.showError('Failed to save to gallery');
        }
    }

    setLoadingState(isLoading) {
        this.generateBtn.disabled = isLoading;
        this.generateBtn.innerHTML = isLoading 
            ? '<i class="fas fa-spinner fa-spin"></i> Generating...'
            : '<i class="fas fa-magic"></i> Generate';
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.resultContainer.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 3000);
    }

    clearResult() {
        this.resultContainer.innerHTML = '';
    }
}

// Initialize the generator when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AIGenerator();
});

});