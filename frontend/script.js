document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('input-text');
    let ghostTextElement = null;

    async function fetchPrediction(text) {
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            const data = await response.json();
            return data.prediction;
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
            return '';
        }
    }

    function showGhostText(predictedWord) {
        if (ghostTextElement) {
            ghostTextElement.remove();
        }

        if (predictedWord && inputText.value.trim() !== '') {
            ghostTextElement = document.createElement('span');
            ghostTextElement.classList.add('ghost-text');
            ghostTextElement.textContent = predictedWord;
            inputText.parentNode.insertBefore(ghostTextElement, inputText.nextSibling);

            const { x, y } = inputText.getBoundingClientRect();
            const inputStyle = window.getComputedStyle(inputText);
            const paddingLeft = parseFloat(inputStyle.paddingLeft);
            const paddingTop = parseFloat(inputStyle.paddingTop);

            const tempSpan = document.createElement('span');
            tempSpan.style.whiteSpace = 'pre';
            tempSpan.style.visibility = 'hidden';
            tempSpan.style.font = inputStyle.font;
            tempSpan.style.letterSpacing = inputStyle.letterSpacing;
            tempSpan.textContent = inputText.value;
            document.body.appendChild(tempSpan);
            const textWidth = tempSpan.offsetWidth;
            document.body.removeChild(tempSpan);
            
            ghostTextElement.style.left = `${x + paddingLeft + textWidth}px`;
            ghostTextElement.style.top = `${y + paddingTop}px`;
        }
    }

    inputText.addEventListener('input', async () => {
        const text = inputText.value.trim();
        if (text) {
            const prediction = await fetchPrediction(text);
            showGhostText(prediction);
        } else {
            if (ghostTextElement) {
                ghostTextElement.remove();
                ghostTextElement = null;
            }
        }
    });

    inputText.addEventListener('keydown', (e) => {
        if (e.key === 'Tab' && ghostTextElement) {
            e.preventDefault();
            const currentText = inputText.value;
            const newText = `${currentText.trim()} ${ghostTextElement.textContent} `;
            inputText.value = newText;
            
            const event = new Event('input', {
                bubbles: true,
                cancelable: true,
            });
            inputText.dispatchEvent(event);
        }
    });
});