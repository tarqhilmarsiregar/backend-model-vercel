const tf = require('@tensorflow/tfjs-node');
const path = require('path');

let model;

// Fungsi untuk memuat model TensorFlow.js
async function loadModel() {
    if (!model) {
    try {
        const modelPath = path.join(__dirname, '..', 'model', 'model.json');
        model = await tf.loadGraphModel(`file://${modelPath}`);
        console.log('Model loaded successfully!');
    } catch (error) {
        console.error('Failed to load model:', error);
        process.exit(1);
    }
    }
    return model;
}

module.exports = loadModel;