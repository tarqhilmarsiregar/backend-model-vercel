const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');

const inferenceModel = async (model, imageData) => {
    // 1. Dekode Base64 ke buffer
    const base64Data = imageData.replace(/^data:image\/(png|jpeg);base64,/, ""); // Hapus header
    const imageBuffer = Buffer.from(base64Data, 'base64');

    // 2. Pra-pemrosesan gambar menggunakan sharp
    const { data, info } = await sharp(imageBuffer)
        .resize(224, 224) // Sesuaikan dengan IMAGE_WIDTH, IMAGE_HEIGHT dari pelatihan model
        .raw()         // Mengambil data piksel mentah (biasanya RGBA jika input, tapi bisa diatur)
        .toBuffer({ resolveWithObject: true }); // Dapatkan buffer dan info gambar

    // Untuk memastikan 3 channel (RGB):
    const pixelsArray = new Uint8Array(data); // Pastikan ini Uint8Array
    const inputTensor = tf.tensor3d(pixelsArray, [info.height, info.width, info.channels], 'int32');

    // Jika model hanya menerima 3 channel, dan sharp mengembalikan 4 (RGBA),
    // maka perlu membuang channel alpha di sini:
    const finalTensor = info.channels === 4 ? inputTensor.slice([0, 0, 0], [-1, -1, 3]) : inputTensor;

    const normalizedTensor = finalTensor.toFloat().div(tf.scalar(255)); // Normalisasi 0-1
    const batchedTensor = normalizedTensor.expandDims(0); // Tambah dimensi batch

    // 3. Prediksi
    const predictions = model.predict(batchedTensor);
    const predictionArray = await predictions.array();

    // Pastikan untuk membuang tensor setelah digunakan untuk mencegah kebocoran memori
    inputTensor.dispose();
    finalTensor.dispose(); // Gunakan finalTensor jika Anda membuat slice baru
    normalizedTensor.dispose();
    batchedTensor.dispose();
    predictions.dispose();

    // 4. Proses hasil prediksi
    const OPENED_CLASS_INDEX = 0; // Indeks untuk kelas 'Opened'
    const CLOSED_CLASS_INDEX = 1; // Indeks untuk kelas 'Closed'

    const openedConfidence = predictionArray[0][OPENED_CLASS_INDEX];
    const closedConfidence = predictionArray[0][CLOSED_CLASS_INDEX];

    let predictedClass;
    let confidence;

    // Jika confidence score untuk 'Opened' di atas 0.5
    if (openedConfidence > 0.5) {
        predictedClass = 'Opened';
        confidence = openedConfidence;
    } else {
        // Jika tidak, bisa mengambil kelas dengan confidence tertinggi
        // secara default 'Closed' jika openedConfidence tidak cukup tinggi
        if (closedConfidence > openedConfidence) { // Periksa apakah Closed lebih tinggi
            predictedClass = 'Closed';
            confidence = closedConfidence;
        } else {
            // Ini terjadi jika openedConfidence tidak > 0.5 Tapi lebih tinggi dari closedConfidence,
            // atau jika keduanya sangat rendah, bisa memutuskan default menjadi 'Closed'.
            predictedClass = 'Closed';
            confidence = closedConfidence; // Bisa juga menggunakan openedConfidence jika itu lebih tinggi
        }
    }

    return { predictedClass, confidence };
}

module.exports = inferenceModel;