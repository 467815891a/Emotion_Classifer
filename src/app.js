var MODEL_CONFIGS = {
    efficientnet_lite0_features: {
        label: 'EfficientNet-Lite0',
        IMAGE_SIZE: 224,
        LEARNING_RATE: 6e-2,
        FEATURE_MODEL_PATH: 'models/efficientnet_lite0_features.onnx',
        MEAN: [0.485, 0.456, 0.406],
        STD: [0.229, 0.224, 0.225]
    }
};

var currentModelName = 'efficientnet_lite0_features';

var CONFIG = {
    NUM_CLASSES: 2,
    BATCH_SIZE: 4,
    NUM_EPOCHS: 10,
    FEATURE_BATCH_SIZE: 8,
    CLASS_NAMES: ['happy', 'sad'],
    PER_PAGE: 20,
    ADAM_BETA1: 0.9,
    ADAM_BETA2: 0.999,
    ADAM_EPS: 1e-8
};

function applyModelConfig(modelName) {
    var mc = MODEL_CONFIGS[modelName];
    if (!mc) return;
    CONFIG.IMAGE_SIZE = mc.IMAGE_SIZE;
    CONFIG.LEARNING_RATE = mc.LEARNING_RATE;
    CONFIG.FEATURE_MODEL_PATH = mc.FEATURE_MODEL_PATH;
    CONFIG.MEAN = mc.MEAN.slice();
    CONFIG.STD = mc.STD.slice();
}

applyModelConfig(currentModelName);

var featureSession = null;
var classifierHead = null;
var isModelTrained = false;
var isTraining = false;
var trainStartTime = null;
var elapsedTimer = null;

var trainData = { happy: [], sad: [] };
var valData = { happy: [], sad: [] };

var previewCategory = 'happy';
var previewPage = 1;
var valPreviewCategory = 'happy';
var valPreviewPage = 1;

var cameraStream = null;
var capturedImageData = null;
var croppedBlob = null;
var isSingleImageMode = false;
var uploadedImgElement = null;

function ClassifierHead(inputDim, outputDim) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.weights = new Float32Array(outputDim * inputDim);
    this.bias = new Float32Array(outputDim);

    var scale = Math.sqrt(2.0 / (inputDim + outputDim));
    for (var i = 0; i < this.weights.length; i++) {
        this.weights[i] = (Math.random() * 2 - 1) * scale;
    }
    this.bias.fill(0);

    this.mW = new Float32Array(this.weights.length);
    this.vW = new Float32Array(this.weights.length);
    this.mB = new Float32Array(this.bias.length);
    this.vB = new Float32Array(this.bias.length);
    this.t = 0;
}

ClassifierHead.prototype.forward = function(features) {
    var logits = new Float32Array(this.outputDim);
    for (var i = 0; i < this.outputDim; i++) {
        var sum = this.bias[i];
        var offset = i * this.inputDim;
        for (var j = 0; j < this.inputDim; j++) {
            sum += this.weights[offset + j] * features[j];
        }
        logits[i] = sum;
    }
    return logits;
};

ClassifierHead.prototype.forwardBatch = function(featuresBatch) {
    var batchSize = featuresBatch.length;
    var logitsBatch = new Array(batchSize);
    for (var b = 0; b < batchSize; b++) {
        logitsBatch[b] = this.forward(featuresBatch[b]);
    }
    return logitsBatch;
};

ClassifierHead.prototype.backwardBatch = function(featuresBatch, logitsBatch, labels, learningRate) {
    var batchSize = featuresBatch.length;
    var inputDim = this.inputDim;
    var outputDim = this.outputDim;

    var gradW = new Float32Array(this.weights.length);
    var gradB = new Float32Array(outputDim);

    for (var b = 0; b < batchSize; b++) {
        var probs = softmax(Array.from(logitsBatch[b]));
        var dLogits = new Float32Array(outputDim);
        for (var i = 0; i < outputDim; i++) {
            dLogits[i] = probs[i];
        }
        dLogits[labels[b]] -= 1.0;

        for (var i2 = 0; i2 < outputDim; i2++) {
            gradB[i2] += dLogits[i2];
            var wOffset = i2 * inputDim;
            for (var j = 0; j < inputDim; j++) {
                gradW[wOffset + j] += dLogits[i2] * featuresBatch[b][j];
            }
        }
    }

    var invB = 1.0 / batchSize;
    for (var k = 0; k < gradW.length; k++) {
        gradW[k] *= invB;
    }
    for (var k2 = 0; k2 < gradB.length; k2++) {
        gradB[k2] *= invB;
    }

    this.t += 1;
    this._adamUpdate(this.weights, gradW, this.mW, this.vW, learningRate);
    this._adamUpdate(this.bias, gradB, this.mB, this.vB, learningRate);
};

ClassifierHead.prototype._adamUpdate = function(params, grads, m, v, lr) {
    var beta1 = CONFIG.ADAM_BETA1;
    var beta2 = CONFIG.ADAM_BETA2;
    var eps = CONFIG.ADAM_EPS;
    var t = this.t;
    var bc1 = 1.0 - Math.pow(beta1, t);
    var bc2 = 1.0 - Math.pow(beta2, t);

    for (var i = 0; i < params.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
        var mHat = m[i] / bc1;
        var vHat = v[i] / bc2;
        params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
};

ClassifierHead.prototype.getState = function() {
    return {
        inputDim: this.inputDim,
        outputDim: this.outputDim,
        weights: Array.from(this.weights),
        bias: Array.from(this.bias)
    };
};

ClassifierHead.prototype.loadState = function(state) {
    if (state.inputDim !== this.inputDim || state.outputDim !== this.outputDim) {
        throw new Error('维度不匹配: 期望 (' + this.inputDim + ',' + this.outputDim + '), 实际 (' + state.inputDim + ',' + state.outputDim + ')');
    }
    for (var i = 0; i < this.weights.length; i++) {
        this.weights[i] = state.weights[i];
    }
    for (var j = 0; j < this.bias.length; j++) {
        this.bias[j] = state.bias[j];
    }
    this.mW = new Float32Array(this.weights.length);
    this.vW = new Float32Array(this.weights.length);
    this.mB = new Float32Array(this.bias.length);
    this.vB = new Float32Array(this.bias.length);
    this.t = 0;
};

async function initORT() {
    ort.env.wasm.wasmPaths = '/src/';
    if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
        ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
    } else {
        ort.env.wasm.numThreads = 1;
        addLog('页面未启用 Cross-Origin Isolation，WASM 使用单线程模式', 'warning');
    }
    ort.env.logLevel = 'info';
    addLog('ONNX Runtime Web (Inference) 初始化完成', 'info');
}

function addLog(msg, type) {
    type = type || 'info';
    var logArea = document.getElementById('train-log');
    if (!logArea) return;
    var line = document.createElement('div');
    line.className = 'log-line log-' + type;
    line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
    logArea.appendChild(line);
    logArea.scrollTop = logArea.scrollHeight;
}

function toggleCollapse(bodyId, arrowId) {
    var body = document.getElementById(bodyId);
    var arrow = document.getElementById(arrowId);
    if (body.classList.contains('always-open')) return;
    body.classList.toggle('open');
    arrow.classList.toggle('open');
    if (body.classList.contains('open') && bodyId === 'val-preview-body') {
        loadValPreview();
    }
}

async function loadFeatureExtractor() {
    try {
        var modelLabel = MODEL_CONFIGS[currentModelName] ? MODEL_CONFIGS[currentModelName].label : currentModelName;
        addLog('加载特征提取模型 (' + modelLabel + ')...', 'info');

        var sessionOptions = {
            executionProviders: ['webgpu','wasm'],
            graphOptimizationLevel: 'all'
        };
        featureSession = await ort.InferenceSession.create(CONFIG.FEATURE_MODEL_PATH, sessionOptions);

        addLog('特征提取模型加载成功', 'success');
        addLog('输入: ' + featureSession.inputNames.join(', '), 'info');
        addLog('输出: ' + featureSession.outputNames.join(', '), 'info');

        var dummyData = new Float32Array(3 * CONFIG.IMAGE_SIZE * CONFIG.IMAGE_SIZE);
        var dummyInput = new ort.Tensor('float32', dummyData, [1, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE]);
        var dummyFeeds = {};
        dummyFeeds[featureSession.inputNames[0]] = dummyInput;
        var dummyOut = await featureSession.run(dummyFeeds);
        var outTensor = dummyOut[featureSession.outputNames[0]];
        var outDims = outTensor.dims;
        var outLen = outTensor.data.length;
        addLog('输出形状: [' + outDims.join(', ') + '], 数组长度: ' + outLen, 'info');

        if (outDims.length === 2) {
            CONFIG.FEATURE_DIM = outDims[1];
        } else if (outDims.length === 4) {
            CONFIG.FEATURE_DIM = outDims[1];
        } else if (outDims.length === 3) {
            CONFIG.FEATURE_DIM = outDims[2];
        } else {
            CONFIG.FEATURE_DIM = outLen;
        }
        addLog('自动检测 FEATURE_DIM = ' + CONFIG.FEATURE_DIM, 'info');

        classifierHead = new ClassifierHead(CONFIG.FEATURE_DIM, CONFIG.NUM_CLASSES);
        addLog('分类头已初始化: ' + CONFIG.FEATURE_DIM + ' → ' + CONFIG.NUM_CLASSES + ' (Xavier)', 'info');

        return true;
    } catch (e) {
        var errMsg = e ? (e.message || e.stack || String(e)) : 'unknown error';
        addLog('特征提取模型加载失败: ' + errMsg, 'error');
        console.error('InferenceSession.create full error:', e);
        return false;
    }
}

function l2Normalize(vec) {
    var sumSq = 0;
    for (var i = 0; i < vec.length; i++) {
        if (!isFinite(vec[i])) return new Float32Array(vec.length);
        sumSq += vec[i] * vec[i];
    }
    var norm = Math.sqrt(sumSq);
    if (norm < 1e-12) return new Float32Array(vec.length);
    var result = new Float32Array(vec.length);
    for (var j = 0; j < vec.length; j++) result[j] = vec[j] / norm;
    return result;
}

function poolFeatureMap(rawData, dims) {
    if (dims.length === 2) {
        return new Float32Array(rawData.slice(0, dims[1]));
    }
    if (dims.length === 4) {
        var C = dims[1], H = dims[2], W = dims[3];
        var pooled = new Float32Array(C);
        var hw = H * W;
        for (var c = 0; c < C; c++) {
            var sum = 0;
            var offset = c * hw;
            for (var i = 0; i < hw; i++) {
                sum += rawData[offset + i];
            }
            pooled[c] = sum / hw;
        }
        return pooled;
    }
    if (dims.length === 3) {
        var lastDim = dims[dims.length - 1];
        return new Float32Array(rawData.slice(0, lastDim));
    }
    return new Float32Array(rawData.slice(0, rawData.length));
}

function extractFeatureVector(output) {
    var pooled = poolFeatureMap(output.data, output.dims);
    if (pooled.length !== CONFIG.FEATURE_DIM) {
        addLog('特征维度不匹配: 池化后 ' + pooled.length + ', 期望 ' + CONFIG.FEATURE_DIM, 'error');
        return null;
    }
    return l2Normalize(pooled);
}

async function extractFeatures(imgElement) {
    if (!featureSession || !imgElement) return null;
    var float32Data = preprocessImage(imgElement);
    var inputTensor = new ort.Tensor('float32', float32Data, [1, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE]);
    var feeds = {};
    feeds[featureSession.inputNames[0]] = inputTensor;
    try {
        var results = await featureSession.run(feeds);
        var outputName = featureSession.outputNames[0];
        var output = results[outputName];
        var feature = extractFeatureVector(output);
        if (output && output.dispose) output.dispose();
        if (inputTensor && inputTensor.dispose) inputTensor.dispose();
        return feature;
    } catch (e) {
        addLog('特征提取失败: ' + e.message, 'error');
        if (inputTensor && inputTensor.dispose) inputTensor.dispose();
        return null;
    }
}

async function extractFeaturesBatch(imgElements) {
    if (!featureSession) return [];
    var validIndices = [];
    var validImgs = [];
    for (var i = 0; i < imgElements.length; i++) {
        if (imgElements[i]) {
            validIndices.push(i);
            validImgs.push(imgElements[i]);
        }
    }
    if (validImgs.length === 0) return new Array(imgElements.length).fill(null);

    var outputArray = new Array(imgElements.length).fill(null);

    for (var b = 0; b < validImgs.length; b++) {
        try {
            var singleData = preprocessImage(validImgs[b]);
            var inputTensor = new ort.Tensor('float32', singleData, [1, 3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE]);
            var feeds = {};
            feeds[featureSession.inputNames[0]] = inputTensor;
            var results = await featureSession.run(feeds);
            var outputName = featureSession.outputNames[0];
            var output = results[outputName];
            outputArray[validIndices[b]] = extractFeatureVector(output);
            if (output && output.dispose) output.dispose();
            if (inputTensor && inputTensor.dispose) inputTensor.dispose();
        } catch (e) {
            addLog('特征提取失败 (第' + (b + 1) + '张): ' + e.message, 'error');
        }
    }

    return outputArray;
}

var _preprocessCanvas = null;
var _preprocessCtx = null;

function preprocessImage(img) {
    if (!_preprocessCanvas) {
        _preprocessCanvas = document.createElement('canvas');
        _preprocessCtx = _preprocessCanvas.getContext('2d');
    }
    _preprocessCanvas.width = CONFIG.IMAGE_SIZE;
    _preprocessCanvas.height = CONFIG.IMAGE_SIZE;
    _preprocessCtx.drawImage(img, 0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
    var imageData = _preprocessCtx.getImageData(0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
    var pixels = imageData.data;
    var float32Data = new Float32Array(3 * CONFIG.IMAGE_SIZE * CONFIG.IMAGE_SIZE);
    for (var y = 0; y < CONFIG.IMAGE_SIZE; y++) {
        for (var x = 0; x < CONFIG.IMAGE_SIZE; x++) {
            var idx = (y * CONFIG.IMAGE_SIZE + x) * 4;
            float32Data[0 * CONFIG.IMAGE_SIZE * CONFIG.IMAGE_SIZE + y * CONFIG.IMAGE_SIZE + x] = (pixels[idx] / 255.0 - CONFIG.MEAN[0]) / CONFIG.STD[0];
            float32Data[1 * CONFIG.IMAGE_SIZE * CONFIG.IMAGE_SIZE + y * CONFIG.IMAGE_SIZE + x] = (pixels[idx + 1] / 255.0 - CONFIG.MEAN[1]) / CONFIG.STD[1];
            float32Data[2 * CONFIG.IMAGE_SIZE * CONFIG.IMAGE_SIZE + y * CONFIG.IMAGE_SIZE + x] = (pixels[idx + 2] / 255.0 - CONFIG.MEAN[2]) / CONFIG.STD[2];
        }
    }
    return float32Data;
}

function loadImageFromFile(file) {
    return new Promise(function(resolve, reject) {
        var reader = new FileReader();
        reader.onload = function(e) {
            var img = new Image();
            img.onload = function() { resolve(img); };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function softmax(logits) {
    var maxVal = Math.max.apply(null, logits);
    var exps = logits.map(function(v) { return Math.exp(v - maxVal); });
    var sum = exps.reduce(function(a, b) { return a + b; }, 0);
    return exps.map(function(v) { return v / sum; });
}

function crossEntropyLoss(probs, label) {
    return -Math.log(Math.max(probs[label], 1e-7));
}

async function runEval(imgElement) {
    if (!featureSession || !classifierHead || !imgElement) return null;
    var features = await extractFeatures(imgElement);
    if (!features) return null;
    var logits = classifierHead.forward(features);
    var probs = softmax(Array.from(logits));
    return probs;
}

async function predict(imgElement) {
    if (!featureSession || !classifierHead) return null;
    var probs = await runEval(imgElement);
    if (!probs) return null;
    var maxIdx = probs.indexOf(Math.max.apply(null, probs));
    var label = CONFIG.CLASS_NAMES[maxIdx];
    var confidence = probs[maxIdx];
    return {
        label: label,
        confidence: confidence,
        prob_happy: probs[0].toFixed(4),
        prob_sad: probs[1].toFixed(4),
        probs: probs
    };
}

async function startTrain() {
    if (isTraining) return;
    if (!featureSession) {
        alert('特征提取模型未加载，请刷新页面重试！');
        return;
    }
    var totalTrain = trainData.happy.length + trainData.sad.length;
    var totalVal = valData.happy.length + valData.sad.length;
    if (totalTrain === 0) {
        alert('请先添加训练样本！');
        return;
    }
    isTraining = true;
    document.getElementById('btn-train').disabled = true;
    document.getElementById('train-badge').className = 'badge badge-running';
    document.getElementById('train-badge').textContent = '训练中';
    trainStartTime = Date.now();
    elapsedTimer = setInterval(updateElapsed, 1000);

    var logArea = document.getElementById('train-log');
    if (logArea) logArea.innerHTML = '';

    classifierHead = new ClassifierHead(CONFIG.FEATURE_DIM, CONFIG.NUM_CLASSES);
    isModelTrained = false;
    _valFeatureCache = null;
    addLog('分类头已重新初始化: ' + CONFIG.FEATURE_DIM + ' → ' + CONFIG.NUM_CLASSES + ' (从头训练)', 'info');

    addLog('类别: happy=0, sad=1', 'info');
    addLog('训练集: ' + totalTrain + ' 样本, 验证集: ' + totalVal + ' 样本', 'info');
    addLog('学习率: ' + CONFIG.LEARNING_RATE + ', 批大小: ' + CONFIG.BATCH_SIZE, 'info');
    addLog('开始训练, 共 ' + CONFIG.NUM_EPOCHS + ' 个 Epoch ...', 'info');

    try {
        await trainWithTransferLearning(totalTrain, totalVal);
    } catch (e) {
        addLog('训练错误: ' + e.message, 'error');
        console.error(e);
    }

    isTraining = false;
    document.getElementById('btn-train').disabled = false;
    document.getElementById('train-badge').className = 'badge badge-idle';
    document.getElementById('train-badge').textContent = '空闲';
    document.getElementById('progress-fill').style.width = '100%';
    document.getElementById('progress-pct').textContent = '100%';
    document.getElementById('progress-text').textContent = '训练完成!';
    clearInterval(elapsedTimer);
    updateElapsed();
    isModelTrained = true;
    document.getElementById('btn-export').disabled = false;
    addLog('训练完成!', 'success');
}

async function trainWithTransferLearning(totalTrain, totalVal) {
    var modelLabel = MODEL_CONFIGS[currentModelName] ? MODEL_CONFIGS[currentModelName].label : currentModelName;
    addLog('=== 阶段1: 批量提取训练特征 (' + modelLabel + ' 冻结) ===', 'info');
    addLog('特征提取批大小: ' + CONFIG.FEATURE_BATCH_SIZE, 'info');

    var allFeatures = [];
    var allLabels = [];

    var needExtractImgs = [];
    var needExtractMeta = [];
    var cachedCount = 0;

    for (var ci = 0; ci < CONFIG.CLASS_NAMES.length; ci++) {
        var cat = CONFIG.CLASS_NAMES[ci];
        var label = ci;
        for (var j = 0; j < trainData[cat].length; j++) {
            var item = trainData[cat][j];
            if (item._features) {
                allFeatures.push(item._features);
                allLabels.push(label);
                cachedCount++;
            } else {
                needExtractImgs.push(item.img);
                needExtractMeta.push({cat: cat, idx: j, label: label});
            }
        }
    }

    if (cachedCount > 0) {
        addLog('复用缓存特征: ' + cachedCount + ' 样本', 'info');
    }

    var featBatchSize = CONFIG.FEATURE_BATCH_SIZE;
    for (var start = 0; start < needExtractImgs.length; start += featBatchSize) {
        var end = Math.min(start + featBatchSize, needExtractImgs.length);
        var batchImgs = needExtractImgs.slice(start, end);
        var batchMeta = needExtractMeta.slice(start, end);
        var batchFeatures = await extractFeaturesBatch(batchImgs);
        for (var fi = 0; fi < batchFeatures.length; fi++) {
            if (batchFeatures[fi]) {
                allFeatures.push(batchFeatures[fi]);
                allLabels.push(batchMeta[fi].label);
                trainData[batchMeta[fi].cat][batchMeta[fi].idx]._features = batchFeatures[fi];
            }
        }
        var featRatio = allFeatures.length / totalTrain;
        var totalProgress = Math.min(featRatio, 1) * 0.7;
        document.getElementById('progress-text').textContent = '提取特征 ' + allFeatures.length + '/' + totalTrain + (cachedCount > 0 ? ' (缓存' + cachedCount + ')' : '');
        document.getElementById('progress-fill').style.width = (totalProgress * 100).toFixed(1) + '%';
        document.getElementById('progress-pct').textContent = (totalProgress * 100).toFixed(0) + '%';
        await new Promise(function(r) { setTimeout(r, 0); });
    }

    if (cachedCount === totalTrain) {
        document.getElementById('progress-fill').style.width = '70%';
        document.getElementById('progress-pct').textContent = '70%';
        document.getElementById('progress-text').textContent = '特征已全部缓存，跳过提取';
    }

    addLog('特征提取完成: ' + allFeatures.length + ' 样本 (缓存' + cachedCount + ', 新提取' + (allFeatures.length - cachedCount) + '), 维度: ' + CONFIG.FEATURE_DIM, 'success');

    addLog('=== 阶段2: 微调分类头 (纯 JS Adam 优化器) ===', 'info');

    var batchSize = CONFIG.BATCH_SIZE;
    var numBatches = Math.ceil(allFeatures.length / batchSize);

    for (var epoch = 1; epoch <= CONFIG.NUM_EPOCHS; epoch++) {
        var indices = [];
        for (var i = 0; i < allFeatures.length; i++) indices.push(i);
        shuffleArray(indices);
        var runningLoss = 0;
        var correct = 0;
        var total = 0;

        for (var bi = 0; bi < numBatches; bi++) {
            var batchStart = bi * batchSize;
            var batchEnd = Math.min(batchStart + batchSize, indices.length);
            var batchIndices = indices.slice(batchStart, batchEnd);
            var currentBatchSize = batchIndices.length;

            var featBatch = new Array(currentBatchSize);
            var labelBatch = new Array(currentBatchSize);

            for (var si = 0; si < currentBatchSize; si++) {
                var idx = batchIndices[si];
                featBatch[si] = allFeatures[idx];
                labelBatch[si] = allLabels[idx];
            }

            var logitsBatch = classifierHead.forwardBatch(featBatch);

            var batchLoss = 0;
            var batchCorrect = 0;
            for (var si2 = 0; si2 < currentBatchSize; si2++) {
                var probs = softmax(Array.from(logitsBatch[si2]));
                batchLoss += crossEntropyLoss(probs, labelBatch[si2]);
                var pred = probs.indexOf(Math.max.apply(null, probs));
                if (pred === labelBatch[si2]) batchCorrect++;
            }

            classifierHead.backwardBatch(featBatch, logitsBatch, labelBatch, CONFIG.LEARNING_RATE);

            runningLoss += batchLoss;
            correct += batchCorrect;
            total += currentBatchSize;

            var epochProgress = (bi + 1) / numBatches;
            var totalProgress2 = 0.7 + ((epoch - 1 + epochProgress) / CONFIG.NUM_EPOCHS) * 0.3;
            updateTrainProgress(epoch, epochProgress, totalProgress2, runningLoss / total, correct / total);
            if (bi % 5 === 0 || bi === numBatches - 1) {
                await new Promise(function(r) { setTimeout(r, 0); });
            }
        }

        var avgLoss = runningLoss / total;
        var trainAcc = correct / total;
        addLog('Epoch ' + epoch + '/' + CONFIG.NUM_EPOCHS + ' - Loss: ' + avgLoss.toFixed(4) + ', Acc: ' + (trainAcc * 100).toFixed(1) + '%', 'info');
        document.getElementById('s-train-acc').textContent = (trainAcc * 100).toFixed(1) + '%';

        if (totalVal > 0) await validateClassifier(epoch, totalVal);
    }
}

var _valFeatureCache = null;
var _valFeatureCacheKey = '';

function _getValCacheKey() {
    var key = '';
    for (var ci = 0; ci < CONFIG.CLASS_NAMES.length; ci++) {
        var cat = CONFIG.CLASS_NAMES[ci];
        key += cat + ':' + valData[cat].length + ',';
    }
    return key;
}

async function _ensureValFeatures() {
    var key = _getValCacheKey();
    if (_valFeatureCache && _valFeatureCacheKey === key) return _valFeatureCache;

    var allImgs = [];
    var allLabels = [];
    for (var ci = 0; ci < CONFIG.CLASS_NAMES.length; ci++) {
        var cat = CONFIG.CLASS_NAMES[ci];
        var label = ci;
        for (var j = 0; j < valData[cat].length; j++) {
            allImgs.push(valData[cat][j].img);
            allLabels.push(label);
        }
    }

    var allFeatures = [];
    var featBatchSize = CONFIG.FEATURE_BATCH_SIZE;
    for (var start = 0; start < allImgs.length; start += featBatchSize) {
        var end = Math.min(start + featBatchSize, allImgs.length);
        var batchImgs = allImgs.slice(start, end);
        var batchLabels = allLabels.slice(start, end);
        try {
            var batchFeatures = await extractFeaturesBatch(batchImgs);
            for (var fi = 0; fi < batchFeatures.length; fi++) {
                if (batchFeatures[fi]) {
                    allFeatures.push({ feature: batchFeatures[fi], label: batchLabels[fi] });
                }
            }
        } catch (e) { }
    }

    _valFeatureCache = allFeatures;
    _valFeatureCacheKey = key;
    return allFeatures;
}

async function validateClassifier(epoch, totalVal) {
    var allValItems = await _ensureValFeatures();
    var correct = 0;
    var total = allValItems.length;

    for (var i = 0; i < allValItems.length; i++) {
        var logits = classifierHead.forward(allValItems[i].feature);
        var probs = softmax(Array.from(logits));
        var pred = probs.indexOf(Math.max.apply(null, probs));
        if (pred === allValItems[i].label) correct++;
    }

    var valAcc = total > 0 ? correct / total : 0;
    var valAccEl = document.getElementById('s-val-acc');
    if (valAccEl) valAccEl.textContent = (valAcc * 100).toFixed(1) + '%';
    addLog('Epoch ' + epoch + ' 验证 - Acc: ' + (valAcc * 100).toFixed(1) + '%', 'info');
}

function updateTrainProgress(epoch, epochProgress, totalProgress, loss, acc) {
    document.getElementById('s-train-acc').textContent = (acc * 100).toFixed(1) + '%';
    document.getElementById('progress-fill').style.width = (totalProgress * 100).toFixed(1) + '%';
    document.getElementById('progress-text').textContent = 'Epoch ' + epoch + ' 训练中';
    document.getElementById('progress-pct').textContent = (totalProgress * 100).toFixed(0) + '%';
}

function updateElapsed() {
    if (!trainStartTime) return;
    var s = Math.floor((Date.now() - trainStartTime) / 1000);
    var m = Math.floor(s / 60);
    var sec = s % 60;
    document.getElementById('s-elapsed').textContent = m > 0 ? m + 'm ' + sec + 's' : sec + 's';
}

function shuffleArray(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

function switchPreview(cat, el) {
    previewCategory = cat;
    previewPage = 1;
    var tabs = document.querySelectorAll('#preview-body .preview-tab');
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
    el.classList.add('active');
    loadPreview();
}

function loadPreview() {
    var items = trainData[previewCategory] || [];
    var total = items.length;
    var pages = Math.max(1, Math.ceil(total / CONFIG.PER_PAGE));
    previewPage = Math.max(1, Math.min(previewPage, pages));
    var start = (previewPage - 1) * CONFIG.PER_PAGE;
    var pageItems = items.slice(start, start + CONFIG.PER_PAGE);

    var gallery = document.getElementById('preview-gallery');
    gallery.innerHTML = '';
    for (var i = 0; i < pageItems.length; i++) {
        var item = pageItems[i];
        var div = document.createElement('div');
        div.className = 'gallery-item';
        var img = document.createElement('img');
        img.src = item.thumbUrl;
        var nameDiv = document.createElement('div');
        nameDiv.className = 'name';
        nameDiv.textContent = item.name;
        var delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = '×';
        delBtn.setAttribute('data-idx', start + i);
        delBtn.onclick = function() {
            var idx = parseInt(this.getAttribute('data-idx'));
            deleteTrainImage(previewCategory, idx);
        };
        div.appendChild(img);
        div.appendChild(nameDiv);
        div.appendChild(delBtn);
        gallery.appendChild(div);
    }

    var pagination = document.getElementById('preview-pagination');
    pagination.innerHTML = '';
    if (pages > 1) {
        var prevBtn = document.createElement('button');
        prevBtn.textContent = '上一页';
        prevBtn.disabled = previewPage <= 1;
        prevBtn.onclick = function() { previewPage--; loadPreview(); };
        pagination.appendChild(prevBtn);
        var span = document.createElement('span');
        span.textContent = previewPage + ' / ' + pages;
        pagination.appendChild(span);
        var nextBtn = document.createElement('button');
        nextBtn.textContent = '下一页';
        nextBtn.disabled = previewPage >= pages;
        nextBtn.onclick = function() { previewPage++; loadPreview(); };
        pagination.appendChild(nextBtn);
    }

    document.getElementById('train-upload-status').textContent = previewCategory + ': ' + total + ' 张';
}

function uploadTrainImages() {
    var input = document.getElementById('train-upload-input');
    var files = input.files;
    if (!files || files.length === 0) return;
    var count = 0;
    var total = files.length;

    for (var i = 0; i < files.length; i++) {
        (function(file) {
            loadImageFromFile(file).then(function(img) {
                var thumbUrl = getThumbUrl(img);
                trainData[previewCategory].push({
                    img: img,
                    thumbUrl: thumbUrl,
                    name: file.name
                });
                count++;
                if (count === total) {
                    loadPreview();
                    addLog('添加 ' + total + ' 张' + previewCategory + '训练图片', 'success');
                }
            });
        })(files[i]);
    }
    input.value = '';
}

function deleteTrainImage(category, idx) {
    trainData[category].splice(idx, 1);
    loadPreview();
}

function deleteAllTrainImages() {
    if (!confirm('确认删除所有 ' + previewCategory + ' 训练图片？')) return;
    trainData[previewCategory] = [];
    loadPreview();
    addLog('已删除所有 ' + previewCategory + ' 训练图片', 'info');
}

function switchValPreview(cat, el) {
    valPreviewCategory = cat;
    valPreviewPage = 1;
    var tabs = document.querySelectorAll('#val-preview-body .preview-tab');
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove('active');
    el.classList.add('active');
    loadValPreview();
}

function loadValPreview() {
    var items = valData[valPreviewCategory] || [];
    var total = items.length;
    var pages = Math.max(1, Math.ceil(total / CONFIG.PER_PAGE));
    valPreviewPage = Math.max(1, Math.min(valPreviewPage, pages));
    var start = (valPreviewPage - 1) * CONFIG.PER_PAGE;
    var pageItems = items.slice(start, start + CONFIG.PER_PAGE);

    var gallery = document.getElementById('val-preview-gallery');
    gallery.innerHTML = '';
    for (var i = 0; i < pageItems.length; i++) {
        var item = pageItems[i];
        var div = document.createElement('div');
        div.className = 'gallery-item';
        var img = document.createElement('img');
        img.src = item.thumbUrl;
        var nameDiv = document.createElement('div');
        nameDiv.className = 'name';
        nameDiv.textContent = item.name;
        var delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = '×';
        delBtn.setAttribute('data-idx', start + i);
        delBtn.onclick = function() {
            var idx = parseInt(this.getAttribute('data-idx'));
            deleteValImage(valPreviewCategory, idx);
        };
        div.appendChild(img);
        div.appendChild(nameDiv);
        div.appendChild(delBtn);
        gallery.appendChild(div);
    }

    var pagination = document.getElementById('val-preview-pagination');
    pagination.innerHTML = '';
    if (pages > 1) {
        var prevBtn = document.createElement('button');
        prevBtn.textContent = '上一页';
        prevBtn.disabled = valPreviewPage <= 1;
        prevBtn.onclick = function() { valPreviewPage--; loadValPreview(); };
        pagination.appendChild(prevBtn);
        var span = document.createElement('span');
        span.textContent = valPreviewPage + ' / ' + pages;
        pagination.appendChild(span);
        var nextBtn = document.createElement('button');
        nextBtn.textContent = '下一页';
        nextBtn.disabled = valPreviewPage >= pages;
        nextBtn.onclick = function() { valPreviewPage++; loadValPreview(); };
        pagination.appendChild(nextBtn);
    }

    document.getElementById('val-upload-status').textContent = valPreviewCategory + ': ' + total + ' 张';
}

function uploadValImages() {
    var input = document.getElementById('val-upload-input');
    var files = input.files;
    if (!files || files.length === 0) return;
    var count = 0;
    var total = files.length;

    for (var i = 0; i < files.length; i++) {
        (function(file) {
            loadImageFromFile(file).then(function(img) {
                var thumbUrl = getThumbUrl(img);
                valData[valPreviewCategory].push({
                    img: img,
                    thumbUrl: thumbUrl,
                    name: file.name
                });
                count++;
                if (count === total) {
                    loadValPreview();
                    addLog('添加 ' + total + ' 张' + valPreviewCategory + '验证图片', 'success');
                }
            });
        })(files[i]);
    }
    input.value = '';
}

function deleteValImage(category, idx) {
    valData[category].splice(idx, 1);
    loadValPreview();
}

function deleteAllValImages() {
    if (!confirm('确认删除所有 ' + valPreviewCategory + ' 验证图片？')) return;
    valData[valPreviewCategory] = [];
    loadValPreview();
    addLog('已删除所有 ' + valPreviewCategory + ' 验证图片', 'info');
}

function getThumbUrl(img) {
    var canvas = document.createElement('canvas');
    var maxThumb = 200;
    var ratio = Math.min(maxThumb / img.naturalWidth, maxThumb / img.naturalHeight, 1);
    canvas.width = Math.round(img.naturalWidth * ratio);
    canvas.height = Math.round(img.naturalHeight * ratio);
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7);
}

function toggleCamera() {
    if (cameraStream) {
        stopCamera();
        return;
    }
    startCamera();
}

function isMobileDevice() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
}

async function startCamera() {
    try {
        resetSingleImageMode();
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
        });
        var video = document.getElementById('camera-video');
        video.srcObject = cameraStream;
        document.getElementById('camera-placeholder').style.display = 'none';
        document.getElementById('camera-container').style.display = 'block';
        document.getElementById('btn-camera').textContent = '关闭摄像头';
        document.getElementById('btn-capture').disabled = false;
        document.getElementById('btn-capture').textContent = '📸 拍照并裁剪人脸';
        document.getElementById('btn-capture').onclick = captureAndCrop;
        addLog('摄像头已开启', 'success');
    } catch (e) {
        addLog('摄像头开启失败: ' + e.message, 'error');
        alert('无法访问摄像头: ' + e.message);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(function(t) { t.stop(); });
        cameraStream = null;
    }
    var video = document.getElementById('camera-video');
    video.srcObject = null;
    document.getElementById('camera-placeholder').style.display = '';
    document.getElementById('camera-container').style.display = 'none';
    document.getElementById('btn-camera').textContent = '开启摄像头';
    document.getElementById('btn-capture').disabled = true;
    document.getElementById('btn-capture').textContent = '📸 拍照并裁剪人脸';
    document.getElementById('btn-capture').onclick = captureAndCrop;
    addLog('摄像头已关闭', 'info');
}

function verifySingleImage(event) {
    var file = event.target.files[0];
    if (!file) return;

    if (cameraStream) {
        stopCamera();
    }

    var objectUrl = URL.createObjectURL(file);

    // 先立即显示预览，无需等待
    var uploadedEl = document.getElementById('uploaded-image');
    uploadedEl.src = objectUrl;
    uploadedEl.style.display = 'block';
    document.getElementById('camera-video').style.display = 'none';
    document.getElementById('camera-placeholder').style.display = 'none';
    document.getElementById('camera-container').style.display = 'block';
    document.getElementById('capture-overlay').style.display = 'none';
    document.getElementById('cropped-preview').style.display = 'none';
    document.getElementById('predict-result').style.display = 'none';

    var btnCapture = document.getElementById('btn-capture');
    btnCapture.style.display = '';
    btnCapture.disabled = false;
    btnCapture.textContent = '✂️ 裁切人脸';
    btnCapture.onclick = cropSingleImage;

    // 先隐藏裁剪框，避免显示默认的扁形状
    var cropBox = document.getElementById('live-crop-box');
    cropBox.style.display = 'none';

    // 延迟加载 Image 对象用于后续裁切，不阻塞预览
    var img = new Image();
    img.onload = function() {
        uploadedImgElement = img;
        isSingleImageMode = true;

        // 图片渲染完成后计算1:1正方形再显示
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                initCropBoxSquare();
            });
        });

        addLog('已加载单张图片: ' + file.name, 'success');
    };
    img.src = objectUrl;

    event.target.value = '';
}

function initCropBoxSquare() {
    var cropBox = document.getElementById('live-crop-box');
    var containerEl = document.getElementById('camera-container');
    var cW = containerEl.offsetWidth;
    if (!cW) return;
    var cH = containerEl.offsetHeight || cW * 0.75;
    var sidePctW = 50;
    var sidePctH = sidePctW * cW / cH; // 修正：宽/高比例
    var leftPct = (100 - sidePctW) / 2;
    var topPct = (100 - sidePctH) / 2;
    cropBox.style.display = '';
    cropBox.style.left = leftPct + '%';
    cropBox.style.top = topPct + '%';
    cropBox.style.width = sidePctW + '%';
    cropBox.style.height = sidePctH + '%';
}

function cropSingleImage() {
    var cropBox = document.getElementById('live-crop-box');
    var img = uploadedImgElement;
    if (!img || !isSingleImageMode) {
        addLog('图片尚未加载完成，请稍候', 'error');
        return;
    }

    var containerRect = document.getElementById('camera-container').getBoundingClientRect();
    var displayW = containerRect.width;
    var displayH = document.getElementById('uploaded-image').offsetHeight;
    var imgW = img.naturalWidth;
    var imgH = img.naturalHeight;

    var scaleX = imgW / displayW;
    var scaleY = imgH / displayH;

    var cropLeft = parseFloat(cropBox.style.left) / 100 * displayW * scaleX;
    var cropTop = parseFloat(cropBox.style.top) / 100 * displayH * scaleY;
    var cropWidth = parseFloat(cropBox.style.width) / 100 * displayW * scaleX;
    var cropHeight = parseFloat(cropBox.style.height) / 100 * displayH * scaleY;

    cropLeft = Math.max(0, Math.round(cropLeft));
    cropTop = Math.max(0, Math.round(cropTop));
    cropWidth = Math.min(Math.round(cropWidth), imgW - cropLeft);
    cropHeight = Math.min(Math.round(cropHeight), imgH - cropTop);

    if (cropWidth < 10 || cropHeight < 10) {
        addLog('裁剪区域太小', 'error');
        return;
    }

    var freezeCanvas = document.getElementById('freeze-canvas');
    freezeCanvas.width = imgW;
    freezeCanvas.height = imgH;
    var fctx = freezeCanvas.getContext('2d');
    fctx.drawImage(img, 0, 0, imgW, imgH);

    fctx.fillStyle = 'rgba(0,0,0,0.55)';
    fctx.fillRect(0, 0, imgW, imgH);

    fctx.save();
    fctx.beginPath();
    fctx.rect(cropLeft, cropTop, cropWidth, cropHeight);
    fctx.clip();
    fctx.drawImage(img, 0, 0, imgW, imgH);
    fctx.restore();

    fctx.strokeStyle = '#60a5fa';
    fctx.lineWidth = 3;
    fctx.strokeRect(cropLeft, cropTop, cropWidth, cropHeight);

    document.getElementById('capture-overlay').style.display = 'block';
    document.getElementById('live-crop-box').style.display = 'none';

    var cropCanvas = document.getElementById('capture-canvas');
    cropCanvas.width = cropWidth;
    cropCanvas.height = cropHeight;
    var cctx = cropCanvas.getContext('2d');
    cctx.drawImage(img, cropLeft, cropTop, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

    capturedImageData = cropCanvas.toDataURL('image/jpeg', 0.9);
    croppedBlob = null;
    cropCanvas.toBlob(function(blob) { croppedBlob = blob; }, 'image/jpeg', 0.9);

    document.getElementById('cropped-preview').style.display = '';
    document.getElementById('predict-result').style.display = 'none';

    var btnCapture = document.getElementById('btn-capture');
    btnCapture.textContent = '🔄 重选';
    btnCapture.onclick = retakeSingleImage;

    updateCaptureCount();
    addLog('裁切成功, 区域: ' + cropWidth + 'x' + cropHeight, 'success');
}

function retakeSingleImage() {
    capturedImageData = null;
    croppedBlob = null;
    document.getElementById('capture-overlay').style.display = 'none';
    document.getElementById('live-crop-box').style.display = '';
    document.getElementById('cropped-preview').style.display = 'none';
    document.getElementById('predict-result').style.display = 'none';

    var btnCapture = document.getElementById('btn-capture');
    btnCapture.textContent = '✂️ 裁切人脸';
    btnCapture.onclick = cropSingleImage;

    // 重置裁剪框为1:1正方形
    setTimeout(initCropBoxSquare, 50);
}

function resetSingleImageMode() {
    isSingleImageMode = false;
    uploadedImgElement = null;
    document.getElementById('uploaded-image').style.display = 'none';
    document.getElementById('uploaded-image').src = '';
    document.getElementById('camera-video').style.display = '';
}

function captureAndCrop() {
    var video = document.getElementById('camera-video');
    var cropBox = document.getElementById('live-crop-box');

    var vw = video.videoWidth;
    var vh = video.videoHeight;
    if (!vw || !vh) {
        addLog('视频流未就绪', 'error');
        return;
    }

    var containerRect = document.getElementById('camera-container').getBoundingClientRect();
    var displayW = containerRect.width;
    var displayH = video.offsetHeight;

    var scaleX = vw / displayW;
    var scaleY = vh / displayH;

    var cropLeft = parseFloat(cropBox.style.left) / 100 * displayW * scaleX;
    var cropTop = parseFloat(cropBox.style.top) / 100 * displayH * scaleY;
    var cropWidth = parseFloat(cropBox.style.width) / 100 * displayW * scaleX;
    var cropHeight = parseFloat(cropBox.style.height) / 100 * displayH * scaleY;

    cropLeft = Math.max(0, Math.round(cropLeft));
    cropTop = Math.max(0, Math.round(cropTop));
    cropWidth = Math.min(Math.round(cropWidth), vw - cropLeft);
    cropHeight = Math.min(Math.round(cropHeight), vh - cropTop);

    if (cropWidth < 10 || cropHeight < 10) {
        addLog('裁剪区域太小', 'error');
        return;
    }

    var freezeCanvas = document.getElementById('freeze-canvas');
    freezeCanvas.width = vw;
    freezeCanvas.height = vh;
    var fctx = freezeCanvas.getContext('2d');

    fctx.save();
    fctx.translate(vw, 0);
    fctx.scale(-1, 1);
    fctx.drawImage(video, 0, 0, vw, vh);
    fctx.restore();

    fctx.fillStyle = 'rgba(0,0,0,0.55)';
    fctx.fillRect(0, 0, vw, vh);

    fctx.save();
    fctx.beginPath();
    fctx.rect(cropLeft, cropTop, cropWidth, cropHeight);
    fctx.clip();
    fctx.save();
    fctx.translate(vw, 0);
    fctx.scale(-1, 1);
    fctx.drawImage(video, 0, 0, vw, vh);
    fctx.restore();
    fctx.restore();

    fctx.strokeStyle = '#60a5fa';
    fctx.lineWidth = 3;
    fctx.strokeRect(cropLeft, cropTop, cropWidth, cropHeight);

    document.getElementById('capture-overlay').style.display = 'block';
    document.getElementById('live-crop-box').style.display = 'none';

    var cropCanvas = document.getElementById('capture-canvas');
    cropCanvas.width = cropWidth;
    cropCanvas.height = cropHeight;
    var cctx = cropCanvas.getContext('2d');
    cctx.save();
    cctx.translate(cropWidth, 0);
    cctx.scale(-1, 1);
    cctx.drawImage(video, cropLeft, cropTop, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
    cctx.restore();

    capturedImageData = cropCanvas.toDataURL('image/jpeg', 0.9);
    croppedBlob = null;
    cropCanvas.toBlob(function(blob) { croppedBlob = blob; }, 'image/jpeg', 0.9);

    document.getElementById('cropped-preview').style.display = '';
    document.getElementById('predict-result').style.display = 'none';

    var btnCapture = document.getElementById('btn-capture');
    btnCapture.textContent = '🔄 重拍';
    btnCapture.onclick = retakePhoto;

    updateCaptureCount();
    addLog('拍照成功, 裁剪区域: ' + cropWidth + 'x' + cropHeight, 'success');
}

function retakePhoto() {
    capturedImageData = null;
    croppedBlob = null;
    document.getElementById('capture-overlay').style.display = 'none';
    document.getElementById('live-crop-box').style.display = '';
    document.getElementById('cropped-preview').style.display = 'none';
    document.getElementById('predict-result').style.display = 'none';

    var btnCapture = document.getElementById('btn-capture');
    btnCapture.textContent = '📸 拍照并裁剪人脸';
    btnCapture.onclick = captureAndCrop;

    // 重置裁剪框为1:1正方形
    setTimeout(initCropBoxSquare, 50);
}

async function verifyCropped() {
    if (!capturedImageData || !featureSession) {
        addLog('请先拍照或确认模型已加载', 'error');
        return;
    }

    var img = new Image();
    img.src = capturedImageData;
    await new Promise(function(r) { img.onload = r; });

    var result = await predict(img);
    if (result) {
        var labelEl = document.getElementById('predict-label');
        labelEl.textContent = result.label === 'happy' ? '😊 开心' : '😢 伤心';
        labelEl.className = 'result-label ' + (result.label === 'happy' ? 'label-happy' : 'label-sad');
        document.getElementById('predict-detail').textContent =
            '开心: ' + result.prob_happy + ' | 伤心: ' + result.prob_sad + ' | 置信度: ' + (result.confidence * 100).toFixed(1) + '%';
        document.getElementById('predict-result').style.display = '';
        addLog('验证结果: ' + result.label + ' (' + (result.confidence * 100).toFixed(1) + '%)', 'success');
    }
}

function showCategoryModal() {
    document.getElementById('category-modal').classList.add('show');
}

function hideCategoryModal() {
    document.getElementById('category-modal').classList.remove('show');
}

function addToDataset(category) {
    if (!capturedImageData) return;
    var img = new Image();
    img.src = capturedImageData;
    var thumbUrl = getThumbUrl(img);
    valData[category].push({
        img: img,
        thumbUrl: thumbUrl,
        name: 'capture_' + Date.now() + '.jpg'
    });
    hideCategoryModal();
    updateCaptureCount();
    addLog('已添加到验证集: ' + category, 'success');
}

function updateCaptureCount() {
}

async function runTest() {
    if (!featureSession) {
        alert('请先加载模型！');
        return;
    }
    if (!isModelTrained) {
        alert('请先训练模型！');
        return;
    }

    var totalVal = valData.happy.length + valData.sad.length;
    if (totalVal === 0) {
        alert('验证数据集为空，请先添加验证样本！');
        return;
    }

    var testBtn = document.getElementById('btn-test');
    testBtn.disabled = true;
    testBtn.textContent = '测试中...';

    var testImages = { happy: [], sad: [] };
    for (var ci2 = 0; ci2 < CONFIG.CLASS_NAMES.length; ci2++) {
        var cat2 = CONFIG.CLASS_NAMES[ci2];
        for (var vi = 0; vi < valData[cat2].length; vi++) {
            testImages[cat2].push({ img: valData[cat2][vi].img, name: valData[cat2][vi].name });
        }
    }

    var totalTest = testImages.happy.length + testImages.sad.length;
    if (totalTest === 0) {
        alert('没有可用的测试图片！');
        testBtn.disabled = false;
        testBtn.textContent = '运行测试';
        return;
    }

    addLog('开始测试, 共 ' + totalTest + ' 张图片', 'info');

    var correct = 0;
    var total = 0;
    var results = [];
    var tested = 0;

    var allTestImgs = [];
    var allTestMeta = [];
    for (var ci3 = 0; ci3 < CONFIG.CLASS_NAMES.length; ci3++) {
        var cat3 = CONFIG.CLASS_NAMES[ci3];
        var trueLabel = ci3;
        for (var ti = 0; ti < testImages[cat3].length; ti++) {
            allTestImgs.push(testImages[cat3][ti].img);
            allTestMeta.push({
                img: testImages[cat3][ti].img,
                name: testImages[cat3][ti].name,
                trueLabel: cat3,
                trueLabelIdx: trueLabel
            });
        }
    }

    var featBatchSize = CONFIG.FEATURE_BATCH_SIZE;
    for (var start = 0; start < allTestImgs.length; start += featBatchSize) {
        var end = Math.min(start + featBatchSize, allTestImgs.length);
        var batchImgs = allTestImgs.slice(start, end);
        var batchMeta = allTestMeta.slice(start, end);
        try {
            var batchFeatures = await extractFeaturesBatch(batchImgs);
            for (var fi = 0; fi < batchFeatures.length; fi++) {
                var meta = batchMeta[fi];
                if (batchFeatures[fi]) {
                    var logits = classifierHead.forward(batchFeatures[fi]);
                    var probs = softmax(Array.from(logits));
                    var maxIdx = probs.indexOf(Math.max.apply(null, probs));
                    var predLabelStr = CONFIG.CLASS_NAMES[maxIdx];
                    var confidence = probs[maxIdx];
                    var isCorrect = maxIdx === meta.trueLabelIdx;
                    if (isCorrect) correct++;
                    total++;
                    results.push({
                        img: meta.img,
                        name: meta.name,
                        trueLabel: meta.trueLabel,
                        predLabel: predLabelStr,
                        confidence: confidence,
                        correct: isCorrect
                    });
                }
                tested++;
            }
        } catch (e) { }
        tested = Math.min(start + featBatchSize, allTestImgs.length);
        testBtn.textContent = '⏳ ' + tested + '/' + totalTest;
        await new Promise(function(r) { setTimeout(r, 0); });
    }

    var accuracy = total > 0 ? correct / total : 0;
    addLog('测试完成: ' + correct + '/' + total + ' = ' + (accuracy * 100).toFixed(1) + '%', 'success');

    var summaryEl = document.getElementById('test-summary');
    summaryEl.style.display = '';
    summaryEl.innerHTML =
        '<div class="stat"><div class="label">总样本</div><div class="value">' + total + '</div></div>' +
        '<div class="stat"><div class="label">正确数</div><div class="value">' + correct + '</div></div>' +
        '<div class="stat"><div class="label">准确率</div><div class="value">' + (accuracy * 100).toFixed(1) + '%</div></div>';

    var detailCard = document.getElementById('test-detail-card');
    detailCard.style.display = '';
    var resultsEl = document.getElementById('test-results');
    resultsEl.innerHTML = '';

    for (var ri = 0; ri < results.length; ri++) {
        var r = results[ri];
        var card = document.createElement('div');
        card.className = 'result-card';
        var thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = 150;
        thumbCanvas.height = 110;
        var tctx = thumbCanvas.getContext('2d');
        var scale = Math.min(150 / r.img.naturalWidth, 110 / r.img.naturalHeight);
        var dw = r.img.naturalWidth * scale;
        var dh = r.img.naturalHeight * scale;
        tctx.drawImage(r.img, (150 - dw) / 2, (110 - dh) / 2, dw, dh);

        var info = document.createElement('div');
        info.className = 'info';
        info.innerHTML = r.predLabel + ' (' + (r.confidence * 100).toFixed(0) + '%) ' +
            '<span class="tag ' + (r.correct ? 'tag-correct' : 'tag-wrong') + '">' +
            (r.correct ? '✓' : '✗') + '</span>';

        card.appendChild(thumbCanvas);
        card.appendChild(info);
        resultsEl.appendChild(card);
    }

    testBtn.disabled = false;
    testBtn.textContent = '运行测试';
}

function loadImageFromUrl(url) {
    return new Promise(function(resolve, reject) {
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function() { resolve(img); };
        img.onerror = function() { reject(new Error('加载失败: ' + url)); };
        img.src = url;
    });
}

async function exportModel() {
    if (!isModelTrained || !classifierHead) {
        alert('请先训练模型！');
        return;
    }

    addLog('导出模型...', 'info');
    var zip = new JSZip();

    try {
        var headState = classifierHead.getState();
        var modelLabel = MODEL_CONFIGS[currentModelName] ? MODEL_CONFIGS[currentModelName].label : currentModelName;
        var modelInfo = {
            version: 4,
            timestamp: new Date().toISOString(),
            architecture: modelLabel + '-FeatureExtractor + JS-ClassifierHead',
            featureModel: currentModelName,
            numClasses: CONFIG.NUM_CLASSES,
            classNames: CONFIG.CLASS_NAMES,
            imageSize: CONFIG.IMAGE_SIZE,
            featureDim: CONFIG.FEATURE_DIM,
            trainingMode: 'feature_extraction_js_finetune',
            classifierHead: headState
        };
        zip.file('model_info.json', JSON.stringify(modelInfo, null, 2));

        var weightsBuf = new Float32Array(headState.weights);
        var biasBuf = new Float32Array(headState.bias);
        zip.file('classifier_weights.bin', weightsBuf.buffer);
        zip.file('classifier_bias.bin', biasBuf.buffer);
    } catch (e) {
        addLog('导出失败: ' + e.message, 'error');
        return;
    }

    var content = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(content);
    a.download = 'emotion_model_' + new Date().toISOString().slice(0, 10) + '.zip';
    a.click();
    URL.revokeObjectURL(a.href);
    addLog('模型导出完成', 'success');
}

async function importModel(event) {
    var file = event.target.files[0];
    if (!file) return;

    addLog('导入模型: ' + file.name, 'info');

    try {
        var zip = await JSZip.loadAsync(file);
        var modelInfoFile = zip.file('model_info.json');
        if (!modelInfoFile) {
            throw new Error('无效的模型文件: 缺少 model_info.json');
        }

        var modelInfoStr = await modelInfoFile.async('string');
        var modelInfo = JSON.parse(modelInfoStr);
        addLog('模型信息: ' + modelInfo.architecture + ', ' + modelInfo.numClasses + ' 类', 'info');

        if (!classifierHead || classifierHead.inputDim !== CONFIG.FEATURE_DIM) {
            classifierHead = new ClassifierHead(CONFIG.FEATURE_DIM, CONFIG.NUM_CLASSES);
            addLog('分类头已按当前模型维度重建: ' + CONFIG.FEATURE_DIM + ' → ' + CONFIG.NUM_CLASSES, 'info');
        }

        if (!featureSession) {
            throw new Error('请先加载特征提取模型再导入参数');
        }

        if (modelInfo.classifierHead) {
            classifierHead.loadState(modelInfo.classifierHead);
            addLog('分类头参数已加载 (v4 JSON格式)', 'success');
        } else {
            var weightsFile = zip.file('classifier_weights.bin');
            var biasFile = zip.file('classifier_bias.bin');
            if (weightsFile && biasFile) {
                var weightsBuf = await weightsFile.async('arraybuffer');
                var biasBuf = await biasFile.async('arraybuffer');
                var headState = {
                    inputDim: modelInfo.featureDim || CONFIG.FEATURE_DIM,
                    outputDim: modelInfo.numClasses || CONFIG.NUM_CLASSES,
                    weights: Array.from(new Float32Array(weightsBuf)),
                    bias: Array.from(new Float32Array(biasBuf))
                };
                classifierHead.loadState(headState);
                addLog('分类头参数已加载 (v4 二进制格式)', 'success');
            } else if (modelInfo.version === 3) {
                var paramsFile = zip.file('trained_params.bin');
                if (paramsFile) {
                    addLog('v3 格式不兼容新架构，请使用 v4 格式模型', 'warn');
                    throw new Error('v3 TrainingSession 格式不兼容当前 FeatureExtraction+JS 架构');
                }
            } else {
                throw new Error('未找到有效的分类头参数');
            }
        }

        isModelTrained = true;
        document.getElementById('btn-export').disabled = false;
        addLog('模型导入完成', 'success');
    } catch (e) {
        addLog('模型导入失败: ' + e.message, 'error');
        alert('导入失败: ' + e.message);
    }

    event.target.value = '';
}

function initCropBox() {
    var cropBox = document.getElementById('live-crop-box');
    var container = document.getElementById('camera-container');
    var isDragging = false;
    var isResizing = false;
    var resizeHandle = '';
    var startX, startY, startLeft, startTop, startWidth, startHeight;

    function getPercent(val, total) {
        return (val / total * 100) + '%';
    }

    function enforceSquare(leftPct, topPct, widthPct, heightPct) {
        var containerW = container.offsetWidth;
        var containerH = container.offsetHeight || containerW * 0.75;
        var pxW = widthPct / 100 * containerW;
        var pxH = heightPct / 100 * containerH;
        var side = Math.min(pxW, pxH);
        var newWPct = side / containerW * 100;
        var newHPct = side / containerH * 100;
        return { left: leftPct, top: topPct, width: newWPct, height: newHPct };
    }

    cropBox.addEventListener('mousedown', function(e) {
        var target = e.target;
        if (target.classList.contains('handle')) {
            isResizing = true;
            resizeHandle = target.getAttribute('data-handle');
        } else {
            isDragging = true;
        }
        startX = e.clientX;
        startY = e.clientY;
        startLeft = parseFloat(cropBox.style.left);
        startTop = parseFloat(cropBox.style.top);
        startWidth = parseFloat(cropBox.style.width);
        startHeight = parseFloat(cropBox.style.height);
        e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
        if (!isDragging && !isResizing) return;
        var dx = e.clientX - startX;
        var dy = e.clientY - startY;
        var containerW = container.offsetWidth;
        var containerH = container.offsetHeight || containerW * 0.75;
        var dxPct = dx / containerW * 100;
        var dyPct = dy / containerH * 100;

        if (isDragging) {
            var newLeft = Math.max(0, Math.min(100 - startWidth, startLeft + dxPct));
            var newTop = Math.max(0, Math.min(100 - startHeight, startTop + dyPct));
            cropBox.style.left = newLeft + '%';
            cropBox.style.top = newTop + '%';
        }

        if (isResizing) {
            var newW = startWidth;
            var newH = startHeight;
            var newL = startLeft;
            var newT = startTop;

            var delta = 0;
            if (resizeHandle.indexOf('r') >= 0) {
                delta = Math.max(10 - startWidth, Math.min(100 - startLeft, startWidth + dxPct)) - startWidth;
            }
            if (resizeHandle.indexOf('l') >= 0) {
                var dw = Math.min(startWidth - 10, dxPct);
                delta = -dw;
            }
            if (resizeHandle.indexOf('b') >= 0) {
                var db = Math.max(10 - startHeight, Math.min(100 - startTop, startHeight + dyPct)) - startHeight;
                if (resizeHandle.indexOf('r') < 0 && resizeHandle.indexOf('l') < 0) delta = db;
                else delta = (Math.abs(delta) > Math.abs(db)) ? delta : db;
            }
            if (resizeHandle.indexOf('t') >= 0) {
                var dt = -Math.min(startHeight - 10, dyPct);
                if (resizeHandle.indexOf('r') < 0 && resizeHandle.indexOf('l') < 0) delta = dt;
                else delta = (Math.abs(delta) > Math.abs(dt)) ? delta : dt;
            }

            newW = startWidth + delta;
            newH = startHeight + delta;

            if (resizeHandle.indexOf('l') >= 0) {
                newL = startLeft - delta;
            }
            if (resizeHandle.indexOf('t') >= 0) {
                newT = startTop - delta;
            }

            if (newL < 0) { newW += newL; newH += newL; newL = 0; }
            if (newT < 0) { newH += newT; newW += newT; newT = 0; }
            if (newL + newW > 100) { newW = 100 - newL; newH = newW; }
            if (newT + newH > 100) { newH = 100 - newT; newW = newH; }
            if (newW < 10) { newW = 10; newH = 10; }

            var sq = enforceSquare(newL, newT, newW, newH);
            cropBox.style.left = sq.left + '%';
            cropBox.style.top = sq.top + '%';
            cropBox.style.width = sq.width + '%';
            cropBox.style.height = sq.height + '%';
        }
    });

    document.addEventListener('mouseup', function() {
        isDragging = false;
        isResizing = false;
    });

    cropBox.addEventListener('touchstart', function(e) {
        var touch = e.touches[0];
        var target = e.target;
        if (target.classList.contains('handle')) {
            isResizing = true;
            resizeHandle = target.getAttribute('data-handle');
        } else {
            isDragging = true;
        }
        startX = touch.clientX;
        startY = touch.clientY;
        startLeft = parseFloat(cropBox.style.left);
        startTop = parseFloat(cropBox.style.top);
        startWidth = parseFloat(cropBox.style.width);
        startHeight = parseFloat(cropBox.style.height);
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchmove', function(e) {
        if (!isDragging && !isResizing) return;
        var touch = e.touches[0];
        var dx = touch.clientX - startX;
        var dy = touch.clientY - startY;
        var containerW = container.offsetWidth;
        var containerH = container.offsetHeight || containerW * 0.75;
        var dxPct = dx / containerW * 100;
        var dyPct = dy / containerH * 100;

        if (isDragging) {
            var newLeft = Math.max(0, Math.min(100 - startWidth, startLeft + dxPct));
            var newTop = Math.max(0, Math.min(100 - startHeight, startTop + dyPct));
            cropBox.style.left = newLeft + '%';
            cropBox.style.top = newTop + '%';
        }

        if (isResizing) {
            var newW = startWidth;
            var newH = startHeight;
            var newL = startLeft;
            var newT = startTop;

            var delta = 0;
            if (resizeHandle.indexOf('r') >= 0) {
                delta = Math.max(10 - startWidth, Math.min(100 - startLeft, startWidth + dxPct)) - startWidth;
            }
            if (resizeHandle.indexOf('l') >= 0) {
                var dw = Math.min(startWidth - 10, dxPct);
                delta = -dw;
            }
            if (resizeHandle.indexOf('b') >= 0) {
                var db = Math.max(10 - startHeight, Math.min(100 - startTop, startHeight + dyPct)) - startHeight;
                if (resizeHandle.indexOf('r') < 0 && resizeHandle.indexOf('l') < 0) delta = db;
                else delta = (Math.abs(delta) > Math.abs(db)) ? delta : db;
            }
            if (resizeHandle.indexOf('t') >= 0) {
                var dt = -Math.min(startHeight - 10, dyPct);
                if (resizeHandle.indexOf('r') < 0 && resizeHandle.indexOf('l') < 0) delta = dt;
                else delta = (Math.abs(delta) > Math.abs(dt)) ? delta : dt;
            }

            newW = startWidth + delta;
            newH = startHeight + delta;

            if (resizeHandle.indexOf('l') >= 0) {
                newL = startLeft - delta;
            }
            if (resizeHandle.indexOf('t') >= 0) {
                newT = startTop - delta;
            }

            if (newL < 0) { newW += newL; newH += newL; newL = 0; }
            if (newT < 0) { newH += newT; newW += newT; newT = 0; }
            if (newL + newW > 100) { newW = 100 - newL; newH = newW; }
            if (newT + newH > 100) { newH = 100 - newT; newW = newH; }
            if (newW < 10) { newW = 10; newH = 10; }

            var sq = enforceSquare(newL, newT, newW, newH);
            cropBox.style.left = sq.left + '%';
            cropBox.style.top = sq.top + '%';
            cropBox.style.width = sq.width + '%';
            cropBox.style.height = sq.height + '%';
        }
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchend', function() {
        isDragging = false;
        isResizing = false;
    });

    // 初始化裁剪框为1:1正方形
    setTimeout(initCropBoxSquare, 200);
}

async function loadBundleAsImages(exampleDir, btn, label) {
    btn.disabled = true;
    btn.textContent = '⏳ 0%';
    addLog('加载' + label + '...', 'info');

    var dataMap = {};
    var totalFiles = 0;

    for (var ci = 0; ci < CONFIG.CLASS_NAMES.length; ci++) {
        var cat = CONFIG.CLASS_NAMES[ci];
        var dir = exampleDir + '/' + cat;
        try {
            var resp = await fetch('/api/bundle?dir=' + encodeURIComponent(dir));
            if (!resp.ok) continue;
            var fileCount = parseInt(resp.headers.get('X-File-Count') || '0', 10);
            totalFiles += fileCount;
            var blob = await resp.blob();
            var zip = await JSZip.loadAsync(blob);
            var fileNames = Object.keys(zip.files).filter(function(n) { return !zip.files[n].dir; });
            for (var fi = 0; fi < fileNames.length; fi++) {
                var fileData = await zip.files[fileNames[fi]].async('blob');
                var url = URL.createObjectURL(fileData);
                var img = await loadImageFromUrl(url);
                dataMap[cat] = dataMap[cat] || [];
                dataMap[cat].push({ img: img, thumbUrl: getThumbUrl(img), name: fileNames[fi] });
                var pct = Math.round(((ci * 1000 + fi + 1) / (CONFIG.CLASS_NAMES.length * 1000)) * 100);
                btn.textContent = '⏳ ' + pct + '%';
            }
        } catch (e) {
            addLog('加载 ' + dir + ' 失败: ' + e.message, 'error');
        }
    }

    btn.disabled = false;
    btn.textContent = '📂 加载默认' + label;

    var total = 0;
    for (var k in dataMap) total += dataMap[k].length;
    if (total > 0) {
        addLog('已加载 ' + total + ' 张' + label, 'success');
    } else {
        addLog('未找到' + label, 'warn');
    }
    return dataMap;
}

async function loadTestExamplesAsTrainData() {
    var btn = document.getElementById('btn-load-train');
    var dataMap = await loadBundleAsImages('train_examples', btn, '训练样本');
    trainData = { happy: [], sad: [] };
    for (var cat in dataMap) {
        trainData[cat] = dataMap[cat];
    }
    if (Object.keys(dataMap).length > 0) loadPreview();
}

async function loadTestExamplesAsValData() {
    var btn = document.getElementById('btn-load-val');
    var dataMap = await loadBundleAsImages('test_examples', btn, '验证样本');
    valData = { happy: [], sad: [] };
    for (var cat in dataMap) {
        valData[cat] = dataMap[cat];
    }
    if (Object.keys(dataMap).length > 0) {
        updateCaptureCount();
        loadValPreview();
    }
}

async function init() {
    addLog('初始化情绪分类器...', 'info');

    setupCameraUI();

    var modelLabel = MODEL_CONFIGS[currentModelName].label;
    addLog('架构: ' + modelLabel + ' 特征提取 (ORT) + JS 分类头微调', 'info');
    await initORT();

    var loaded = await loadFeatureExtractor();
    if (!loaded) {
        addLog('特征提取模型加载失败，部分功能不可用', 'error');
    }

    initCropBox();
    loadPreview();
    updateCaptureCount();

    addLog('系统就绪', 'success');
    addLog('提示: 请上传训练图片或使用测试样本开始训练', 'info');
}

function setupCameraUI() {
    var isMobile = isMobileDevice();
    var isSecure = window.isSecureContext;
    var hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    var canUseLiveCamera = !isMobile && isSecure && hasMediaDevices;

    var btnCamera = document.getElementById('btn-camera');
    var btnCameraLabel = document.getElementById('btn-camera-label');

    if (canUseLiveCamera) {
        btnCamera.style.display = '';
        btnCameraLabel.style.display = 'none';
    } else {
        btnCamera.style.display = 'none';
        btnCameraLabel.style.display = '';
        if (!isSecure && !isMobile) {
            addLog('非安全上下文: 实时摄像头不可用，请用 localhost 访问', 'warn');
            addLog('或 Chrome 访问 chrome://flags/#unsafely-treat-insecure-origin-as-secure', 'info');
        }
    }
}

window.addEventListener('DOMContentLoaded', init);
