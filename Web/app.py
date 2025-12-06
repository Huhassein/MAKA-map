import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from MAKAmodel.model import MAKA

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 模型参数路径
model_weights_path = 'MAKAmodel/maka.pth'
expected_n_channels = 49

def get_feature(pkl_file, expected_n_channels):
    features = pickle.load(open(pkl_file, 'rb'))
    l = len(features['seq'])
    X = np.full((l, l, expected_n_channels), 0.0)
    fi = 0
    pssm = features['pssm']
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis=0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    entropy = features['entropy']
    a = np.repeat(entropy.reshape(1, l), l, axis=0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    X[:, :, fi] = features['ccmpred']
    fi += 1
    X[:, :, fi] = features['freecon']
    fi += 1
    X[:, :, fi] = features['potential']
    fi += 1
    assert fi == expected_n_channels
    return X

def predict_and_plot(pkl_path, output_prefix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_tensor = get_feature(pkl_path, expected_n_channels)
    feature_tensor = np.transpose(feature_tensor, (2, 0, 1))  # [C, L, L]
    feature_tensor = torch.tensor(feature_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # [1, C, L, L]

    model = MAKA(expected_n_channels=expected_n_channels).to(device)
    state_dict = torch.load(model_weights_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if torch.cuda.device_count() > 1 and not k.startswith('module.'):
            name = 'module.' + k
        elif torch.cuda.device_count() <= 1 and k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    with torch.no_grad():
        prediction = model(feature_tensor).cpu().numpy()
        while prediction.ndim > 2:
            prediction = prediction[0]

    npy_path = f"{output_prefix}.npy"
    np.save(npy_path, prediction)

    matrix = np.clip(prediction, 0, 25)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='Spectral', square=True, xticklabels=False, yticklabels=False)
    plt.title('Predicted Distance Map')
    plt.tight_layout()
    png_path = f"{output_prefix}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    return npy_path, png_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    f = request.files['file']
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)

    output_prefix = os.path.splitext(file_path)[0]
    npy_file, png_file = predict_and_plot(file_path, output_prefix)

    return jsonify({
        "resultMessage": "Prediction completed.",
        "npy": os.path.basename(npy_file),
        "png": os.path.basename(png_file)
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/preview/<filename>')
def preview_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
