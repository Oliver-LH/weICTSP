# Wavelet Enhanced In-context Time Series Predictor
# AMA564 Deep Learning individual project

---

#### 1. Install Required Packages
This task require torch >= 2.6.0 & CUDA >- 11.8

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt
```

#### 2. Download Other Datasets (Optional)

You can use the link provided by [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) to download the datasets.

#### 3. Track the Training

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > tensorb.log 2>&1 &
```

#### 4. Run the Scripts

Run the training scripts under `./scripts` folder.

#### 5. Trained Models

Due to limited computational resouces, only few-shot experiments were tried. Check them in `./trained_models` folder.
![image](https://github.com/user-attachments/assets/66bfa929-3b32-40cd-b25a-5b0c73f6aac5)

