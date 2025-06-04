# 🧠 Object Detection Sample Projects

Welcome to the **Object Detection Sample Projects** repository — a modular and reproducible deep learning pipeline for object detection tasks using PyTorch. This repo includes training scripts, inference demos, and evaluation tools to accelerate your computer vision experiments.

---

## 🚀 Getting Started

Follow the steps below to set up your environment and start training your models.

### 📦 1. Install Python Virtual Environment (Ubuntu)

```bash
sudo apt install -y python3-virtualenv
```

### 🔧 2. Create & Activate Virtual Environment

```bash

virtualenv ENV_NAME
source ENV_NAME/bin/activate

```

### 📚 3. Install Required Dependencies

```bash

pip install -r requirements.txt

```

### 🧠 4. Register Jupyter Kernel (Optional)

```bash

python -m ipykernel install --name "ENV_NAME" --user

```

## 📁 Available Datasets

The links to the datasets can be found in the [fetch script](https://github.com/bekhzod-olimov/ObjectDetectionSampleProjects/blob/f470170f29d56f12ca5b018ff2afe95db2d3e6d6/data/fetch.py#L7).
 * Baggage Detection Dataset;

 * The more is coming...

 ## 🛠️ Manual to Use This Repo

 ### 🔁 Train and Evaluate a Model
Run the training and evaluation pipeline:

```bash

python main.py --dataset_name pet_disease --dataset_root PATH_TO_YOUR_DATA --batch_size 32 --device "cuda" --model_name yolo11n.pt --epochs 20

```

### 🌐 Streamlit Demo

```bash

streamlit run demo.py

```

## 🤝 Contributing
Have an idea or found a bug? Feel free to open an issue or a pull request. Contributions are welcome!

## 📃 License
This project is open source and available under the MIT License.
