# 🧠 Image Classification In Medicine

Welcome to the **Image Classification In Medicine** repository — a modular and reproducible deep learning pipeline for medical image classification tasks using PyTorch. This repo includes training scripts, inference demos, and evaluation tools to accelerate your computer vision experiments.

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

The links to the datasets can be found in the [fetch script](https://github.com/bekhzod-olimov/MedicalImageClassificationProjects/blob/7e97ebf29be8a2fb52d0e405921b08cf55683f12/data/fetch.py#L7).
 * Covid Dataset;

 * Malaria Dataset;

 * The more is coming...

 ## 🛠️ Manual to Use This Repo

 ### 🔁 Train and Evaluate a Model
Run the training and evaluation pipeline:

```bash

python main.py --dataset_name pet_disease --dataset_root PATH_TO_YOUR_DATA --batch_size 32 --device "cuda"

```

### 🌐 Streamlit Demo

```bash

streamlit run demo.py --ds_nomi pet_disease --outputs_dir PATH_TO_YOUR_OUTPUTS_DIR --model_name "rexnet_150"

```

## 🤝 Contributing
Have an idea or found a bug? Feel free to open an issue or a pull request. Contributions are welcome!

## 📃 License
This project is open source and available under the MIT License.
