import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import streamlit as st
import torch
import random
import os
import pickle
from PIL import Image
from glob import glob
from src.infer import YOLOv11Inference 
from ultralytics import YOLO

st.set_page_config(page_title="Image Classification Demo", layout="wide")
@st.cache_resource
def load_model(save_path):  return YOLO(os.path.join(save_path, "weights", "best.pt"))

class StreamlitApp:
    def __init__(self, ds_nomi, model_name, device):
        self.ds_nomi = ds_nomi
        self.device  = device
        self.model_name = model_name        
        self.lang_code = "en" 

        self.LANGUAGES = {
            "en": {
                "title": "AI Model Inference Visualization",
                "description": "Select or upload an image to run inference and visualize the results.",
                "upload_button": "Upload Your Image",
                "random_images_label": "Select a Random Image",
                "result_label": "Model Results",
                "accuracy_label": "Model Accuracy:",
                "f1_score_label": "F1 Score:",
                "select_language": "Select Language",
            },
            "ko": {
                "title": "AI 모델 추론 시각화",
                "description": "이미지를 선택하거나 업로드하여 추론을 실행하고 결과를 시각화합니다.",
                "upload_button": "이미지 업로드",
                "random_images_label": "랜덤 이미지 선택",
                "result_label": "모델 결과",
                "accuracy_label": "모델 정확도:",
                "f1_score_label": "F1 점수:",
                "select_language": "언어 선택",
            }
        }

    def run(self):
        sample_ims_dir = "demo_ims"
        ims_dir = "/home/bekhzod/Desktop/backup/object_detection_project_datasets"

        language = st.selectbox(
            self.LANGUAGES['en']['select_language'],
            options=['English', 'Korean'],
            index=0
        )
        self.lang_code = "en" if language == "English" else "ko"

        st.title(self.LANGUAGES[self.lang_code]["title"])
        st.write(self.LANGUAGES[self.lang_code]["description"])

        train_name = f"{self.ds_nomi}_{os.path.splitext(self.model_name)[0]}"

        # Load class names and model
        model = load_model(save_path=os.path.join("runs", "detect", f"{train_name}"))
                
        yolo_infer = YOLOv11Inference(model, train_name=train_name)        

        # Prepare sample images
        save_dir = os.path.join(sample_ims_dir, self.ds_nomi)
        os.makedirs(save_dir, exist_ok=True)
        sample_image_paths = glob(os.path.join(save_dir, "*.png"))

        if len(sample_image_paths) == 0:
            rasm_yolaklari = glob(f"{ims_dir}/{self.ds_nomi}/{self.ds_nomi}/test/images/*.jpg")
            random_images = random.sample(rasm_yolaklari, 5)
            for idx, path in enumerate(random_images):                
                with Image.open(path).convert("RGB") as im: im.save(os.path.join(save_dir, f"sample_im_{idx + 1}.png"))
            sample_image_paths = glob(os.path.join(save_dir, "*.png"))

        # UI: Image selection or upload
        selected_image = st.selectbox(self.LANGUAGES[self.lang_code]["random_images_label"], sample_image_paths)
        uploaded_image = st.file_uploader(self.LANGUAGES[self.lang_code]["upload_button"], type=["jpg", "png", "jpeg"])

        im_path = uploaded_image if uploaded_image else selected_image

        if im_path:
            with st.spinner("Running inference..."):
                result = yolo_infer.demo(im_path)                                

            st.subheader(self.LANGUAGES[self.lang_code]["result_label"])

            row1_col1, row1_col2 = st.columns(2)
            row2_col, = st.columns(1)                   

            with row1_col1:
                st.markdown(f"<h3 style='text-align: center;'> Original Input Image </h3>", unsafe_allow_html=True)
                st.image(result["original_im"], use_container_width=True)

            with row1_col2:
                st.markdown(f"<h3 style='text-align: center;'> Predicted Bounding Boxes </h3>", unsafe_allow_html=True)
                st.image(result["pred"], use_container_width=True)

            with row2_col: 
                n_obs = result['n_bboxes']
                text = "objects are" if n_obs >= 2 else "object is"
                
                st.markdown(f"<h1 style='text-align: center;'>  {n_obs} {text} detected. </h1>", unsafe_allow_html=True)                
        else:
            st.warning("Please select or upload an image.")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument("--device", type=str, default="cuda", help="CPU | GPU")
    parser.add_argument("--model_name", type=str, default="yolo11n.pt", help="Model architecture from ultralytics")        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    available_datasets = [os.path.basename(res).split("_results")[0].split("_")[0] for res in glob(f"results/*.png")]
    ds_nomi = st.sidebar.selectbox("Choose Dataset", options=available_datasets, index=0)
    model_name = st.sidebar.text_input("Model name", value=args.model_name)    
    device = args.device if args.device == "cpu" else [0]

    app = StreamlitApp(
        ds_nomi=ds_nomi,
        model_name=model_name,
        device=device
    )
    app.run()