import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import streamlit as st
import torch
import random
import os
import cv2
import tempfile
from PIL import Image
from glob import glob
from src.utils import makedirs
from src.infer import YOLOv11Inference 
from ultralytics import YOLO

st.set_page_config(page_title="AI Detection Demo", layout="wide")

@st.cache_resource
def load_model(save_path):  
    return YOLO(os.path.join(save_path, "weights", "best.pt"))

class StreamlitApp:
    def __init__(self, ds_nomi, model_name, device):
        self.ds_nomi = ds_nomi
        self.device = device        
        self.model_name = model_name        
        self.lang_code = "en"
        self.mode = "image"  # New mode selector

        self.LANGUAGES = {
            "en": {
                "title": "AI Model Inference Visualization",
                "description": "Select or upload media to run inference and visualize results.",
                "upload_button": "Upload Your Media",
                "random_images_label": "Select a Random Image",
                "result_label": "Model Results",
                "select_language": "Select Language",
                "mode_select": "Select Input Type",  
                "video_save_path": "Saved Videos",  
            },
            "ko": {
                "title": "AI 모델 추론 시각화",
                "description": "미디어를 선택하거나 업로드하여 추론을 실행하고 결과를 시각화합니다.",
                "upload_button": "미디어 업로드",
                "random_images_label": "랜덤 이미지 선택",
                "result_label": "모델 결과",
                "select_language": "언어 선택",
                "mode_select": "입력 유형 선택",  
                "video_save_path": "저장된 비디오",  
            }
        }

    def process_video(self, model, video_path):
        """Process video frames and save annotated results"""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create temp output file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            result = model.demo(frame)
            annotated_frame = result["pred"]                      
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            
        cap.release()
        out.release()
        progress_bar.empty()
        return output_path

    def run(self):
        # New mode selector in sidebar
        self.mode = st.sidebar.radio(
            self.LANGUAGES[self.lang_code]["mode_select"],
            ["image", "video"],
            index=0
        )

        # Existing language selector
        language = st.sidebar.selectbox(
            self.LANGUAGES['en']['select_language'],
            options=['English', 'Korean'],
            index=0
        )
        self.lang_code = "en" if language == "English" else "ko"

        st.title(self.LANGUAGES[self.lang_code]["title"])
        st.write(self.LANGUAGES[self.lang_code]["description"])

        self.train_name = f"{self.ds_nomi}_{os.path.splitext(self.model_name)[0]}"        
        model = load_model(save_path=os.path.join("runs", "detect", f"{self.train_name}"))
        yolo_infer = YOLOv11Inference(model, train_name=self.train_name, device = self.device)

        if self.mode == "image":
            # Existing image processing code
            self.handle_image_mode(yolo_infer)
        else:
            # New video processing code
            self.handle_video_mode(yolo_infer)

    def handle_image_mode(self, yolo_infer):
        # Original image handling code remains unchanged
        sample_ims_dir = "demo_ims"
        ims_dir = "/home/bekhzod/Desktop/backup/object_detection_project_datasets"       
        
        # Original image sampling logic
        sample_ims_dir = "demo_ims"
        ims_dir = "/home/bekhzod/Desktop/backup/object_detection_project_datasets"
        
        save_dir = os.path.join(sample_ims_dir, self.ds_nomi)
        makedirs(save_dir)
        sample_image_paths = glob(os.path.join(save_dir, "*.png"))

        if not sample_image_paths:
            source_dir = f"{ims_dir}/{self.ds_nomi}/{self.ds_nomi}/test/images/*.jpg"
            for idx, path in enumerate(random.sample(glob(source_dir), 5)):
                with Image.open(path).convert("RGB") as im:
                    im.save(os.path.join(save_dir, f"sample_im_{idx+1}.png"))
            sample_image_paths = glob(os.path.join(save_dir, "*.png"))

        # UI Elements
        selected_image = st.selectbox(
            self.LANGUAGES[self.lang_code]["random_images_label"],
            sample_image_paths
        )
        uploaded_image = st.file_uploader(
            self.LANGUAGES[self.lang_code]["upload_button"],
            type=["jpg", "png", "jpeg"]
        )

        im_path = uploaded_image or selected_image
        res_save_dir = os.path.join("results", "images", self.train_name)
        makedirs(res_save_dir)

        if im_path:
            with st.spinner("Running inference..."):
                result = yolo_infer.demo(im_path)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3 style='text-align: center;'>Original Image</h3>", 
                           unsafe_allow_html=True)
                st.image(result["original_im"], use_container_width=True)
            
            with col2:
                st.markdown("<h3 style='text-align: center;'>Predictions</h3>", 
                           unsafe_allow_html=True)
                st.image(result["pred"], use_container_width=True)
            
            num_bboxes  = result['n_bboxes']
            cnt_objects = "objects are" if num_bboxes >= 2 else "object is"
            
            st.markdown(f"<h2 style='text-align: center;'>{num_bboxes} {cnt_objects} detected.</h2>",
                       unsafe_allow_html=True)

        else:
            st.warning("Please select or upload an image.")

    def handle_video_mode(self, yolo_infer):
        st.sidebar.subheader(self.LANGUAGES[self.lang_code]["video_save_path"])
        save_dir = st.sidebar.text_input("Output Directory", "results/videos")
        os.makedirs(save_dir, exist_ok=True)

        # Video selection/upload
        sample_vid_dir = "sample_videos"
        os.makedirs(sample_vid_dir, exist_ok=True)
        sample_videos = glob(os.path.join(sample_vid_dir, "*.mp4"))

        if not sample_videos:
            # Add sample video handling logic if needed
            st.warning("No sample videos found in sample_videos directory")
            sample_videos = []

        selected_video = st.selectbox("Select sample video", sample_videos)
        uploaded_video = st.file_uploader("Or upload video", type=["mp4", "avi", "mov"])

        video_path = uploaded_video if uploaded_video else selected_video

        if video_path:
            # Save uploaded video to temp file
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name

            # Process video
            output_path = self.process_video(yolo_infer, video_path)

            st.subheader("Processed Video")
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)  # This ensures immediate playback[4][6]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument("--device", type=str, default="cpu", help="CPU | GPU")
    parser.add_argument("--model_name", type=str, default="yolo11n.pt", help="Model architecture from ultralytics")        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    available_datasets = [os.path.basename(res).split("_cls_names")[0].split("_")[0] for res in glob(f"saved_cls_names/*.pkl")]
    # available_datasets = [os.path.splitext(os.path.basename(res))[0] for res in glob(f"results/videos/*.mp4")]
    ds_nomi = st.sidebar.selectbox("Choose Dataset", options=available_datasets, index=0)
    model_name = st.sidebar.text_input("Model name", value=args.model_name)    
    device = args.device if args.device == "cpu" else [0]    

    app = StreamlitApp(
        ds_nomi=ds_nomi,
        model_name=model_name,
        device=device
    )
    app.run()
