import numpy as np
import cv2
import torch
import os
import onnxruntime as ort
from insightface.utils import face_align
from insightface.model_zoo import RetinaFace, ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESOLUTION = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTION_MODEL_PATH = "det.onnx"
RECOGNITION_MODEL_PATH = "rec.onnx"
BINARY_MODEL_PATH = "binary_encoder_best.onnx"
BATCH_SIZE = 32

class BinaryEncoderNetwork(torch.nn.Module):
    def __init__(self, input_dim=512, output_dim=RESOLUTION):
        super(BinaryEncoderNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def get_onnx_provider():
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        logger.info("Using CUDA for inference")
        return ['CUDAExecutionProvider']
    logger.info("CUDA not available, using CPU for inference")
    return ['CPUExecutionProvider']

def load_detection_model(model_path=DETECTION_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detection model not found at {model_path}")
    
    providers = get_onnx_provider()
    
    session = PickableInferenceSession(model_path, providers=providers)
    model = RetinaFace(model_path, session=session)
    logger.info(f"Detection model loaded from {model_path}")
    return model

def load_recognition_model(model_path=RECOGNITION_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Recognition model not found at {model_path}")
    
    providers = get_onnx_provider()
    session = PickableInferenceSession(model_path, providers=providers)
    model = ArcFaceONNX(model_path, session=session)
    logger.info(f"Recognition model loaded from {model_path}")
    return model

def load_binary_encoder(model_path=BINARY_MODEL_PATH):
    class ONNXBinaryEncoder(BinaryEncoderNetwork):
        def __init__(self, session, output_dim=RESOLUTION):
            super().__init__(input_dim=512, output_dim=output_dim)
            self.session = session
            self._is_eval = True
            self._device = torch.device(DEVICE)
            
        def eval(self):
            self._is_eval = True
            return self
            
        def to(self, device):
            self._device = device
            return self
            
        def forward(self, x):
            # Handle input conversion
            if isinstance(x, torch.Tensor):
                x_numpy = x.detach().cpu().numpy()
            else:
                x_numpy = x
                
            # Run inference
            outputs = self.session.run(
                None, 
                {"input": x_numpy}
            )
            
            # Convert back to PyTorch tensor and send to the right device
            return torch.tensor(outputs[0]).to(self._device)
    
    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)
    
    # Create and return the wrapper model
    model = ONNXBinaryEncoder(ort_session)
    
    return model

def try_detect_face(img, detection_model):
    try:
        bboxes, kpss = detection_model.detect(img, max_num=1, input_size=(640, 640))
        if bboxes is None or len(bboxes) == 0:
            return None
            
        kps = None
        if kpss is not None and len(kpss) > 0:
            kps = kpss[0]
            
        aligned_face = face_align.norm_crop(img, landmark=kps, image_size=112)
        return aligned_face
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return None

def batch_detect_faces(images, detection_model):
    if not len(images):
        return []
    
    all_faces = []
    for img in images:
        if img is None or img.size == 0:
            continue
            
        face = try_detect_face(img, detection_model)
        if face is not None:
            all_faces.append(face)
    
    return all_faces

def get_face_embedding(face, recognition_model):
    try:
        embedding = recognition_model.get_feat(face)
        return embedding[0]
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        return None

def batch_get_embeddings(faces, recognition_model):
    if not len(faces):
        return []
    
    all_embeddings = []
    for face in faces:
        if face is None or face.size == 0:
            continue
            
        embedding = get_face_embedding(face, recognition_model)
        if embedding is not None:
            all_embeddings.append(embedding)
    
    return all_embeddings

def batch_binarize_embeddings(embeddings, binary_model):
    if not len(embeddings):
        return []
    
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        binary_codes = binary_model(embeddings_tensor)
        binary_codes = (binary_codes > 0.5).int()
        
    return binary_codes.cpu().numpy().tolist()

def process_batch(images, detection_model, recognition_model, binary_model):
    faces = batch_detect_faces(images, detection_model)
    if not len(faces):
        return []
        
    embeddings = batch_get_embeddings(faces, recognition_model)
    if not len(embeddings):
        return []
        
    binary_codes = batch_binarize_embeddings(embeddings, binary_model)
    return binary_codes

def extract_frames_from_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return []
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
            
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    return frames