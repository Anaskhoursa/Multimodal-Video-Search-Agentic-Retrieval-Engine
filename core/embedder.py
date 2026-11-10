import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List

class VisionEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        # Select GPU if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing CLIP Embedder on device: {self.device}")
        
        # Load the pre-trained CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True, weights_only=False).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Put model in evaluation mode (disables dropout, etc.)
        self.model.eval()

    def generate_frame_embedding(self, image_path: str) -> List[float]:
        """Loads a single extracted frame image and converts it into a 512-dimensional vector embedding."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return []

        # Preprocess the image and move tensors to target device
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Extract vision features from the CLIP transformer
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize vectors to unit length (Crucial for cosine similarity matching later)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Convert tensor matrix into a standard flat Python float list
            embedding = image_features.cpu().numpy().flatten().tolist()
            
        return embedding