"""Load trained weights once and predict CIFAR-10 classes."""
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError
import torch

from app.cnn_model import AssignmentCNN, CLASSES, image_transform

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "cifar10_cnn.pt"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class ImageClassifier:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("classes") != list(CLASSES):
            raise ValueError("Checkpoint has incompatible class labels.")
        if checkpoint.get("preprocessing") != "resize64_to_tensor_v1":
            raise ValueError("Checkpoint has incompatible preprocessing.")
        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = image_transform()

    def predict(self, data: bytes):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise ValueError("Upload a PNG or JPEG image.")
                    image.load()
                    tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("The file is not a readable, safely sized PNG or JPEG image.") from exc
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(probabilities.argmax())
        return {"class_id": index, "class_name": CLASSES[index],
                "confidence": float(probabilities[index]),
                "probabilities": dict(zip(CLASSES, probabilities.tolist()))}
