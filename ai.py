from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy

# Initialize the model
model = ModelManager(name="lama", device="cpu")  # or "cuda"

# Load image and mask
import cv2
image = cv2.imread("img3.jpeg")
mask = cv2.bitwise_not(cv2.imread("img3_mask.jpeg", cv2.IMREAD_GRAYSCALE))

# Define inpainting config
config = Config(
    ldm_steps=20,
    ldm_sampler="plms",
    hd_strategy=HDStrategy.ORIGINAL,
    hd_strategy_crop_margin=128,  # New required field
    hd_strategy_crop_trigger_size=512,  # New required field
    hd_strategy_resize_limit=1024  # New required field
)

# Run inpainting
result = model(image, mask, config)
