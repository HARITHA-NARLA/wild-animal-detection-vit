import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import cv2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=11,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load("animal_model.pth", map_location=device))
model.to(device)
model.eval()

# ✅ Your exact class order
classes = [
    "bear", "cheetah", "crocodile", "elephant", "fox",
    "giraffe", "gorilla", "lion", "rhino", "tiger", "zebra"
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Alert system
def get_alert(animal):
    if animal in ["lion", "tiger"]:
        return "HIGH ALERT 🚨"
    elif animal in ["bear", "rhino"]:
        return "MEDIUM ALERT ⚠️"
    elif animal == "Unknown":
        return "UNCERTAIN ⚠️"
    else:
        return "LOW RISK ✅"

# Prediction function (UPDATED)
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted = torch.max(probs, 1)

    # ✅ Confidence threshold
    if confidence.item() < 0.6:
        return "Unknown", confidence.item()

    return classes[predicted.item()], confidence.item()

# Upload image
def upload_image():
    path = input("Enter image path: ")
    
    try:
        image = Image.open(path).convert("RGB")
    except:
        print("Invalid image path ❌")
        return
    
    animal, conf = predict_image(image)
    
    print(f"\nDetected: {animal}")
    print(f"Confidence: {conf:.2f}")
    print(get_alert(animal))

# Camera detection
def camera_detection():
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to capture")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    animal, conf = predict_image(image)

    print(f"\nDetected: {animal}")
    print(f"Confidence: {conf:.2f}")
    print(get_alert(animal))

# Menu
print("\n--- Wild Animal Detection System ---")
print("1. Upload Image")
print("2. Open Camera")

choice = input("Choose option: ")

if choice == "1":
    upload_image()
elif choice == "2":
    camera_detection()
else:
    print("Invalid choice")