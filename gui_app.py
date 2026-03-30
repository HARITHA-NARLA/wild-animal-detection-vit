import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
import winsound
import timm
import cv2

# ===== DEVICE =====
device = torch.device("cpu")

# ===== CLASS NAMES =====
class_names = [
    'bear', 'cheetah', 'crocodile', 'elephant', 'fox',
    'giraffe', 'gorilla', 'lion', 'rhino', 'tiger', 'zebra'
]

# ===== MODEL (MATCH TRAINING EXACTLY) =====
model = timm.create_model("vit_base_patch16_224", pretrained=False)

model.head = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.head.in_features, 11)
)

model.load_state_dict(torch.load("animal_model.pth", map_location=device))
model.to(device)
model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== ALERT SYSTEM =====
def get_alert(animal):
    if animal in ['lion', 'tiger']:
        return "HIGH ALERT 🚨"
    elif animal in ['bear', 'rhino']:
        return "MEDIUM ALERT ⚠️"
    else:
        return "LOW RISK ✅"

# ===== PREDICTION =====
def predict_image(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probs, 1)

    return class_names[predicted.item()], confidence.item()

# ===== UPLOAD IMAGE =====
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = Image.open(file_path).convert("RGB")

    # Slight brightness boost for better prediction
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)

    animal, conf = predict_image(image)

    result = f"Detected: {animal}\nConfidence: {conf:.2f}\n{get_alert(animal)}"

    messagebox.showinfo("Result", result)
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

# ===== CAMERA =====
def open_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Press Q to Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Brightness fix
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)

    animal, conf = predict_image(image)

    result = f"Detected: {animal}\nConfidence: {conf:.2f}\n{get_alert(animal)}"

    messagebox.showinfo("Result", result)
    winsound.PlaySound("alert.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

# ===== GUI =====
root = tk.Tk()
root.title("Wild Animal Detection")
root.geometry("350x250")

tk.Label(root, text="Wild Animal Detection System", font=("Arial", 14)).pack(pady=20)

tk.Button(root, text="Upload Image", command=upload_image, width=20).pack(pady=10)
tk.Button(root, text="Open Camera", command=open_camera, width=20).pack(pady=10)

root.mainloop()