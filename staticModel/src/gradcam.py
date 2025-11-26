# src/gradcam.py
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2, numpy as np, torch
from torchvision import transforms
from PIL import Image

def make_gradcam(model, target_layer, img_path, device='cuda'):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485,0.456,0.406],
                                                          std=[0.229,0.224,0.225])])
    inp = transform(img).unsqueeze(0).to(device)
    # prepare visualization image (float RGB 0..1)
    vis_img = np.array(img.resize((224,224))).astype(np.float32)/255.0

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=='cuda'))
    grayscale_cam = cam(input_tensor=inp)[0]  # HxW
    cam_image = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
    out_path = img_path.replace('.png','_gradcam.png')
    cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    return out_path
