
from django.shortcuts import render,redirect
import numpy as np
from django.http import HttpResponse
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from home.Unet_model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

#DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE =  torch.device("cpu")
brain_model = torch.load("home/brain_tumour_entire_model.pth")
brain_model.to("cpu")
brain_model.eval() 

skin_model = UNet().to(DEVICE)
skin_model.load_state_dict(torch.load('home/checkpointN20_.pth (1).tar')['state_dict'])
skin_model.eval()

def index(request):
    return render(request,'index.html')
 

def brain_tumour(request):
    if request.method == "POST":
        brain_image = request.FILES['brain_image']
        brain_input = preprocess_image(brain_image)
        x_tensor = torch.from_numpy(brain_input)
        x= x_tensor.to('cpu')
        x = x.float()
        prediction = brain_model.predict(x)
        prediction = torch.where(prediction > 0.5, 1, 0)
        prediction1 = prediction.to('cpu')[0][0]
        mask_array = prediction1.cpu().numpy()
        result = "No Tumour"
        mask_rgb = np.stack((mask_array, mask_array, mask_array), axis=-1)
        overlay_image = image_show(brain_image)  # Create a copy of the original image
        print(overlay_image.shape)
        print(mask_rgb.shape)
        for y in range(overlay_image.shape[0]):
            for x in range(overlay_image.shape[1]):
                if mask_array[y, x] == 1:
                    overlay_image[y, x, :] = [1, 0, 0]
                    result = "Tumour Detected"

        print(overlay_image.shape)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(result, fontsize=16)
        ax[0].imshow(image_show(brain_image))
        ax[0].set_title('Original Image')      
        ax[1].imshow(mask_array)
        ax[1].set_title('Generated Mask')
        ax[2].imshow(overlay_image)
        ax[2].set_title('Generated Mask')
        now = datetime.datetime.now()
        unique_name = f"plot_brain_{now.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"media/{unique_name}")
        #plt.show()
        plot_path = f"/media/{unique_name}"
        return render(request, 'result.html', {'plot_path': plot_path})

    return render(request,'brain_tumour.html')

def image_show(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return image_array


def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:  
        image_array = np.stack((image_array,) * 3, axis=-1)
    image_array = np.expand_dims(image_array.transpose(2, 0, 1), axis=0)  
    return image_array


def skin_lease(request):
    return render(request,'index.html')

def skin_lease(request):
    if request.method == "POST":
        skin_image = request.FILES['skin_image']
        resized_image = preprocess_image(skin_image)
        img_tensor = torch.Tensor(resized_image).to(DEVICE)
        #predicted_array = getpredictionskin.getpred(img_tensor)
        generated_mask = skin_model(img_tensor).squeeze().cpu()
        predicted_array = (generated_mask > 0.5).float().detach().numpy()
        overlay_image = image_show(skin_image)
        result = "No Lease"
        for y in range(overlay_image.shape[0]):
            for x in range(overlay_image.shape[1]):
                if predicted_array[y, x] == 1:
                    overlay_image[y, x, :] = [1, 0, 0]
                    result = "Lease Detected"

        print(overlay_image.shape)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(result, fontsize=16)
        ax[0].imshow(image_show(skin_image))
        ax[0].set_title('Original Image')
        ax[1].imshow(predicted_array)
        ax[1].set_title('Generated Mask')
        ax[2].imshow(overlay_image)
        ax[2].set_title('Generated Mask')
        now = datetime.datetime.now()
        unique_name = f"plot_skin_{now.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"media/{unique_name}")
        #plt.show()
        plot_path = f"/media/{unique_name}"
        return render(request, 'result.html', {'plot_path': plot_path})

    return render(request,'skin_lease.html')


