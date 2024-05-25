import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random


def get_image_metadata(image):
    # Отримання розміру зображення
    image_size = image.shape
    # Отримання кількості каналів
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    # Тип кодування кольорової карти
    color_encoding = 'RGB' if len(image.shape) == 3 else 'Grayscale'

    # Повернення метаданих
    return {'розмір': image_size, 'канали': num_channels, 'кодування': color_encoding}


def crop_patches(crop_ndvi1, crop_ndvi2, crop_patch_size=64, crop_stride=32, num_patches=None):
    crop_patches_ndvi1 = []
    crop_patches_ndvi2 = []
    crop_patches_diff_ndvi = []

    height, width = crop_ndvi1.shape

    center_x = width // 2
    center_y = height // 2

    start_x = center_x - crop_patch_size // 2
    start_y = center_y - crop_patch_size // 2

    for y in range(start_y, height - crop_patch_size + 1, crop_stride):
        for x in range(start_x, width - crop_patch_size + 1, crop_stride):
            patch_ndvi1 = crop_ndvi1[y:y + crop_patch_size, x:x + crop_patch_size]
            patch_ndvi2 = crop_ndvi2[y:y + crop_patch_size, x:x + crop_patch_size]
            if random.choice([True, False]):
                patch_ndvi1 = np.flip(patch_ndvi1, axis=0)
                patch_ndvi2 = np.flip(patch_ndvi2, axis=0)
            if random.choice([True, False]):
                patch_ndvi1 = np.flip(patch_ndvi1, axis=1)
                patch_ndvi2 = np.flip(patch_ndvi2, axis=1)
            crop_patches_ndvi1.append(patch_ndvi1)
            crop_patches_ndvi2.append(patch_ndvi2)
            diff_patch = patch_ndvi2 - patch_ndvi1
            crop_patches_diff_ndvi.append(diff_patch)
            if num_patches is not None and len(crop_patches_ndvi1) >= num_patches:
                return np.array(crop_patches_ndvi1), np.array(crop_patches_ndvi2), np.array(crop_patches_diff_ndvi)

    return np.array(crop_patches_ndvi1), np.array(crop_patches_ndvi2), np.array(crop_patches_diff_ndvi)


def augment_ndvi_patches(augment_patches_ndvi1, augment_patches_ndvi2, augment_patches_diff_ndvi):
    flip_horizontal = random.random() < 0.5
    flip_vertical = random.random() < 0.5
    augmented_patches_ndvi1 = []
    augmented_patches_ndvi2 = []
    augmented_patches_diff_ndvi = []
    for augment_ndvi1, augment_ndvi2, diff_ndvi in zip(augment_patches_ndvi1,
                                                       augment_patches_ndvi2,
                                                       augment_patches_diff_ndvi):
        if flip_horizontal:
            augment_ndvi1 = np.flipud(augment_ndvi1)
            augment_ndvi2 = np.flipud(augment_ndvi2)
            diff_ndvi = np.flipud(diff_ndvi)
        if flip_vertical:
            augment_ndvi1 = np.fliplr(augment_ndvi1)
            augment_ndvi2 = np.fliplr(augment_ndvi2)
            diff_ndvi = np.fliplr(diff_ndvi)
        augmented_patches_ndvi1.append(augment_ndvi1)
        augmented_patches_ndvi2.append(augment_ndvi2)
        augmented_patches_diff_ndvi.append(diff_ndvi)

    return augmented_patches_ndvi1, augmented_patches_ndvi2, augmented_patches_diff_ndvi


def plot_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for j, ax in enumerate(axes.flat):
        ax.imshow(images[j], cmap='viridis', vmin=-1, vmax=1)
        ax.set_title(titles[j])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_images_with_predictions(model, dataloader, date1, time1, lat1, lon1, date2, time2, lat2, lon2):
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            limit = 10
            for k in range(inputs_np.shape[0]):
                if k >= limit:
                    break
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(inputs_np[k, 0])
                plt.title('Input Image 1\nDate: ' + date1 + '\nTime: ' + time1 + '\nLat: ' + str(lat1) + '\nLon: ' + str(lon1))

                plt.subplot(1, 4, 2)
                plt.imshow(inputs_np[k, 1])
                plt.title('Input Image 2\nDate: ' + date2 + '\nTime: ' + time2 + '\nLat: ' + str(lat2) + '\nLon: ' + str(lon2))

                plt.subplot(1, 4, 3)
                plt.imshow(targets_np[k, 0])
                plt.title('Actual Differential NDVI')

                plt.subplot(1, 4, 4)
                plt.imshow(outputs_np[k, 0])
                plt.title('Predicted Differential NDVI')

                meta_data = get_image_metadata(inputs_np[0, 0])

                plt.figtext(0.5, 0.1, f"Розмір: {meta_data['розмір']}\n"
                                        f"Канали: {meta_data['канали']}\n"
                                        f"Кодування: {meta_data['кодування']}",
                            horizontalalignment='center', fontsize=8, color='white',
                            bbox=dict(facecolor='black', alpha=0.5))

                plt.show()


def evaluate_model(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss = val_loss / len(dataloader.dataset)

    return val_loss


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1_encoder_output = self.encoder(x1)
        x2_encoder_output = self.encoder(x2)
        x1_decoder_output = self.decoder(x1_encoder_output)
        x2_decoder_output = self.decoder(x2_encoder_output)
        x = torch.mean(torch.stack([x1_decoder_output, x2_decoder_output]), dim=0)
        return x


def load_band(image_path):
    with rasterio.open(image_path) as src:
        return src.read(1)


def calculate_ndvi(nir_img, red_img):
    nir_img = nir_img.astype(np.float32)
    red_img = red_img.astype(np.float32)
    denominator = nir_img + red_img
    ndvi = np.divide((nir_img - red_img), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    return ndvi


def crop_image(image_path, x, y, w, h):
    with rasterio.open(image_path) as src:
        window = Window(x, y, w, h)
        cropped_img = src.read(window=window)
    return cropped_img


def is_monochrome_image(image):
    unique_values = np.unique(image)
    return len(unique_values) == 1


red_band_path1 = 'E:/VSE/LC08_L1TP_029037_20240507_20240507_02_RT/LC08_L1TP_029037_20240507_20240507_02_RT_B4.TIF'
nir_band_path1 = 'E:/VSE/LC08_L1TP_029037_20240507_20240507_02_RT/LC08_L1TP_029037_20240507_20240507_02_RT_B5.TIF'

red_band_path2 = 'E:/VSE/LC09_L1TP_029037_20240328_20240328_02_T1/LC09_L1TP_029037_20240328_20240328_02_T1_B4.TIF'
nir_band_path2 = 'E:/VSE/LC09_L1TP_029037_20240328_20240328_02_T1/LC09_L1TP_029037_20240328_20240328_02_T1_B5.TIF'

red_band1 = load_band(red_band_path1)
nir_band1 = load_band(nir_band_path1)

red_band2 = load_band(red_band_path2)
nir_band2 = load_band(nir_band_path2)

ndvi1 = calculate_ndvi(nir_band1, red_band1)
ndvi2 = calculate_ndvi(nir_band2, red_band2)

patch_size = 1024
stride = 512
patches_ndvi1, patches_ndvi2, patches_diff_ndvi = crop_patches(ndvi1, ndvi2, patch_size, stride, 4)
patches_ndvi1, patches_ndvi2, patches_diff_ndvi = augment_ndvi_patches(patches_ndvi1, patches_ndvi2, patches_diff_ndvi)

# n_patches = len(patches_ndvi1)
# plot_images(patches_ndvi1[:n_patches], titles=["NDVI 1 Patch"]*n_patches, rows=3, cols=6)
# plot_images(patches_ndvi2[:n_patches], titles=["NDVI 2 Patch"]*n_patches, rows=3, cols=6)
# plot_images(patches_diff_ndvi[:n_patches], titles=["Differential NDVI Patch"]*n_patches, rows=3, cols=6)

total_images = 4
train_size = int(0.25 * total_images)
val_size = total_images - train_size

X_ndvi1_train = np.array(patches_ndvi1[:train_size])
X_ndvi1_val = np.array(patches_ndvi1[train_size:train_size + val_size])

X_ndvi2_train = np.array(patches_ndvi2[:train_size])
X_ndvi2_val = np.array(patches_ndvi2[train_size:train_size + val_size])

y_train = np.array(patches_diff_ndvi[:train_size])
y_val = np.array(patches_diff_ndvi[train_size:train_size + val_size])

X_ndvi1_mean = np.mean(X_ndvi1_train)
X_ndvi1_std = np.std(X_ndvi1_train)
X_ndvi2_mean = np.mean(X_ndvi2_train)
X_ndvi2_std = np.std(X_ndvi2_train)

X_ndvi1_train = (np.array(X_ndvi1_train) - X_ndvi1_mean) / X_ndvi1_std
X_ndvi1_val = (np.array(X_ndvi1_val) - X_ndvi1_mean) / X_ndvi1_std
X_ndvi2_train = (np.array(X_ndvi2_train) - X_ndvi2_mean) / X_ndvi2_std
X_ndvi2_val = (np.array(X_ndvi2_val) - X_ndvi2_mean) / X_ndvi2_std

mask_train = np.array([not is_monochrome_image(img) for img in X_ndvi1_train])

X_ndvi1_train = X_ndvi1_train[mask_train]
X_ndvi2_train = X_ndvi2_train[mask_train]
y_train = y_train[mask_train]

mask_val = np.array([not is_monochrome_image(img) for img in X_ndvi1_val])
X_ndvi1_val = X_ndvi1_val[mask_val]
X_ndvi2_val = X_ndvi2_val[mask_val]
y_val = y_val[mask_val]

train_size = len(X_ndvi1_train)
total_images = train_size

X_ndvi1_train_tensor = torch.Tensor(X_ndvi1_train).unsqueeze(1)
X_ndvi2_train_tensor = torch.Tensor(X_ndvi2_train).unsqueeze(1)
X_train_tensor = torch.cat((X_ndvi1_train_tensor, X_ndvi2_train_tensor), dim=1)
y_train_tensor = torch.Tensor(y_train).unsqueeze(1)

X_ndvi1_val_tensor = torch.Tensor(X_ndvi1_val).unsqueeze(1)
X_ndvi2_val_tensor = torch.Tensor(X_ndvi2_val).unsqueeze(1)
X_val_tensor = torch.cat((X_ndvi1_val_tensor, X_ndvi2_val_tensor), dim=1)
y_val_tensor = torch.Tensor(y_val).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size_train = 64
batch_size_val = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

model_path = 'unet_model.pth'
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(model_path))

visualize_images_with_predictions(model, val_loader, "2024-05-07", "17:13:51 UTC", 33.91658, -99.21669, "2024-03-28", "17:14:34 UTC", 33.91658, -99.21669)

criterion = nn.SmoothL1Loss(beta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 15
val_loss = evaluate_model(model, val_loader, criterion)
print(f"Validation Loss: {val_loss:.4f}")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * total_images
    epoch_loss = running_loss / total_images
    val_loss = evaluate_model(model, val_loader, criterion)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    visualize_images_with_predictions(model, val_loader, "2024-05-07", "17:13:51 UTC", 33.91658, -99.21669, "2024-03-28", "17:14:34 UTC", 33.91658, -99.21669)
model_path = 'unet_model.pth'
torch.save(model.state_dict(), model_path)
