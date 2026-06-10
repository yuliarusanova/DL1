import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import os
import random
from collections import defaultdict
from glob import glob
import unicodedata
import copy


print(os.getcwd())
print(os.listdir('dl-lab-1-image-classification'))

# важно - зафиксировать все сиды
SEED = 9999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class_to_idx = { "Апельсин": 0, "Бананы": 1, "Груши": 2, "Кабачки": 3, "Капуста": 4, "Картофель": 5, "Киви": 6, "Лимон": 7, "Лук": 8, "Мандарины": 9, "Морковь": 10, "Огурцы": 11, "Томаты": 12, "Яблоки зелёные": 13, "Яблоки красные": 14 }


class MyDataset(Dataset):
    def __init__(self, images_filepaths, name2label, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform
        self.name2label = name2label

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imdecode(np.fromfile(image_filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_name = unicodedata.normalize('NFC', os.path.normpath(image_filepath).split(os.sep)[-3])
        label = self.name2label[class_name]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, label


def train_test_split_inside_subfolders(root_path, train_size=0.8):
    train, test = [], []

    for folder_name in os.listdir(root_path):
        class_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(class_path):
            continue

        for subclass_name in os.listdir(class_path):
            subclass_path = os.path.join(class_path, subclass_name)
            if not os.path.isdir(subclass_path):
                continue
            
            # Собираем фото конкретной подпапки
            images = glob(os.path.join(subclass_path, '*.jpg')) + \
                     glob(os.path.join(subclass_path, '*.png')) + \
                     glob(os.path.join(subclass_path, '*.jpeg'))

            # Делим на части ПРЯМО ТУТ
            random.shuffle(images)
            split_idx = int(train_size * len(images))
            
            train.extend(images[:split_idx])
            test.extend(images[split_idx:])

    # В конце перемешиваем общие списки, чтобы классы не шли по порядку
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test

dataset_path = 'dl-lab-1-image-classification/train/train'
train, test = train_test_split_inside_subfolders(dataset_path)
print(train, test)


from collections import Counter
import os

def count_classes(image_paths, class_to_idx):
    counts = Counter()
    
    for path in image_paths:
        print("Путь:", path)
        print("Разбитый путь:", os.path.normpath(path).split(os.sep))
        class_name = os.path.normpath(path).split(os.sep)[-3].replace('ё', 'ё')
        class_idx = class_to_idx[class_name]
        counts[class_idx] += 1
    return counts

for key in class_to_idx.keys():
    print(repr(key))

train_counts = count_classes(train, class_to_idx)
test_counts  = count_classes(test, class_to_idx)




import matplotlib.pyplot as plt
import numpy as np

idx2class = {v: k for k, v in class_to_idx.items()}

classes = list(idx2class.keys())
class_names = [idx2class[i] for i in classes]


train_values = [train_counts.get(i, 0) for i in classes]
test_values  = [test_counts.get(i, 0) for i in classes]
x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, train_values, width, label='Train')
plt.bar(x + width/2, test_values,  width, label='Test')

plt.xticks(x, class_names, rotation=45, ha='right')
plt.ylabel('Количество изображений')
plt.title('Распределение классов в train и test')
plt.legend()
plt.tight_layout()
plt.show()


import pandas as pd

rows = []

for class_idx, class_name in idx2class.items():
    train_cnt = train_counts.get(class_idx, 0)
    test_cnt  = test_counts.get(class_idx, 0)
    total     = train_cnt + test_cnt

    test_ratio = test_cnt / total if total > 0 else 0

    rows.append({
        "Класс": class_name,
        "Train": train_cnt,
        "Test": test_cnt,
        "Total": total,
        "Test %": round(test_ratio * 100, 2)
    })

df = pd.DataFrame(rows)
df




train_transforms = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])


# Validation transforms - deterministic
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])



train_dataset = MyDataset(images_filepaths=train, name2label=class_to_idx, transform=train_transforms)
test_dataset = MyDataset(images_filepaths=test, name2label=class_to_idx, transform=val_transforms)

# Оптимально для VS Code на Mac
train_loader = DataLoader(
    train_dataset,
    batch_size=64,           
    shuffle=True,
    num_workers=0,        
    pin_memory=False,        
    persistent_workers=False  
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,          
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Используемое устройство: {device}")



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")



model = timm.create_model('convnext_nano', pretrained=True, num_classes=15)

model.to(device)

# Данные из вашей таблицы (столбец Total)
counts = torch.tensor([
    872, 804, 410, 298, 836, 789, 188, 683, 
    665, 788, 623, 595, 764, 849, 725
], dtype=torch.float)

weights = 1.0 / counts

# Нормализуем, чтобы средний вес был равен 1 (стабильнее для градиентов)
weights = weights / weights.sum() * len(counts)
weights = weights.to(device)

# Инициализируем функцию потерь с весами и сглаживанием
loss_fn = torch.nn.CrossEntropyLoss(
    weight=weights, 
    label_smoothing=0.1
)


# выбираем алгоритм оптимизации и learning_rate
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

steps_per_epoch = len(train_loader)

# Инициализируем планировщик
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,             
    epochs=15,               
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,           # 30% времени на разогрев (warmup)
    anneal_strategy='cos'    # Косинусное снижение (очень плавное)
)




import numpy as np
import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, desc="Val"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size

        y_pred = logits.argmax(dim=1)
        total_correct += (y_pred == y_batch).sum().item()
        total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return accuracy, avg_loss


def train_model(model, loss_fn, optimizer, scheduler, train_loader, val_loader, device, writer=None, n_epoch=3):
    num_iter = 0
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(1, n_epoch + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epoch}", leave=True)

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step() # Добавляем сюда!
    

            # накопим метрики для прогресс-бара
            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            y_pred = logits.argmax(dim=1)
            total_correct += (y_pred == y_batch).sum().item()

            avg_loss = total_loss / max(total_samples, 1)
            acc = total_correct / max(total_samples, 1)

            # tqdm live-metrics
            pbar.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{acc:.4f}")
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6e}")

            # логирование (по итерациям)
            num_iter += 1
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), num_iter)
                writer.add_scalar("Accuracy/train", (y_pred == y_batch).float().mean().item(), num_iter)

        # Валидация (тоже с tqdm)
        val_acc, val_loss = evaluate(model, val_loader, loss_fn, device, desc=f"Val {epoch}/{n_epoch}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if writer is not None:
            writer.add_scalar("Loss/val", val_loss, num_iter)
            writer.add_scalar("Accuracy/val", val_acc, num_iter)

        print(f"Epoch {epoch}/{n_epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Обучение завершено. Возвращена модель с лучшей точностью: {best_val_loss:.4f}")

    return model



from sklearn.model_selection import KFold
from torch.utils.data import Subset

# Параметры K-Fold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
indices = list(range(len(train_dataset)))

print(f"Начинаем K-Fold обучение на {n_splits} фолдов...")

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n{'='*30}\nFOLD {fold + 1}/{n_splits}\n{'='*30}")
    
    # 1. Создаем подвыборки для текущего фолда
    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(train_dataset, val_idx)
    
    # 2. Лоадеры (num_workers=0 для стабильности в VS Code на Mac)
    curr_train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=0)
    curr_val_loader = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=0)
    
    # 3. Инициализируем НОВУЮ модель для каждого фолда
    fold_model = timm.create_model('resnet50', pretrained=True, num_classes=15).to(device)
    
    # 4. Настраиваем оптимизатор и планировщик под текущий размер фолда
    fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-3)
    fold_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        fold_optimizer,
        max_lr=1e-3,
        epochs=15, 
        steps_per_epoch=len(curr_train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 5. Обучаем модель (используем вашу функцию train_model)
    # n_epoch=15 для каждого фолда для лучшего результата
    fold_model = train_model(
        model=fold_model,
        loss_fn=loss_fn, # Используем loss_fn с весами из 3-го блока
        optimizer=fold_optimizer,
        scheduler=fold_scheduler,
        train_loader=curr_train_loader,
        val_loader=curr_val_loader,
        device=device,
        n_epoch=15
    )
    
    # 6. Сохраняем веса фолда
    model_filename = f"resnet50_fold_{fold + 1}.pth"
    torch.save(fold_model.state_dict(), model_filename)
    print(f"Фолд {fold + 1} завершен. Модель сохранена в {model_filename}")



import torch.nn.functional as F

# Процесс предсказания ансамблем + TTA
with torch.no_grad():
    for image_id in tqdm(submission["image_id"], desc="Ensemble + TTA Predicting"):
        image_path = os.path.join(test_images_dir, image_id)

        # 1. Загрузка и базовая обработка
        image_raw = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        # 2. Подготовка двух вариантов: оригинал и флип (отзеркаливание)
        image_flipped = cv2.flip(image_raw, 1) # 1 — горизонтальный флип [1.3]

        # Применяем трансформации к обоим вариантам
        img_orig = val_transforms(image=image_raw)["image"].unsqueeze(0).to(device)
        img_flip = val_transforms(image=image_flipped)["image"].unsqueeze(0).to(device)

        # 3. Собираем вероятности от всех 5 моделей для ОБОИХ вариантов
        combined_probs = torch.zeros((1, 15)).to(device)
        
        for m in ensemble_models:
            # Предсказание для оригинала
            logits_orig = m(img_orig)
            combined_probs += torch.softmax(logits_orig, dim=1)
            
            # Предсказание для флипнутой копии
            logits_flip = m(img_flip)
            combined_probs += torch.softmax(logits_flip, dim=1)
        
        # Теперь в combined_probs сумма 10 предсказаний [1.1]
        final_pred = combined_probs.argmax(dim=1).item()
        pred_labels.append(final_pred)

# Сохранение (назовем файл иначе, чтобы сравнить результат)
submission["label"] = pred_labels
submission.to_csv("submission_ensemble_tta.csv", index=False)
