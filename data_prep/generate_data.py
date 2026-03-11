import pandas as pd
import os
import json
import shutil
import torch
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# НАСТРОЙКИ ПУТЕЙ
INPUT_CSV = r'D:\University\Sem6\NIRS\data\articles.csv'
IMAGES_DIR = r'D:\University\Sem6\NIRS\data\images'
OUTPUT_DIR = '../web_app/public'
SUBSET_SIZE = 5000 

print("--- Инициализация системы ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

if os.path.exists(OUTPUT_DIR):
    print(f"Очистка старых данных в {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)

# 1. Загрузка CSV
df = pd.read_csv(INPUT_CSV, dtype={'article_id': str})

# 2. Поиск доступных картинок
print("Проверка наличия файлов на диске...")
def get_path(aid):
    return os.path.join(IMAGES_DIR, aid[:3], f"{aid}.jpg")

# Быстрая проверка существования
df['exists'] = df['article_id'].apply(lambda x: os.path.exists(get_path(x)))
df_available = df[df['exists'] == True].copy()
print(f"Доступно товаров с картинками: {len(df_available)}")

# 3. Формирование выборки 
groups = df_available['product_group_name'].unique()
per_group = SUBSET_SIZE // len(groups)
subset_dfs = []

for group in groups:
    group_df = df_available[df_available['product_group_name'] == group]
    count = min(len(group_df), per_group)
    if count > 0:
        subset_dfs.append(group_df.sample(count))

df_combined = pd.concat(subset_dfs)

# Добираем остаток
remaining_needed = SUBSET_SIZE - len(df_combined)
if remaining_needed > 0:
    others = df_available.drop(df_combined.index)
    extra = others.sample(min(len(others), remaining_needed))
    df_combined = pd.concat([df_combined, extra])

df_subset = df_combined
print(f"Итоговый размер выборки: {len(df_subset)}")

# 4. Модель CLIP
print("\nЗагрузка модели CLIP...")
model = SentenceTransformer('clip-ViT-B-32', device=device)

final_data = []

# 5. Обработка
print(f"Начало генерации эмбеддингов ...")
for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    article_id = row['article_id']
    image_path = get_path(article_id)
    
    try:
        img_obj = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            img_emb = model.encode(img_obj).tolist()
        
        shutil.copy(image_path, os.path.join(OUTPUT_DIR, 'images', f"{article_id}.jpg"))
        
        final_data.append({
            "id": article_id,
            "name": row['prod_name'],
            "category": row['product_group_name'],
            "image": f"{article_id}.jpg",
            "vector": img_emb
        })
    except:
        continue

# 6. Сохранение
with open(os.path.join(OUTPUT_DIR, 'data.json'), 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False)

print(f"\nГотово! Сохранено {len(final_data)} объектов.")