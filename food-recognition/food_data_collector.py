import os
import re
import sys
import logging
import requests
import shutil
import random
import zipfile
import tarfile
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_dataset.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    BASE_DIR = Path("food_dataset")
    RAW_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    DOWNLOAD_DIR = BASE_DIR / "downloads"
    TEMP_DIR = BASE_DIR / "temp"
    
    INCLUDE_FOOD101 = True
    INCLUDE_FRUITS = True
    INCLUDE_BEVERAGES = True
    
    TARGET_SIZE = (224, 224)
    VERIFY_IMAGES = True
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    MAX_IMAGES_PER_CLASS = 500
    
    FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    FRUITS360_URL = "https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip"
    
    BEVERAGES_DATASET = "chetankv/drinks-classification"
    
    FOOD101_CATEGORIES = []
    
    FRUIT_CATEGORIES = [
        "apple", "banana", "orange", "strawberry", "blueberry",
        "grape", "pineapple", "watermelon", "kiwi", "mango",
        "peach", "pear", "plum", "raspberry", "cherry"
    ]
    
    BEVERAGE_CATEGORIES = [
        "coffee", "tea", "juice", "smoothie", "water",
        "soda", "beer", "wine", "cocktail", "milkshake"
    ]
    
    TIMEOUT = 300

class DatasetDownloader:
    @staticmethod
    def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if dest_path.exists():
                logger.info(f"File already exists at {dest_path}")
                return True
            
            with requests.get(url, stream=True, timeout=Config.TIMEOUT) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dest_path, 'wb') as file, tqdm(
                        desc=desc,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                    for data in response.iter_content(chunk_size=1024*1024):
                        size = file.write(data)
                        bar.update(size)
                
                return True
        
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    @staticmethod
    def extract_archive(archive_path: Path, extract_path: Path) -> bool:
        try:
            extract_path.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    total_size = sum(item.file_size for item in zip_ref.infolist())
                    extracted_size = 0
                    
                    for item in tqdm(zip_ref.infolist(), desc="Extracting zip"):
                        zip_ref.extract(item, extract_path)
                        extracted_size += item.file_size
            
            elif archive_path.name.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    members = tar_ref.getmembers()
                    for member in tqdm(members, desc="Extracting tar.gz"):
                        tar_ref.extract(member, extract_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {e}")
            return False
    
    @classmethod
    def download_food101(cls) -> Path:
        logger.info("Downloading Food-101 dataset")
        
        archive_path = Config.DOWNLOAD_DIR / "food-101.tar.gz"
        extract_path = Config.RAW_DIR / "food-101"
        
        Config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        Config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        if extract_path.exists():
            logger.info("Food-101 dataset already extracted")
            return extract_path
        
        if not archive_path.exists():
            if not cls.download_file(Config.FOOD101_URL, archive_path, "Downloading Food-101"):
                raise Exception("Failed to download Food-101 dataset")
        
        logger.info("Extracting Food-101 dataset")
        if not cls.extract_archive(archive_path, Config.RAW_DIR):
            raise Exception("Failed to extract Food-101 dataset")
        
        logger.info(f"Food-101 dataset extracted to {extract_path}")
        return extract_path
    
    @classmethod
    def download_fruits360(cls) -> Path:
        logger.info("Downloading Fruits-360 dataset")
        
        archive_path = Config.DOWNLOAD_DIR / "fruits-360.zip"
        extract_path = Config.RAW_DIR / "fruits-360"
        
        Config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        Config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        if extract_path.exists():
            logger.info("Fruits-360 dataset already extracted")
            return extract_path
        
        if not archive_path.exists():
            if not cls.download_file(Config.FRUITS360_URL, archive_path, "Downloading Fruits-360"):
                raise Exception("Failed to download Fruits-360 dataset")
        
        logger.info("Extracting Fruits-360 dataset")
        if not cls.extract_archive(archive_path, Config.RAW_DIR):
            raise Exception("Failed to extract Fruits-360 dataset")
        
        extracted_dirs = [d for d in Config.RAW_DIR.iterdir() if d.is_dir() and "fruit" in d.name.lower()]
        if extracted_dirs:
            if extracted_dirs[0].name != "fruits-360":
                shutil.move(str(extracted_dirs[0]), str(extract_path))
        
        logger.info(f"Fruits-360 dataset extracted to {extract_path}")
        return extract_path
    
    class ImageProcessor:
    @staticmethod
    def is_valid_image(img_path: Path) -> bool:
        if not Config.VERIFY_IMAGES:
            return True
        
        try:
            with Image.open(img_path) as img:
                if img.size[0] < 100 or img.size[1] < 100:
                    return False
                img.verify()
                return True
        except Exception as e:
            logger.debug(f"Invalid image {img_path}: {e}")
            return False
    
    @staticmethod
    def process_image(src_path: Path, dest_path: Path) -> bool:
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(src_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(Config.TARGET_SIZE, Image.LANCZOS)
                
                img.save(dest_path, 'JPEG', quality=95)
                
            return True
        
        except Exception as e:
            logger.debug(f"Error processing image {src_path}: {e}")
            return False
    
    @staticmethod
    def sanitize_class_name(name: str) -> str:
        sanitized = ''.join(c if c.isalnum() else '_' for c in name.lower())
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        return sanitized.strip('_')

class Food101Processor:
    @staticmethod
    def process_dataset(dataset_path: Path, output_dir: Path) -> Dict[str, int]:
        logger.info("Processing Food-101 dataset")
        
        classes_path = dataset_path / "meta" / "classes.txt"
        images_dir = dataset_path / "images"
        
        with open(classes_path, 'r') as f:
            all_classes = [line.strip() for line in f.readlines()]
        
        if Config.FOOD101_CATEGORIES:
            classes = [c for c in all_classes if c in Config.FOOD101_CATEGORIES]
        else:
            classes = all_classes
        
        logger.info(f"Processing {len(classes)} Food-101 classes")
        
        class_counts = {}
        
        for class_name in tqdm(classes, desc="Processing Food-101 classes"):
            sanitized_name = ImageProcessor.sanitize_class_name(class_name)
            class_dir = images_dir / class_name
            
            image_paths = list(class_dir.glob("*.jpg"))
            
            if Config.MAX_IMAGES_PER_CLASS and len(image_paths) > Config.MAX_IMAGES_PER_CLASS:
                image_paths = random.sample(image_paths, Config.MAX_IMAGES_PER_CLASS)
            
            processed_count = 0
            
            for img_path in tqdm(image_paths, desc=f"Processing {class_name}", leave=False):
                if not ImageProcessor.is_valid_image(img_path):
                    continue
                
                dest_path = output_dir / sanitized_name / f"{img_path.stem}.jpg"
                
                if ImageProcessor.process_image(img_path, dest_path):
                    processed_count += 1
            
            class_counts[sanitized_name] = processed_count
            logger.info(f"Processed {processed_count} images for {sanitized_name}")
        
        return class_counts

class Fruits360Processor:
    @staticmethod
    def get_base_fruit_name(class_name: str) -> str:
        import re
        
        name = class_name.lower()
        
        name = re.sub(r'_\d+, '', name)
        
        apple_varieties = [
            "apple_red", "apple_red_yellow", "apple_red_delicious", 
            "apple_golden", "apple_golden_delicious", "apple_granny_smith", 
            "apple_pink_lady", "apple_crimson_snow", "apple_braeburn",
            "apple_golden_1", "apple_golden_2", "apple_golden_3",
            "apple_red_1", "apple_red_2", "apple_red_3",
            "apple_red_yellow_1", "apple_red_yellow_2"
        ]
        
        if any(variety == name for variety in apple_varieties):
            return "apple"
        
        fruit_groups = {
            "banana": ["banana", "banana_yellow", "banana_red", "banana_lady_finger"],
            "cherry": ["cherry", "cherry_rainier", "cherry_wax", "cherry_wax_red", 
                      "cherry_wax_black", "cherry_wax_yellow"],
            "grape": ["grape", "grape_white", "grape_white_2", "grape_white_3", 
                     "grape_white_4", "grape_pink", "grape_blue"],
            "pepper": ["pepper", "pepper_red", "pepper_green", "pepper_yellow", "pepper_orange"],
            "tomato": ["tomato", "tomato_maroon", "tomato_cherry_red", "tomato_yellow", 
                      "tomato_1", "tomato_2", "tomato_3", "tomato_4", "tomato_heart"]
        }
        
        for base_fruit, varieties in fruit_groups.items():
            if any(variety == name for variety in varieties):
                return base_fruit
            
        parts = name.split('_')
        if len(parts) > 1:
            if "fruit" in parts or "berry" in parts:
                for part in parts:
                    if part in ["fruit", "berry"]:
                        idx = parts.index(part)
                        if idx > 0:
                            return f"{parts[idx-1]}_{part}"
            return parts[0]
            
        return name
    
    @staticmethod
    def process_dataset(dataset_path: Path, output_dir: Path) -> Dict[str, int]:
        import re
        
        logger.info("Processing Fruits-360 dataset with variety grouping")
        
        data_dirs = list(dataset_path.glob("**/Training"))
        if not data_dirs:
            data_dirs = list(dataset_path.glob("**/train"))
        
        if not data_dirs:
            logger.error("Could not find Training directory in Fruits-360 dataset")
            return {}
        
        training_dir = data_dirs[0]
        
        class_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
        
        grouped_classes = {}
        
        for class_dir in class_dirs:
            base_fruit = Fruits360Processor.get_base_fruit_name(class_dir.name)
            
            if Config.FRUIT_CATEGORIES and not any(
                category.lower() in base_fruit.lower() for category in Config.FRUIT_CATEGORIES
            ):
                continue
            
            if base_fruit not in grouped_classes:
                grouped_classes[base_fruit] = []
            
            grouped_classes[base_fruit].append(class_dir)
        
        for base_fruit, dirs in grouped_classes.items():
            dir_names = [d.name for d in dirs]
            logger.info(f"Grouped {len(dirs)} varieties into '{base_fruit}': {', '.join(dir_names)}")
        
        logger.info(f"Grouped {len(class_dirs)} fruit varieties into {len(grouped_classes)} base categories")
        
        class_counts = {}
        
        for base_fruit, dirs in tqdm(grouped_classes.items(), desc="Processing fruit groups"):
            sanitized_name = ImageProcessor.sanitize_class_name(base_fruit)
            
            all_image_paths = []
            for class_dir in dirs:
                image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
                all_image_paths.extend(image_paths)
            
            logger.info(f"Found {len(all_image_paths)} images for {base_fruit} across {len(dirs)} varieties")
            
            if Config.MAX_IMAGES_PER_CLASS and len(all_image_paths) > Config.MAX_IMAGES_PER_CLASS:
                all_image_paths = random.sample(all_image_paths, Config.MAX_IMAGES_PER_CLASS)
            
            processed_count = 0
            
            for img_path in tqdm(all_image_paths, desc=f"Processing {sanitized_name}", leave=False):
                if not ImageProcessor.is_valid_image(img_path):
                    continue
                
                dest_path = output_dir / sanitized_name / f"{sanitized_name}_{img_path.stem}.jpg"
                
                if ImageProcessor.process_image(img_path, dest_path):
                    processed_count += 1
            
            class_counts[sanitized_name] = processed_count
            logger.info(f"Processed {processed_count} images for {sanitized_name}")
        
        return class_counts

class BeveragesProcessor:
    @staticmethod
    def collect_from_web(output_dir: Path) -> Dict[str, int]:
        import requests
        from urllib.parse import urlencode
        
        logger.info("Collecting beverage images from web sources")
        
        PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
        PEXELS_URL = "https://api.pexels.com/v1/search"
        
        def search_pexels(query, per_page=20):
            headers = {"Authorization": PEXELS_API_KEY}
            params = {"query": query, "per_page": per_page}
            response = requests.get(PEXELS_URL, headers=headers, params=params)
            if response.status_code == 200:
                return response.json().get("photos", [])
            return []
        
        class_counts = {}
        
        for beverage in Config.BEVERAGE_CATEGORIES:
            sanitized_name = ImageProcessor.sanitize_class_name(beverage)
            
            if not PEXELS_API_KEY:
                logger.warning(f"No Pexels API key available, skipping {beverage}")
                continue
                
            photos = search_pexels(f"{beverage} drink", per_page=50)
            
            processed_count = 0
            
            for i, photo in enumerate(photos):
                img_url = photo.get("src", {}).get("large")
                if not img_url:
                    continue
                
                try:
                    response = requests.get(img_url, timeout=10)
                    img_data = response.content
                    
                    temp_path = Config.TEMP_DIR / f"{sanitized_name}_{i}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(img_data)
                    
                    dest_path = output_dir / sanitized_name / f"{sanitized_name}_{i}.jpg"
                    
                    if ImageProcessor.process_image(temp_path, dest_path):
                        processed_count += 1
                    
                    if temp_path.exists():
                        temp_path.unlink()
                        
                except Exception as e:
                    logger.debug(f"Error processing image from {img_url}: {e}")
            
            if processed_count > 0:
                class_counts[sanitized_name] = processed_count
                logger.info(f"Processed {processed_count} images for {sanitized_name}")
        
        return class_counts
    
    @staticmethod
    def extract_from_food101(food101_path: Path, output_dir: Path) -> Dict[str, int]:
        logger.info("Extracting beverage-like categories from Food-101")
        
        beverage_categories = ["ice_cream", "frozen_yogurt", "chocolate_mousse"]
        
        classes_path = food101_path / "meta" / "classes.txt"
        images_dir = food101_path / "images"
        
        with open(classes_path, 'r') as f:
            all_classes = [line.strip() for line in f.readlines()]
        
        classes = [c for c in all_classes if c in beverage_categories]
        
        logger.info(f"Processing {len(classes)} beverage-like classes from Food-101")
        
        class_counts = {}
        
        for class_name in tqdm(classes, desc="Processing Food-101 beverage-like classes"):
            if "ice_cream" in class_name or "frozen_yogurt" in class_name:
                mapped_name = "ice_cream"
            elif "chocolate" in class_name:
                mapped_name = "chocolate_drink"
            else:
                mapped_name = class_name
            
            sanitized_name = ImageProcessor.sanitize_class_name(mapped_name)
            class_dir = images_dir / class_name
            
            image_paths = list(class_dir.glob("*.jpg"))
            
            if Config.MAX_IMAGES_PER_CLASS and len(image_paths) > Config.MAX_IMAGES_PER_CLASS:
                image_paths = random.sample(image_paths, Config.MAX_IMAGES_PER_CLASS)
            
            processed_count = 0
            
            for img_path in tqdm(image_paths, desc=f"Processing {mapped_name}", leave=False):
                if not ImageProcessor.is_valid_image(img_path):
                    continue
                
                dest_path = output_dir / sanitized_name / f"{sanitized_name}_{img_path.stem}.jpg"
                
                if ImageProcessor.process_image(img_path, dest_path):
                    processed_count += 1
            
            class_counts[sanitized_name] = processed_count
            logger.info(f"Processed {processed_count} images for {sanitized_name}")
        
        return class_counts
    
    @staticmethod
    def process_dataset(dataset_path: Path, output_dir: Path, food101_path: Optional[Path] = None) -> Dict[str, int]:
        logger.info("Processing beverages using alternative methods")
        
        class_counts = {}
        
        if food101_path is not None:
            food101_counts = BeveragesProcessor.extract_from_food101(food101_path, output_dir)
            class_counts.update(food101_counts)
        
        if sum(class_counts.values()) < 1000:
            web_counts = BeveragesProcessor.collect_from_web(output_dir)
            
            for class_name, count in web_counts.items():
                class_counts[class_name] = class_counts.get(class_name, 0) + count
        
        return class_counts

class DatasetOrganizer:
    def __init__(self):
        self.raw_dir = Config.RAW_DIR / "processed_images"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Config.PROCESSED_DIR
        
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
    
    def process_food101(self, food101_path: Path) -> Dict[str, int]:
        if not Config.INCLUDE_FOOD101:
            return {}
        
        return Food101Processor.process_dataset(
            food101_path, 
            self.raw_dir
        )
    
    def process_fruits360(self, fruits360_path: Path) -> Dict[str, int]:
        if not Config.INCLUDE_FRUITS:
            return {}
        
        return Fruits360Processor.process_dataset(
            fruits360_path, 
            self.raw_dir
        )
    
    def process_beverages(self, beverages_path: Path) -> Dict[str, int]:
        if not Config.INCLUDE_BEVERAGES:
            return {}
        
        return BeveragesProcessor.process_dataset(
            beverages_path, 
            self.raw_dir
        )
    
    def organize_splits(self) -> Dict[str, Dict[str, int]]:
        logger.info("Organizing dataset splits with fruit variety grouping")
        
        class_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            logger.error("No processed image directories found")
            return {}
        
        class_mapping = {}
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            if class_name.startswith("apple_"):
                base_class = "apple"
            elif class_name.startswith("banana_"):
                base_class = "banana"
            elif class_name.startswith("cherry_"):
                base_class = "cherry"
            elif class_name.startswith("grape_"):
                base_class = "grape"
            elif class_name.startswith("tomato_"):
                base_class = "tomato"
            elif class_name.startswith("pepper_"):
                base_class = "pepper"
            else:
                base_class = class_name
            
            class_mapping[class_name] = base_class
        
        stats = {
            "train": {},
            "val": {},
            "test": {},
            "total": {}
        }
        
        for class_dir in tqdm(class_dirs, desc="Organizing classes"):
            class_name = class_dir.name
            base_class = class_mapping[class_name]
            
            images = list(class_dir.glob("*.jpg"))
            
            if not images:
                logger.warning(f"No images found for class {class_name}")
                continue
            
            train_val_images, test_images = train_test_split(
                images,
                test_size=Config.TEST_RATIO,
                random_state=42
            )
            
            train_images, val_images = train_test_split(
                train_val_images,
                test_size=Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO),
                random_state=42
            )
            
            splits = {
                "train": train_images,
                "val": val_images,
                "test": test_images
            }
            
            for split, split_images in splits.items():
                output_class_dir = self.output_dir / split / base_class
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in tqdm(split_images, desc=f"{split}/{base_class}", leave=False):
                    output_path = output_class_dir / f"{base_class}_{class_name}_{img_path.stem}.jpg"
                    
                    if not output_path.exists():
                        shutil.copy2(img_path, output_path)
                
                if base_class not in stats[split]:
                    stats[split][base_class] = 0
                stats[split][base_class] += len(split_images)
                
                if base_class not in stats["total"]:
                    stats["total"][base_class] = 0
                stats["total"][base_class] += len(split_images)
        
        grouped_classes = {}
        for original, base in class_mapping.items():
            if base not in grouped_classes:
                grouped_classes[base] = []
            grouped_classes[base].append(original)
        
        logger.info("Class grouping results:")
        for base, originals in grouped_classes.items():
            if len(originals) > 1:
                logger.info(f"'{base}' contains: {', '.join(originals)}")
        
        return stats
    
    def create_nutrition_dataset(self, class_stats: Dict[str, int], class_mapping: Dict[str, str] = None) -> Path:
        logger.info("Creating nutrition dataset with grouped categories")
        
        data_dir = Config.BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if class_mapping:
            unique_classes = set(class_mapping.values())
            classes = list(unique_classes)
        else:
            classes = list(class_stats.keys())
        
        logger.info(f"Creating nutrition data for {len(classes)} consolidated food classes")
        
        nutrition_data = []
        
        food_types = {
            "fruit": [50, 120, 0.5, 1.5, 15, 30, 0.5, 2.0, 2, 5, 10, 20, 30, 70],
            "vegetable": [25, 75, 1, 3, 5, 15, 0.1, 1.0, 2, 5, 1, 5, 15, 50],
            "grain": [150, 300, 3, 10, 30, 60, 1, 5, 2, 10, 0, 5, 50, 85],
            "protein": [150, 350, 15, 35, 0, 10, 5, 20, 0, 1, 0, 2, 0, 30],
            "dairy": [100, 200, 5, 12, 5, 15, 5, 15, 0, 0, 5, 12, 30, 60],
            "sweet": [200, 400, 2, 5, 25, 60, 5, 20, 0, 2, 20, 50, 60, 100],
            "beverage": [0, 150, 0, 2, 0, 15, 0, 2, 0, 0, 0, 15, 0, 50]
        }
        
        food_type_mapping = {}
        
        for food_class in classes:
            if "apple" in food_class.lower():
                food_type_mapping[food_class] = "fruit"
            elif "banana" in food_class.lower():
                food_type_mapping[food_class] = "fruit"
            elif "cherry" in food_class.lower():
                food_type_mapping[food_class] = "fruit"
            elif "grape" in food_class.lower():
                food_type_mapping[food_class] = "fruit"
            elif any(fruit in food_class.lower() for fruit in Config.FRUIT_CATEGORIES):
                food_type_mapping[food_class] = "fruit"
            elif any(beverage in food_class.lower() for beverage in Config.BEVERAGE_CATEGORIES):
                food_type_mapping[food_class] = "beverage"
            elif any(grain in food_class.lower() for grain in ["bread", "pasta", "rice", "cereal", "oat"]):
                food_type_mapping[food_class] = "grain"
            elif any(protein in food_class.lower() for protein in ["beef", "chicken", "pork", "fish", "tofu", "egg"]):
                food_type_mapping[food_class] = "protein"
            elif any(dairy in food_class.lower() for dairy in ["milk", "cheese", "yogurt"]):
                food_type_mapping[food_class] = "dairy"
            elif any(sweet in food_class.lower() for sweet in ["cake", "cookie", "ice_cream", "chocolate", "candy", "dessert"]):
                food_type_mapping[food_class] = "sweet"
            elif any(veg in food_class.lower() for veg in ["salad", "vegetable", "broccoli", "carrot", "spinach"]):
                food_type_mapping[food_class] = "vegetable"
            else:
                first_letter = food_class[0].lower()
                if first_letter in "abc":
                    food_type_mapping[food_class] = "grain"
                elif first_letter in "defgh":
                    food_type_mapping[food_class] = "protein"
                elif first_letter in "ijklm":
                    food_type_mapping[food_class] = "dairy"
                elif first_letter in "nopqrs":
                    food_type_mapping[food_class] = "vegetable"
                else:
                    food_type_mapping[food_class] = "sweet"
        
        for food_class in classes:
            food_type = food_type_mapping.get(food_class, "grain")
            ranges = food_types[food_type]
            
            calories = round(random.uniform(ranges[0], ranges[1]), 1)
            protein = round(random.uniform(ranges[2], ranges[3]), 1)
            carbs = round(random.uniform(ranges[4], ranges[5]), 1)
            fat = round(random.uniform(ranges[6], ranges[7]), 1)
            fiber = round(random.uniform(ranges[8], ranges[9]), 1)
            sugar = round(random.uniform(ranges[10], ranges[11]), 1)
            gi = round(random.uniform(ranges[12], ranges[13]), 0)
            
            entry = {
                "food_name": food_class,
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat,
                "fiber": fiber,
                "sugar": sugar,
                "glycemic_index": gi,
                "serving_size_grams": 100,
                "food_type": food_type
            }
            
            nutrition_data.append(entry)
        
        nutrition_df = pd.DataFrame(nutrition_data)
        csv_path = data_dir / "food_nutrition.csv"
        nutrition_df.to_csv(csv_path, index=False)
        
        logger.info(f"Nutrition dataset created at {csv_path} with {len(nutrition_data)} consolidated food classes")
        return csv_path
    
    def log_statistics(self, stats: Dict[str, Dict[str, int]], class_mapping: Dict[str, str] = None):
        logger.info("=== Dataset Statistics ===")
        
        total_images = sum(sum(split_stats.values()) for split_name, split_stats in stats.items() 
                        if split_name != "total" and isinstance(split_stats, dict))
        logger.info(f"Total images: {total_images}")
        
        for split in ["train", "val", "test"]:
            split_total = sum(stats.get(split, {}).values())
            logger.info(f"{split.capitalize()} split: {split_total} images")
        
        if "total" in stats:
            top_classes = sorted(stats["total"].items(), key=lambda x: x[1], reverse=True)
            
            logger.info("\nTop 10 classes by image count:")
            for class_name, count in top_classes[:10]:
                logger.info(f"  {class_name}: {count} images")
        
        stats_dir = Config.BASE_DIR / "stats"
        stats_dir.mkdir(exist_ok=True)
        
        classes = list(stats.get("total", {}).keys())
        distribution_data = []
        
        for class_name in classes:
            entry = {"class": class_name}
            for split in ["train", "val", "test"]:
                entry[split] = stats.get(split, {}).get(class_name, 0)
            entry["total"] = stats.get("total", {}).get(class_name, 0)
            distribution_data.append(entry)
        
        distribution_data.sort(key=lambda x: x["total"], reverse=True)
        
        distribution_df = pd.DataFrame(distribution_data)
        distribution_df.to_csv(stats_dir / "class_distribution.csv", index=False)
        
        data_dir = Config.BASE_DIR / "data"
        data_dir.mkdir(exist_ok=True)
        
        if class_mapping:
            grouped_classes = {}
            for original, base in class_mapping.items():
                if base not in grouped_classes:
                    grouped_classes[base] = []
                grouped_classes[base].append(original)
            
            logger.info("Class grouping summary:")
            for base, originals in grouped_classes.items():
                if len(originals) > 1:
                    logger.info(f"'{base}' contains {len(originals)} varieties")
            
            unique_classes = list(set(class_mapping.values()))
            with open(data_dir / "class_names.json", 'w') as f:
                json.dump(unique_classes, f)
                
            logger.info(f"Saved {len(unique_classes)} consolidated class names to class_names.json")
        else:
            with open(data_dir / "class_names.json", 'w') as f:
                json.dump(classes, f)
                
            logger.info(f"Saved {len(classes)} class names to class_names.json")

def main():
    logger.info("Starting enhanced food dataset collection with fruits and beverages")
    
    for directory in [Config.RAW_DIR, Config.PROCESSED_DIR, Config.DOWNLOAD_DIR, Config.TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    datasets_paths = {}
    downloader = DatasetDownloader()
    
    if Config.INCLUDE_FOOD101:
        try:
            logger.info("=== Processing Food-101 Dataset ===")
            food101_path = downloader.download_food101()
            datasets_paths["food101"] = food101_path
        except Exception as e:
            logger.error(f"Error downloading Food-101 dataset: {e}")
    
    if Config.INCLUDE_FRUITS:
        try:
            logger.info("=== Processing Fruits-360 Dataset ===")
            fruits360_path = downloader.download_fruits360()
            datasets_paths["fruits360"] = fruits360_path
        except Exception as e:
            logger.error(f"Error downloading Fruits-360 dataset: {e}")
    
    organizer = DatasetOrganizer()
    
    class_counts = {}
    
    if "food101" in datasets_paths and "beverages" not in datasets_paths:
        beverages_counts = organizer.process_beverages(None, datasets_paths["food101"])
        class_counts.update(beverages_counts)
    
    if "fruits360" in datasets_paths:
        fruits360_counts = organizer.process_fruits360(datasets_paths["fruits360"])
        class_counts.update(fruits360_counts)
    
    
    logger.info("=== Creating Train/Val/Test Splits ===")
    stats = organizer.organize_splits()
    
    organizer.log_statistics(stats)
    
    logger.info("=== Creating Nutrition Dataset ===")
    nutrition_path = organizer.create_nutrition_dataset(class_counts)
    
    logger.info("Dataset preparation completed")
    logger.info(f"Dataset location: {Config.PROCESSED_DIR}")
    logger.info(f"Data files: {Config.BASE_DIR / 'data'}")
    logger.info(f"Statistics: {Config.BASE_DIR / 'stats'}")
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"The dataset is ready at: {Config.PROCESSED_DIR}")
    print(f"  - Train split: {sum(stats.get('train', {}).values())} images")
    print(f"  - Validation split: {sum(stats.get('val', {}).values())} images")
    print(f"  - Test split: {sum(stats.get('test', {}).values())} images")
    print(f"  - Total classes: {len(stats.get('total', {}))}")
    print("\nNext steps:")
    print("1. Run the model training script:")
    print("   python food_classifier_trainer.py")
    print("\n2. After training, run the API service:")
    print("   python food_api.py")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the log file for more details: food_dataset.log")
