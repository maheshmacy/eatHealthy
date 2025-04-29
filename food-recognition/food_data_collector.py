import os
import sys
import logging
import requests
import shutil
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load environment variables
load_dotenv()

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_collector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === Configuration ===
class Config:
    SAVE_DIR = Path("dataset")
    RAW_DIR = SAVE_DIR / "raw"
    IMG_PER_CLASS = 50
    IMG_SIZE = 224
    MAX_WORKERS = 4  # For parallel downloads
    TIMEOUT = 10  # Seconds for requests

    # Food categories with variations for better search results
    CUSTOM_DISHES = {
        "pizza": ["pizza", "italian pizza", "homemade pizza"],
        "sushi": ["sushi roll", "japanese sushi", "sushi plate"],
        "hamburger": ["hamburger", "cheeseburger", "beef burger"],
        "pasta": ["pasta dish", "italian pasta", "spaghetti"],
        "tacos": ["tacos", "mexican tacos", "street tacos"],
        "fried_rice": ["fried rice", "chinese fried rice", "asian fried rice"],
        "noodles": ["noodles", "asian noodles", "ramen noodles"]
    }

# === API Configuration ===
class ApiConfig:
    PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
    SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY')
    MEALDB_BASE_URL = "https://www.themealdb.com/api/json/v1/1"
    
    if not all([PEXELS_API_KEY, SPOONACULAR_API_KEY]):
        logger.error("Missing required API keys. Please check your .env file.")
        sys.exit(1)

    HEADERS = {
        "pexels": {"Authorization": PEXELS_API_KEY},
        "spoonacular": {"x-api-key": SPOONACULAR_API_KEY}
    }

# === Utilities ===
class ImageProcessor:
    @staticmethod
    def sanitize(name: str) -> str:
        """Sanitize filename by replacing invalid characters."""
        return ''.join(c if c.isalnum() else '_' for c in name).lower().strip('_')
    
    @staticmethod
    def is_valid_image(img_path: Path) -> bool:
        """Verify if the image is valid and meets minimum requirements."""
        try:
            with Image.open(img_path) as img:
                if img.size[0] < 100 or img.size[1] < 100:
                    return False
                img.verify()  # Verify image integrity
                return True
        except Exception:
            return False

    @staticmethod
    def process_image(img_path: Path) -> bool:
        """Process and standardize the image."""
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize((Config.IMG_SIZE, Config.IMG_SIZE), Image.Resampling.LANCZOS)
                img.save(img_path, "JPEG", quality=95, optimize=True)
            return True
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return False

def download_image(url: str, dest_path: Path) -> bool:
    """Download and process an image from URL."""
    try:
        with requests.get(url, stream=True, timeout=Config.TIMEOUT) as response:
            response.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with dest_path.open('wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            if not ImageProcessor.is_valid_image(dest_path):
                dest_path.unlink()
                return False
                
            return ImageProcessor.process_image(dest_path)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

# === API Clients ===
class ApiClient:
    @staticmethod
    def handle_request(url: str, headers: dict = None, params: dict = None) -> dict:
        """Generic API request handler with retries and error handling."""
        try:
            with requests.Session() as session:
                for attempt in range(3):  # 3 retries
                    try:
                        response = session.get(url, headers=headers, params=params, timeout=Config.TIMEOUT)
                        response.raise_for_status()
                        return response.json()
                    except requests.RequestException as e:
                        if attempt == 2:  # Last attempt
                            raise e
                        continue
        except Exception as e:
            logger.error(f"API request failed for {url}: {e}")
            return {}

class PexelsClient:
    @staticmethod
    def search_images(query: str, count: int = 10) -> List[str]:
        """Search for images on Pexels with error handling."""
        url = "https://api.pexels.com/v1/search"
        params = {"query": query, "per_page": count, "size": "medium"}
        
        results = ApiClient.handle_request(
            url=url,
            headers=ApiConfig.HEADERS["pexels"],
            params=params
        )
        
        return [photo["src"]["medium"] 
                for photo in results.get("photos", []) 
                if "src" in photo and "medium" in photo["src"]]

class MealDBClient:
    @staticmethod
    def fetch_dishes(areas: List[str] = ["American", "Mexican"], limit: int = 5) -> List[Tuple[str, str]]:
        """Fetch dishes from TheMealDB API."""
        all_dishes = []
        for area in areas:
            url = f"{ApiConfig.MEALDB_BASE_URL}/filter.php"
            results = ApiClient.handle_request(url=url, params={"a": area})
            
            meals = results.get("meals", [])[:limit]
            all_dishes.extend([
                (meal["strMeal"], meal["strMealThumb"])
                for meal in meals
                if all(key in meal for key in ["strMeal", "strMealThumb"])
            ])
        return all_dishes

class SpoonacularClient:
    @staticmethod
    def fetch_dishes(limit: int = 5) -> List[Tuple[str, str]]:
        """Fetch dishes from Spoonacular API."""
        url = "https://api.spoonacular.com/recipes/complexSearch"
        params = {
            "apiKey": ApiConfig.SPOONACULAR_API_KEY,
            "number": limit,
            "instructionsRequired": True,
            "addRecipeInformation": True
        }
        results = ApiClient.handle_request(
            url=url,
            headers=ApiConfig.HEADERS["spoonacular"],
            params=params
        )
        return [
            (r["title"], r["image"])
            for r in results.get("results", [])
            if "image" in r and "title" in r
        ]

# === Dataset Organization ===
class DatasetOrganizer:
    def __init__(self):
        self.download_stats = {"success": 0, "failed": 0}

    def download_dish_images(self, dishes: List[Tuple[str, str]]) -> None:
        """Download images with parallel processing and progress tracking."""
        Config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        class_to_images = {}

        # Group URLs by class
        for name, url in dishes:
            class_name = ImageProcessor.sanitize(name)
            if class_name not in class_to_images:
                class_to_images[class_name] = set()
            class_to_images[class_name].add(url)

        # Download images in parallel
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = []
            for class_name, urls in class_to_images.items():
                for i, url in enumerate(list(urls)[:Config.IMG_PER_CLASS]):
                    dest_path = Config.RAW_DIR / f"{class_name}_{i}.jpg"
                    futures.append(
                        executor.submit(download_image, url, dest_path)
                    )

            # Track progress
            for future in tqdm(futures, desc="Downloading images", unit="img"):
                if future.result():
                    self.download_stats["success"] += 1
                else:
                    self.download_stats["failed"] += 1

    def organize_dataset(self) -> None:
        """Organize downloaded images into train/val splits."""
        for split in ["train", "val"]:
            (Config.SAVE_DIR / split).mkdir(parents=True, exist_ok=True)

        # Group files by class
        class_files = {}
        for img_path in Config.RAW_DIR.glob("*.jpg"):
            class_name = "_".join(img_path.stem.split("_")[:-1])
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(img_path)

        # Split and organize
        for class_name, files in class_files.items():
            if len(files) < 2:
                logger.warning(f"Skipping {class_name}: insufficient images ({len(files)})")
                continue

            # Create train/val split
            train_files, val_files = train_test_split(
                files, test_size=0.2, random_state=42
            )

            # Copy files to respective directories
            for split, split_files in [("train", train_files), ("val", val_files)]:
                dest_dir = Config.SAVE_DIR / split / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for file in split_files:
                    shutil.copy2(file, dest_dir / file.name)

        # Log statistics
        self._log_dataset_stats()

    def _log_dataset_stats(self) -> None:
        """Log dataset statistics."""
        stats = {
            "train": {},
            "val": {},
            "total_images": 0
        }

        for split in ["train", "val"]:
            split_dir = Config.SAVE_DIR / split
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    stats[split][class_dir.name] = count
                    stats["total_images"] += count

        logger.info("=== Dataset Statistics ===")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info("Train split:")
        for class_name, count in stats["train"].items():
            logger.info(f"  {class_name}: {count} images")
        logger.info("Validation split:")
        for class_name, count in stats["val"].items():
            logger.info(f"  {class_name}: {count} images")

# === Main ===
def main():
    """Main execution function."""
    logger.info("Starting food image dataset collection")
    all_dishes = []

    # Collect from Pexels
    logger.info("Fetching images from Pexels...")
    pexels = PexelsClient()
    for base_dish, variations in Config.CUSTOM_DISHES.items():
        for query in variations:
            try:
                urls = pexels.search_images(query, count=Config.IMG_PER_CLASS // len(variations))
                all_dishes.extend([(base_dish, url) for url in urls])
            except Exception as e:
                logger.error(f"Failed to fetch from Pexels for '{query}': {e}")

    # Collect from TheMealDB
    logger.info("Fetching from TheMealDB...")
    try:
        mealdb = MealDBClient()
        all_dishes.extend(mealdb.fetch_dishes())
    except Exception as e:
        logger.error(f"MealDB fetch failed: {e}")

    # Collect from Spoonacular
    logger.info("Fetching from Spoonacular...")
    try:
        spoonacular = SpoonacularClient()
        all_dishes.extend(spoonacular.fetch_dishes())
    except Exception as e:
        logger.error(f"Spoonacular fetch failed: {e}")

    # Process and organize dataset
    logger.info(f"Processing {len(all_dishes)} dish instances...")
    organizer = DatasetOrganizer()
    organizer.download_dish_images(all_dishes)
    organizer.organize_dataset()

    logger.info("Dataset creation completed")
    logger.info(f"Download statistics: {organizer.download_stats}")
    logger.info(f"Dataset location: {Config.SAVE_DIR}")

if __name__ == "__main__":
    main()

