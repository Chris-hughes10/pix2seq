import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
import lmdb
import numpy as np
import psutil
from torch.utils.data import Dataset
from tqdm import tqdm

CACHE_VERSION = 1.0
MAX_LMDB_SIZE = 1024 * 1024 * 1024 * 100  # 100GB max size
MEMORY_WARNING_THRESHOLD = 90  # Warn at 90% memory usage


def get_hash(dataset):
    """Create a hash of the dataset based on its content."""
    hasher = hashlib.md5()
    hasher.update(str(len(dataset)).encode())

    # Sample up to 100 items
    sample_size = min(100, len(dataset))
    sampled_indices = np.random.choice(
        len(dataset), sample_size, replace=False
    ).tolist()

    for i in sampled_indices:
        item = dataset[i]
        # Hash image shape
        hasher.update(str(item[0].shape).encode())
        # Hash a subset of pixel values
        hasher.update(str(item[0][0, 0]).encode())
        # Hash bounding boxes
        hasher.update(str(item[1]).encode())
        # Hash class ids
        hasher.update(str(item[2]).encode())

    return hasher.hexdigest()


def get_object_size(obj):
    """Estimate the size of an object in bytes."""
    return sys.getsizeof(obj)


class CacheDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        cache_dir: str = "./labels_cache",
        cache_images_to_disk: bool = False,
        max_memory_usage: float = 0.75,
        split: str = "train",
        use_lmdb: bool = True,
        num_workers: int = 4,
    ):
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.labels_cache_path = self.generate_cache_path("labels")
        self.image_cache_path = self.generate_cache_path("images")
        self.max_memory_usage = max_memory_usage
        self.cache_images_to_disk = cache_images_to_disk
        self.use_lmdb = use_lmdb
        self.num_workers = num_workers

        # Initialize LMDB environment if using LMDB
        if use_lmdb:
            self.image_lmdb_path = self.cache_dir / f"{self.split}_images.lmdb"
            self.labels_lmdb_path = self.cache_dir / f"{self.split}_labels.lmdb"
            self.setup_lmdb()

        # Clean up old cache files
        self.cleanup_old_caches()

        # Check memory availability
        self.check_memory_requirements()

        # Cache data based on memory availability
        self.setup_caching()

    def setup_lmdb(self):
        """Initialize LMDB environments."""
        if self.use_lmdb:
            self.images_env = lmdb.open(
                str(self.image_lmdb_path),
                readonly=False,
                map_size=MAX_LMDB_SIZE,
                create=True,
            )
            self.labels_env = lmdb.open(
                str(self.labels_lmdb_path),
                readonly=False,
                map_size=MAX_LMDB_SIZE,
                create=True,
            )

    def check_memory_requirements(self):
        """Check and setup memory requirements."""
        self.available_memory = psutil.virtual_memory().available / (
            1024 * 1024 * 1024
        )  # GB
        self.max_usable_memory = self.available_memory * self.max_memory_usage

        self.estimated_labels_memory_usage = self.estimate_labels_memory_usage(
            sample_size=1000
        )
        print(
            f"Estimated memory usage for labels cache ({self.split}): {self.estimated_labels_memory_usage:.4f} GB"
        )

        self.max_usable_memory -= self.estimated_labels_memory_usage

        self.estimated_images_memory_usage = self.estimate_images_memory_usage()
        print(
            f"Estimated memory usage for image cache ({self.split}): {self.estimated_images_memory_usage:.2f} GB"
        )

    def setup_caching(self):
        """Setup caching based on memory availability."""
        if self.estimated_images_memory_usage <= self.max_usable_memory:
            print(f"Caching labels for {self.split} dataset...")
            self.labels = self.load_or_cache_labels()

            print(f"Caching images for {self.split} dataset...")
            self.cached_images = self.load_or_cache_images()
        else:
            print(
                f"Not enough memory to cache images for {self.split} dataset. Using disk caching."
            )
            if self.use_lmdb:
                self.cache_to_lmdb()
            self.cached_images = None
            self.labels = None

    def generate_cache_path(self, cache_type):
        """Generate a unique cache path for this dataset split and cache type."""
        base_name = f"{type(self.base_dataset).__name__}_{self.split}_{cache_type}"
        dataset_hash = get_hash(self.base_dataset)
        return self.cache_dir / f"{base_name}_{dataset_hash[:8]}.npy"

    def validate_cache(self, cache: Dict) -> bool:
        """Validate cache integrity."""
        if not isinstance(cache, dict):
            return False
        required_keys = {"version", "hash", "labels"}
        if not all(k in cache for k in required_keys):
            return False
        if len(cache["labels"]) != len(self.base_dataset):
            return False
        if cache["hash"] != get_hash(self.base_dataset):
            return False
        return True

    def check_memory_usage(self) -> bool:
        """Monitor memory usage during caching."""
        memory = psutil.virtual_memory()
        if memory.percent > MEMORY_WARNING_THRESHOLD:
            print(f"WARNING: High memory usage detected: {memory.percent}%")
            return False
        return True

    def cleanup_old_caches(self):
        """Remove old cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.npy"):
                if cache_file.stem != self.labels_cache_path.stem:
                    try:
                        cache_file.unlink()
                        print(f"Removed old cache: {cache_file}")
                    except Exception as e:
                        print(f"Failed to remove old cache {cache_file}: {e}")
        except Exception as e:
            print(f"Error during cache cleanup: {e}")

    def cache_to_lmdb(self):
        """Cache dataset to LMDB format."""
        print(f"Caching dataset to LMDB for {self.split}...")

        # Cache labels
        with self.labels_env.begin(write=True) as txn:
            for idx in tqdm(
                range(len(self.base_dataset)), desc="Caching labels to LMDB"
            ):
                _, bboxes, class_ids, image_id, image_hw = self.base_dataset[idx]
                label_data = {
                    "shape": image_hw,
                    "cls": class_ids.reshape(-1, 1),
                    "bboxes": bboxes,
                    "image_id": image_id,
                }
                txn.put(str(idx).encode(), pickle.dumps(label_data))

        # Cache images if requested
        if self.cache_images_to_disk:
            with self.images_env.begin(write=True) as txn:
                for idx in tqdm(
                    range(len(self.base_dataset)), desc="Caching images to LMDB"
                ):
                    image, _, _, _, _ = self.base_dataset[idx]
                    txn.put(f"img_{idx}".encode(), pickle.dumps(image))

    def load_from_lmdb(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Load a single item from LMDB storage."""
        with self.labels_env.begin(write=False) as txn:
            label_data = pickle.loads(txn.get(str(idx).encode()))

        if self.cache_images_to_disk:
            with self.images_env.begin(write=False) as txn:
                image = pickle.loads(txn.get(f"img_{idx}".encode()))
        else:
            image, _, _, _, _ = self.base_dataset[idx]

        return image, label_data

    def estimate_labels_memory_usage(self, sample_size=1000):
        """Estimate the memory usage of the labels before caching."""
        total_size = 0
        sample_size = min(sample_size, len(self.base_dataset))

        for i in range(sample_size):
            _, bboxes, class_ids, image_id, image_hw = self.base_dataset[i]
            label = {
                "shape": image_hw,
                "cls": class_ids.reshape(-1, 1),
                "bboxes": bboxes,
                "image_id": image_id,
            }
            label_size = sum(get_object_size(v) for v in label.values())
            total_size += label_size

        return (
            (total_size / sample_size) * len(self.base_dataset) / (1024 * 1024 * 1024)
        )

    def load_or_cache_labels(self):
        """Load cached labels from disk or cache them if not available."""
        try:
            labels = self.load_cached_labels()
        except Exception as e:
            print(f"WARNING ⚠️ Cache loading failed: {e}")
            print(f"Creating new cache: {self.labels_cache_path}")
            labels = self.cache_labels_in_memory()

        self.save_labels_to_disk(
            labels,
            f"{type(self.base_dataset).__name__}_{self.split}",
            self.labels_cache_path,
        )

        return labels

    def load_cached_labels(self) -> List[Dict]:
        """Load and validate cached labels."""
        cache = np.load(str(self.labels_cache_path), allow_pickle=True).item()
        if not self.validate_cache(cache):
            raise ValueError("Invalid cache file")
        print(
            f"Loaded {len(cache['labels'])} labels from cache: {self.labels_cache_path}"
        )
        return cache["labels"]

    def cache_labels_in_memory(self) -> List[Dict]:
        """Cache labels in memory with parallel processing."""
        labels = []
        n = len(self.base_dataset)
        with tqdm(total=n, desc=f"Caching labels to {self.labels_cache_path}") as pbar:
            for idx in range(n):
                if not self.check_memory_usage():
                    print("WARNING: High memory usage, consider using LMDB storage")
                image, bboxes, class_ids, image_id, image_hw = self.base_dataset[idx]
                labels.append(
                    {
                        "shape": image_hw,
                        "cls": class_ids.reshape(-1, 1),
                        "bboxes": bboxes,
                        "image_id": image_id,
                    }
                )
                pbar.update(1)
        return labels

    def load_single_image(self, idx: int) -> np.ndarray:
        """Load a single image for parallel processing."""
        return self.base_dataset[idx][0]

    def cache_images_in_memory(self):
        """Cache images using parallel processing."""
        print(
            f"Caching images for {self.split} dataset using {self.num_workers} workers..."
        )
        with Pool(self.num_workers) as pool:
            cached_images = list(
                tqdm(
                    pool.imap(self.load_single_image, range(len(self.base_dataset))),
                    total=len(self.base_dataset),
                    desc="Caching images",
                )
            )
        return cached_images

    def save_labels_to_disk(self, labels: List[Dict], prefix: str, path: Path):
        """Save labels cache to disk."""
        cache = {
            "version": CACHE_VERSION,
            "hash": get_hash(self.base_dataset),
            "labels": labels,
        }
        np.save(str(path), cache)
        print(f"{prefix} New cache created: {path}")
        actual_file_size = os.path.getsize(path) / (1024 * 1024 * 1024)
        print(f"Actual disk usage for labels cache: {actual_file_size:.2f} GB")

    def estimate_images_memory_usage(self):
        """Estimate memory usage for image caching."""
        total_size = 0
        sample_size = min(100, len(self.base_dataset))
        for i in range(sample_size):
            image, _, _, _, _ = self.base_dataset[i]
            total_size += image.nbytes
        return (
            (total_size / sample_size) * len(self.base_dataset) / (1024 * 1024 * 1024)
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int]]:
        if self.use_lmdb:
            image, label_data = self.load_from_lmdb(index)
            return (
                image,
                label_data["bboxes"],
                label_data["cls"].squeeze(),
                label_data["image_id"],
                label_data["shape"],
            )
        elif self.cached_images is not None:
            image = self.cached_images[index]
            cached_label = self.labels[index]
            return (
                image,
                cached_label["bboxes"],
                cached_label["cls"].squeeze(),
                cached_label["image_id"],
                cached_label["shape"],
            )
        else:
            return self.base_dataset[index]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __del__(self):
        """Cleanup LMDB environments on deletion."""
        if hasattr(self, "images_env"):
            self.images_env.close()
        if hasattr(self, "labels_env"):
            self.labels_env.close()
