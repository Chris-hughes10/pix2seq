#!/bin/bash
# Exit on error and undefined vars
set -e
set -u

# Ensure this runs with bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script requires bash. Please run with: bash $0"
    exit 1
fi

# COCO 2017 dataset downloader
# Downloads and organizes COCO dataset including images, labels, and annotations

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# Configuration
BASE_DIR="${SCRIPT_DIR}/../../data"
IMAGES_DIR="${BASE_DIR}/coco/images"
ANNOTATIONS_DIR="${BASE_DIR}/coco/annotations"

# URLs
LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0"
IMAGES_URL="http://images.cocodataset.org/zips"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations"

# Files to download
declare -A files=(
    ["labels"]="coco2017labels.zip"
    ["train"]="train2017.zip"
    ["val"]="val2017.zip"
    ["test"]="test2017.zip"
    ["annotations"]="annotations_trainval2017.zip"
)

# Create necessary directories
mkdir -p "${IMAGES_DIR}" "${ANNOTATIONS_DIR}"

# Function to download and extract files
download_and_extract() {
    local url=$1
    local file=$2
    local dest_dir=$3
    local description=$4

    echo "Downloading ${description} (${file})..."
    if curl -L "${url}/${file}" -o "${file}"; then
        echo "Extracting ${file}..."
        if unzip -q "${file}" -d "${dest_dir}"; then
            rm "${file}"
            echo "Successfully processed ${description}"
            return 0
        else
            echo "Error extracting ${file}"
            return 1
        fi
    else
        echo "Error downloading ${file}"
        return 1
    fi
}

# Download and process files in parallel with proper error handling
{
    # Download labels
    download_and_extract "${LABELS_URL}" "${files[labels]}" "${BASE_DIR}" "COCO labels" &

    # Download images (train and validation)
    download_and_extract "${IMAGES_URL}" "${files[train]}" "${IMAGES_DIR}" "training images" &
    download_and_extract "${IMAGES_URL}" "${files[val]}" "${IMAGES_DIR}" "validation images" &

    # Optionally download test images (commented out by default)
    # download_and_extract "${IMAGES_URL}" "${files[test]}" "${IMAGES_DIR}" "test images" &

    # Download annotations
    download_and_extract "${ANNOTATIONS_URL}" "${files[annotations]}" "$(dirname "${ANNOTATIONS_DIR}")" "annotations" &
} 2>&1 | tee download_log.txt

# Wait for all background processes to complete
wait

# Verify downloads
echo "Verifying downloaded files..."
required_dirs=(
    "${IMAGES_DIR}/train2017"
    "${IMAGES_DIR}/val2017"
    "${ANNOTATIONS_DIR}"
)

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Directory $dir not found. Download may have failed."
        exit 1
    fi
done

echo "Download completed successfully!"
echo "Log file saved as download_log.txt"