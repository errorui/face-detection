import os
import requests
from pathlib import Path

def download_image(url, save_path):
    """
    Download an image from a URL and save it to the specified path
    
    Args:
        url (str): URL of the image
        save_path (str): Path where the image will be saved
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Create sample_person directory if it doesn't exist
    sample_dir = Path("sample_person")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample image URLs (these are placeholder URLs - replace with actual URLs)
    # Note: In a real application, you would use your own images or properly licensed images
    sample_images = [
        # These are example URLs that would need to be replaced with actual working URLs
        # for testing purposes
        ("https://example.com/sample1.jpg", "person1_photo1.jpg"),
        ("https://example.com/sample2.jpg", "person1_photo2.jpg"),
        ("https://example.com/sample3.jpg", "person2_photo1.jpg"),
    ]
    
    print("This script would download sample images for testing.")
    print("However, you need to replace the placeholder URLs with actual image URLs.")
    print("Alternatively, manually add your own test images to the sample_person directory.")
    
    # Uncomment the code below after replacing the URLs
    """
    for url, filename in sample_images:
        save_path = sample_dir / filename
        download_image(url, save_path)
    """

if __name__ == "__main__":
    main()
