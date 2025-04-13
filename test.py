import requests
import time

# URL of the FastAPI endpoint
url = "https://face-api-295757475593.asia-south1.run.app/recognize-face/"  # Update if you deploy your API elsewhere

# Absolute path to the images you want to upload for testing
image_path = r"C:\Users\Raj Raman\Pictures\Screenshots\Screenshot 2025-01-06 213918.png"
image_einstens = r"C:\Users\Raj Raman\Pictures\Screenshots\Screenshot 2025-01-16 131001.png"
image_manik = r"C:\Users\Raj Raman\Pictures\Screenshots\Screenshot 2025-02-06 223643.png"
image_sarthak = r"C:\Users\Raj Raman\Pictures\Screenshots\Screenshot 2025-02-05 210552.png"
image_pandey = r"C:\Users\Raj Raman\Pictures\Screenshots\Screenshot 2025-03-16 104529.png"
image_raj2=r"C:\Users\Raj Raman\Pictures\Screenshots\Screenshot 2024-12-08 133137.png"
image_singer=r"C:\Users\Raj Raman\Downloads\WhatsApp Image 2025-04-11 at 7.22.06 PM.jpeg"
image_singe2r=r"C:\Users\Raj Raman\Desktop\work\facedetection\sample_person\a.png"
image_singe23r=r"C:\Users\Raj Raman\Downloads\WhatsApp Image 2025-04-11 at 7.22.07 PM (1).jpeg"
image_singe231r=r"C:\Users\Raj Raman\Downloads\WhatsApp Image 2025-04-11 at 7.22.05 PM (1).jpeg"
image_singe2311r=r"C:\Users\Raj Raman\Downloads\WhatsApp Image 2025-04-11 at 7.22.05 PM.jpeg"
test3=r"C:\Users\Raj Raman\Downloads\test3.jpeg"

images = [image_path, image_einstens, image_manik, image_sarthak, image_pandey,image_raj2,image_singer,image_singe2r,image_singe23r,image_singe231r,image_singe2311r,test3]

# Function to handle API requests with retries
def send_request(image):
    with open(image, "rb") as image_file:
        files = {"file": ("a.png", image_file, "image/png")}  # Use the correct MIME type for .png
        
        retries = 3  # Retry limit
        for _ in range(retries):
            try:
                response = requests.post(url, files=files)
                
                if response.status_code == 200:
                    print(f"Prediction result for {image}:", response.json())
                    return response.json()
                else:
                    print(f"Error for {image}: {response.status_code} - {response.json()}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Error while requesting {image}: {e}. Retrying...")
                time.sleep(3)  # Wait for 3 seconds before retrying
        else:
            print(f"Failed to get a valid response for {image} after {retries} attempts.")

# Test the API with multiple images
for image in images:
    send_request(image)
