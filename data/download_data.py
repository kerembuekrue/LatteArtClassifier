from google_images_download import google_images_download

def download_google_images(keywords, limit=10, output_directory='downloads'):
    """
    Downloads images from Google Images based on the provided keywords.

    Args:
        keywords (str): The search term for the images.
        limit (int): The maximum number of images to download.
        output_directory (str): The name of the directory to save the images.
    """
    response = google_images_download.googleimagesdownload()

    arguments = {
        "keywords": keywords,
        "limit": limit,
        "output_directory": output_directory
    }

    try:
        paths, errors = response.download(arguments)
        print("--------------------------------------------------")
        print("Image Download Completed")
        print("--------------------------------------------------")
        print("Downloaded images are in the directory:", paths)
        print("Errors (if any):", errors)
    except Exception as e:
        print("An error occurred:", e)
        print("Please ensure you have the 'google_images_download' library installed.")
        print("You can install it using: pip install google_images_download")

if __name__ == "__main__":
    search_term = "latte art heart"
    number_of_images = 10  # You can change this number
    download_directory = "latte_art_images"

    download_google_images(search_term, limit=number_of_images, output_directory=download_directory)