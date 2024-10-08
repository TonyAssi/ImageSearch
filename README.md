# Image Search with Text Prompt
by [Tony Assi](https://www.tonyassi.com/)

Use text prompt to search for images in folders. It can search through folders within folders. Built with 🤗 Transformers and 🤗 Datasets.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Import the module
```python
from ImageSearch import ImageSearch
```
Define folder path
- **image_dir** Parent folder
- **hf_key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
image_search = ImageSearch(image_dir='images',
			   hf_key='HF_KEY')
```
The images will be converted to a 🤗 Dataset and it'll get uploaded to the 🤗 Hub. It'll print out the dataset id. It should look like [tonyassi/images-ds](https://huggingface.co/datasets/tonyassi/images-ds). 

Search for image with text prompt
- **text** Text prompt
- **download_path** Images most similar to text prompt will be downloaded to this path (if download_path='' then images will not be downloaded)
- **num** Number of images (optional) defaults to 5
```python
image_search.search(text='red rose',
		    download_path = 'found_images',
		    num=5)
```
The most similar images will be printed and downloaded.
