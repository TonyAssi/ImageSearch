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
Define folder path:
- **image_dir** parent folder
- **hf_key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
image_search = ImageSearch(image_dir='images',
			   hf_key="HF_KEY")
```
Search for image with text prompt
- **image_dir** parent folder
- **hf_key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
image_search = ImageSearch(image_dir='images',
			   hf_key="HF_KEY")
```
