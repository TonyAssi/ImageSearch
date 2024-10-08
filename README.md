# Image Search with Text
by [Tony Assi](https://www.tonyassi.com/)

Use text to search for images in folders. It can search through folder within folders. Built with 🤗 Transformers and 🤗 Datasets.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Import the module
```python
from ImageSearch import ImageSearch
```
Define the model and dataset parameters:
- **image_dir** parent folder
- **hf_key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
```python
image_search = ImageSearch(image_dir='images',
			   hf_key="HF_KEY")
```
Download images from Bing into the './images' folder. It is suggested to manually go through the image folders to make sure there isn't any incorrect images in their respective folders. 
```python
model.download_images()
```
Upload dataset to HuggingFace
```python
model.upload_dataset()
```
Train the model and upload it to HuggingFace
```python
model.train_model()
```

## Model Usage
### Inference API Widget
Go to the model page, which can be found on your HuggingFace page. Drag and drag images onto the Inference API section to test it.

### Python
```python
from transformers import pipeline

pipe = pipeline("image-classification", model="tonyassi/art_classifier")
result = pipe('image.png')

print(result)
```

### JavaScript API
```js
async function query(filename) {
	const data = fs.readFileSync(filename);
	const response = await fetch(
		"https://api-inference.huggingface.co/models/tonyassi/art_classifier",
		{
			headers: { Authorization: "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" },
			method: "POST",
			body: data,
		}
	);
	const result = await response.json();
	return result;
}

query("art.jpg").then((response) => {
	console.log(JSON.stringify(response));
});
```
