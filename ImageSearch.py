import os
import shutil
import csv
from PIL import Image
from huggingface_hub import create_repo
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class ImageSearch:
    def __init__(self, image_dir, hf_key="HF_KEY"):
        self.image_dir = image_dir
        self.hf_key = hf_key

        self.dest_folder = './data'
        self.model = SentenceTransformer('clip-ViT-B-32')

        # If dataset doesn't exist then create one
        if not os.path.exists('./meta.text'):
            print('Create a dataset...')
            # Create a folder with all images and a metadata.csv file
            self.copy_and_resize_images_with_metadata()

            # Load dataset
            self.dataset = load_dataset('imagefolder', data_dir=self.dest_folder,  split='train')

            print('Generate embeddings...')
            # Compute embeddings
            self.ds_with_embeddings = self.dataset.map(lambda example: {'embeddings':self.model.encode(example['image'])}, batched=True, batch_size=32)

            print('Create repo...')
            # Create a HF repo for dataset
            self.dataset_id = create_repo(self.image_dir + '-ds', token=self.hf_key, repo_type="dataset").repo_id

            print('Upload dataset to Hugging Face...')
            # Upload dataset to HF
            self.ds_with_embeddings.push_to_hub(self.dataset_id, token=self.hf_key)
            print('Dataset uploaded: ', self.dataset_id)

            # Save file with dataset id
            with open('./meta.text', 'w') as file:
                file.write(self.dataset_id)

            # Remove dest_folder folder
            shutil.rmtree(self.dest_folder)
        else: # If dataset already exists
            # Read the dataset id from the meta.txt file
            with open('./meta.text', 'r') as file:
                self.dataset_id = file.read()
                print(self.dataset_id)
            
        # Load dataset
        self.ds_with_embeddings = load_dataset(self.dataset_id)

        # Faiss index
        self.ds_with_embeddings['train'].add_faiss_index(column='embeddings')



    def copy_and_resize_images_with_metadata(self, max_size=1024):
        # Ensure destination folder exists
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)

        # Define supported image extensions
        image_extensions = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

        # Prepare metadata list
        metadata = [['file_name', 'text']]

        # Walk through all folders and subfolders
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(image_extensions):
                    # Construct full file path
                    full_path = os.path.join(root, file)

                    # Create the new file name by replacing '/' with '*'
                    relative_path = os.path.relpath(full_path, self.image_dir)
                    new_file_name = relative_path.replace(os.sep, '*')

                    # Define destination path for the file
                    dest_path = os.path.join(self.dest_folder, new_file_name)

                    # Copy the image to the destination folder
                    shutil.copy2(full_path, dest_path)

                    # Open the copied image and resize it using thumbnail
                    with Image.open(dest_path) as img:
                        # Resize image using thumbnail while maintaining aspect ratio
                        img.thumbnail((max_size, max_size))
                        
                        # Save the resized image (overwrite the copied file)
                        img.save(dest_path)

                    # Add to metadata
                    metadata.append([new_file_name, relative_path])

        # Write metadata to CSV file in the destination folder
        metadata_csv_path = os.path.join(self.dest_folder, 'metadata.csv')
        with open(metadata_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(metadata)

        

    def search(self, text, download_path = '', num=5):
        # Encode text
        prompt = self.model.encode(text)

        # Get images from text
        scores, retrieved_examples = self.ds_with_embeddings['train'].get_nearest_examples('embeddings', prompt, k=num)

        # Print image paths
        for i in range(len(retrieved_examples['text'])):
            print(retrieved_examples['text'][i])
            print('Score:', scores[i])
            print('Index:', i)
            print('')

        # Download images
        if(download_path != ''):
            print('Downloading images...')
            if not os.path.exists(download_path):
                os.makedirs(download_path)

            for i in range(len(retrieved_examples['image'])):
                retrieved_examples['image'][i].save(download_path + '/' + str(i) + '.png')



