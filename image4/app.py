from flask import Flask, render_template, request, send_file, url_for
import aiohttp
import asyncio
import aiofiles
import random
import nltk
from nltk.corpus import wordnet
from PIL import Image
from io import BytesIO
import os

# Download WordNet if not installed
nltk.download('wordnet')

app = Flask(__name__)

# Ensure static/images directory exists
os.makedirs("static/images", exist_ok=True)

# Hugging Face API Key
HUGGINGFACE_API_KEY = "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"

# Hugging Face API URLs
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
PARAPHRASING_API_URL = "https://api-inference.huggingface.co/models/prithivida/parrot_paraphraser_on_T5"

# Headers for API requests
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Function to generate synonyms
def get_synonym(word):
    synonyms = wordnet.synsets(word)
    return synonyms[0].lemmas()[0].name() if synonyms else word

# Function to modify prompt with synonyms
def synonym_replacement(prompt):
    words = prompt.split()
    modified_words = [get_synonym(word) if random.random() > 0.7 else word for word in words]
    return " ".join(modified_words)

# Async function to paraphrase prompt using Hugging Face
async def paraphrase_prompt(session, prompt):
    async with session.post(PARAPHRASING_API_URL, headers=HEADERS, json={"inputs": prompt}) as response:
        if response.status == 200:
            try:
                paraphrased_texts = await response.json()
                first_item = paraphrased_texts[0] if isinstance(paraphrased_texts, list) else None
                if isinstance(first_item, dict) and "generated_text" in first_item:
                    return first_item["generated_text"]
                elif isinstance(first_item, str):
                    return first_item
            except Exception as e:
                print(f"Paraphrase Error: {e}")
        return prompt  # Return original prompt if error occurs

# Async function to generate prompt variations
# Async function to generate prompt variations
async def generate_prompt_variations(prompt, num_variations):
    variations = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_variations):
            choice = random.choice(["synonym", "paraphrase", "both"])
            
            if choice == "synonym":
                new_prompt = synonym_replacement(prompt)
            elif choice == "paraphrase":
                tasks.append(paraphrase_prompt(session, prompt))
                new_prompt = None  # Placeholder to prevent UnboundLocalError
            else:
                paraphrased = await paraphrase_prompt(session, prompt)
                new_prompt = synonym_replacement(paraphrased)

            if new_prompt is not None:  # Append only if new_prompt is assigned
                variations.append(new_prompt)

        # Collect paraphrased prompts
        paraphrased_prompts = await asyncio.gather(*tasks)
        variations.extend(paraphrased_prompts)
        
    return variations[:num_variations]


# Async function to generate images
async def generate_images(prompt_variations, background_style, tone, image_size):
    image_paths = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, modified_prompt in enumerate(prompt_variations):
            seed = random.randint(0, 99999)
            final_prompt = f"{modified_prompt}, background style: {background_style}, tone: {tone}"
            data = {"inputs": final_prompt, "parameters": {"seed": seed}}
            tasks.append(fetch_and_save_image(session, data, i, image_size))

        image_paths = await asyncio.gather(*tasks)
    return [path for path in image_paths if path]

# Async function to fetch and save images
async def fetch_and_save_image(session, data, index, image_size):
    async with session.post(IMAGE_API_URL, headers=HEADERS, json=data) as response:
        if response.status == 200:
            try:
                image_data = BytesIO(await response.read())
                img = Image.open(image_data)
                img = img.resize((image_size, image_size))

                filename = f"static/images/generated_image_{index + 1}.jpg"
                async with aiofiles.open(filename, "wb") as f:
                    await f.write(image_data.getvalue())

                return filename
            except Exception as e:
                print("Error processing image:", e)
    return None  # Return None if image generation fails

@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        user_prompt = request.form['prompt']
        background_style = request.form['background_style']
        tone = request.form['tone']
        num_images = int(request.form['num_images'])
        image_size = int(request.form['image_size'])

        # Generate prompt variations asynchronously
        prompt_variations = await generate_prompt_variations(user_prompt, num_images)

        # Generate images asynchronously
        image_paths = await generate_images(prompt_variations, background_style, tone, image_size)

        return render_template('index.html', images=image_paths)

    return render_template('index.html', images=[])

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join("static/images", filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
