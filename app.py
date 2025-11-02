import os
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import ast

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

with open("meta/classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

model = models.efficientnet_b0(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("best_food101_efficientnetb0.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

df = pd.read_csv("RecipeNLG_dataset.csv")
df = df.dropna(subset=['title'])

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
title_embeddings = np.load("title_embeddings.npy")
title_embeddings = torch.tensor(title_embeddings, dtype=torch.float32)

def predict_food(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        return class_names[pred.item()]

def keyword_search(food_name):
    name = food_name.replace("_", " ").lower()
    matches = df[df['title'].str.lower().str.contains(name, na=False)]
    if len(matches) > 0:
        return matches.sample(min(5, len(matches)))
    else:
        return None

def semantic_search(food_name, top_k=5):
    query_emb = embed_model.encode(food_name.replace("_", " "), convert_to_tensor=True)
    scores = util.cos_sim(query_emb, title_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    recipes = [df.iloc[idx.item()] for idx in top_results.indices]
    return recipes

def get_recipes(food_name):
    result = keyword_search(food_name)
    if result is not None and len(result) > 0:
        return result
    else:
        print("🔍 Using semantic search...")
        return semantic_search(food_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Predict food
            food_name = predict_food(filepath)
            food_display = food_name.replace("_", " ").title()

            # Get top 5 recipes
            recipes = get_recipes(food_name)

            recipe_list = []
            if isinstance(recipes, pd.DataFrame):
                iter_recipes = recipes.iterrows()
            else:
                iter_recipes = [(0, r) for r in recipes]

            for _, r in iter_recipes:
                # Ingredients
                try:
                    ingredients = ast.literal_eval(r['ingredients'])
                    if isinstance(ingredients, str):
                        ingredients = [ingredients]
                except:
                    ingredients = [str(r['ingredients'])]

                # Directions
                try:
                    directions = ast.literal_eval(r['directions'])
                    if isinstance(directions, str):
                        directions = [directions]
                except:
                    directions = [str(r['directions'])]

                ingredients = [i.strip() for i in ingredients if i.strip()]
                directions = [d.strip().rstrip(",") for d in directions if d.strip()]

                recipe_list.append({
                    "title": r['title'],
                    "ingredients": ingredients,
                    "directions": directions
                })

            return render_template('result.html',
                                   food=food_display,
                                   image=file.filename,
                                   recipes=recipe_list)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
