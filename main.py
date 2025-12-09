"""Flask web app for scene classification using a Keras model."""

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from flask import Flask, flash, render_template, request
from tensorflow import keras
from PIL import Image, UnidentifiedImageError


# Classes expected by the provided model.
CLASSES = [
	"buildings",
	"forest",
	"glacier",
	"mountain",
	"sea",
	"street",
]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "intel_scene_classifier.keras"


def create_app() -> Flask:
	app = Flask(__name__)
	app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
	app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap

	@lru_cache(maxsize=1)
	def get_model():
		import tensorflow as tf
		# Handle legacy Keras format with batch_shape
		custom_objects = {
			'InputLayer': tf.keras.layers.InputLayer
		}
		try:
			return keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
		except Exception:
			# Fallback: try loading with unsafe deserialization for legacy models
			return keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

	def get_target_size(model) -> Tuple[int, int]:
		shape = getattr(model, "input_shape", None)
		if not shape or len(shape) < 4:
			return (224, 224)
		height, width = shape[1], shape[2]
		if height is None or width is None:
			return (224, 224)
		return (int(height), int(width))

	def preprocess_image(file_storage) -> np.ndarray:
		model = get_model()
		target_size = get_target_size(model)
		image_bytes = file_storage.read()
		try:
			image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
		except UnidentifiedImageError as exc:  # surface a clean error for non-images
			raise ValueError("Unsupported image file.") from exc

		image = image.resize(target_size)
		array = np.asarray(image, dtype="float32") / 255.0
		array = np.expand_dims(array, axis=0)
		return array

	def predict_image(file_storage) -> Dict[str, str]:
		model = get_model()
		processed = preprocess_image(file_storage)
		probabilities = model.predict(processed, verbose=0)[0]
		best_idx = int(np.argmax(probabilities))
		best_label = CLASSES[best_idx]
		best_confidence = float(probabilities[best_idx])
		top_pairs = sorted(
			zip(CLASSES, probabilities.tolist()), key=lambda x: x[1], reverse=True
		)
		top3 = top_pairs[:3]
		return {
			"label": best_label,
			"confidence": f"{best_confidence * 100:.1f}%",
			"top3": top3,
		}

	@app.route("/", methods=["GET", "POST"])
	def index():
		prediction = None
		error = None
		if request.method == "POST":
			file = request.files.get("image")
			if not file or file.filename == "":
				error = "Please choose an image file to upload."
			else:
				try:
					prediction = predict_image(file)
				except ValueError as exc:
					error = str(exc)
				except Exception as e:
					import traceback
					print("ERROR:", str(e))
					traceback.print_exc()
					error = f"Unable to process the image: {str(e)}"

		if error:
			flash(error, "error")

		return render_template("index.html", prediction=prediction, classes=CLASSES)

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
