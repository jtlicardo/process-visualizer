import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from supabase import create_client

from logging_utils import delete_files_in_folder
from process_bpmn_data import generate_graph_image, process_text

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

@app.route("/text", methods=["POST"])
def receive_text_input():
    data = request.get_json()
    text = data["text"]
    delete_files_in_folder("./output_logs")
    try:
        output = process_text(text)
    except:
        return jsonify({"error": "Error when processing text"})
    try:
        generate_graph_image(output)
    except:
        return jsonify({"error": "Error when generating graph"})
    
    with open("./src/bpmn.jpeg", 'rb+') as f:
        id = uuid.uuid4()
        supabase.storage().from_('image-bucket').upload(f"bpmn/{id}.jpeg", os.path.abspath("./src/bpmn.jpeg"))

    return jsonify({"status": "success", "id": str(id)})

if __name__ == "__main__":
    app.run()