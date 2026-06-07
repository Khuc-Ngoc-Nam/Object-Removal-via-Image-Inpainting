# Object Removal via Image Inpainting (Module 1: Intelligent Scissors MVP)

This project is an object removal system that extracts and removes objects from images. Instead of manually painting over an object, we use the **Intelligent Scissors (Magnetic Lasso)** algorithm to quickly and precisely outline the object using automatic boundary snapping.

Currently, the project is in the MVP stage, focusing on **Module 1**: A web interface interacting with the real-time Magnetic Lasso tool.

## Project Structure
- `main.py` & `lasso_core.py`: Backend handling the Intelligent Scissors algorithm and WebSocket communication using Python (FastAPI + OpenCV).
- `frontend/`: Web interface built with Next.js + React.

## System Requirements
- **Python:** 3.10 or higher.
- **Node.js:** 18 or higher.
- A modern web browser.

---

## MVP Setup & Run Instructions

To run the project, you need to open **2 separate Terminal windows** running in parallel: one for the Backend and one for the Frontend.

### 1. Start the Backend (FastAPI)

Open a Terminal in the project's root directory, install the required dependencies, and start the server:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server (runs on port 8000 by default)
uvicorn main:app --reload
```
*Note: If you are using a Virtual Environment (venv), remember to activate it (`source venv/bin/activate`) before running the server.*

### 2. Start the Frontend (Next.js)

Open a new Terminal window, navigate to the `frontend` directory, and start the Node server:

```bash
cd frontend

# Install NPM packages
npm install

# Start the Web interface
npm run dev
```

---

## MVP Demo Experience
1. Open your browser and go to: **[http://localhost:3000](http://localhost:3000)**.
2. Upload any image from your computer.
3. Left-click on a point along the boundary of the object you want to remove to drop the **Start Anchor**.
4. Slowly move your mouse along the object's boundary. The line will **automatically snap** to the edge thanks to the shortest-path algorithm (Dijkstra) running on the image.
5. Left-click to add more Anchor Points to fix the drawing path if necessary.
6. Once you have outlined the object, move your mouse back near the starting point until a green circle appears. Click to **Close the Loop**.
7. As soon as the loop is closed, the backend will automatically process and generate a **Binary Mask** to be used as input for **Module 2 (Image Inpainting)**.
