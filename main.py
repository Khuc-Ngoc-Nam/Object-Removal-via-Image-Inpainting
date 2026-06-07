from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import base64
import cv2
import numpy as np
from lasso_core import LassoCore

app = FastAPI()

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for simplicity
# In production, use Redis or database
sessions = {}

class InitSessionResponse(BaseModel):
    session_id: str
    width: int
    height: int

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Generate unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    # Initialize Lasso Core
    core = LassoCore()
    w, h = core.apply_image(contents)
    
    # Store session
    sessions[session_id] = core
    
    return InitSessionResponse(session_id=session_id, width=w, height=h)

@app.websocket("/ws/lasso/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in sessions:
        await websocket.close(code=1008)
        return
        
    core = sessions[session_id]
    
    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            action = data.get("type")
            
            if action == "anchor":
                # User clicked to drop an anchor
                pt = [data["x"], data["y"]]
                core.build_map(pt)
                await websocket.send_json({"type": "map_built", "x": pt[0], "y": pt[1]})
                
            elif action == "move":
                # User moved mouse, request smart contour
                start_pt = [data["start_x"], data["start_y"]]
                end_pt = [data["x"], data["y"]]
                
                contour_points = core.get_smart_contour(start_pt, end_pt)
                await websocket.send_json({"type": "contour", "points": contour_points})
                
            elif action == "close":
                # User closed the loop, generate binary mask
                raw_points = data.get("points", [])
                if len(raw_points) > 2:
                    mask = core.generate_mask(raw_points)
                    # For demo purposes, we will return the mask as a base64 encoded image
                    _, buffer = cv2.imencode('.png', mask)
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_json({"type": "mask_generated", "image": f"data:image/png;base64,{b64_img}"})
                    
    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected")
        # Cleanup
        if session_id in sessions:
            del sessions[session_id]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
