from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from typing import Annotated
import cv2
import numpy as np
import base64
import copy
import uvicorn

from app.vision.extractor import process_image, draw_solution
from app.vision.predict import predict_board
from app.core.board import SudokuBoard
from app.core.solver import solve

app = FastAPI(title="Sudoku Solver API")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("frontend/index.html")



@app.post("/solve",responses = {200: {"description": "Returns the solved Sudoku image in Base64 format."}, 400: {"description": "Invalid image file."}, 500: {"description": "Vision processing error."}})
async def solve_sudoku(file: Annotated[UploadFile, File(...)]):
    # Read the uploaded file into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run the Vision Pipeline
    try:
        cells, flat_grid = process_image(img)
        raw_grid = predict_board(cells, model_path="models/sudoku_cnn.onnx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision Error: {str(e)}")

    # Solve the Board
    original_grid = copy.deepcopy(raw_grid)
    board = SudokuBoard(raw_grid)

    if solve(board):
        # 4. Draw the solution and convert it to Base64 for the browser
        final_image = draw_solution(flat_grid, original_grid, board.grid)

        _, buffer = cv2.imencode('.jpg', final_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {"status": "success", "image": img_base64}
    else:
        return {"status": "error", "message": "Could not mathematically solve the board. OCR may have misread a digit."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
