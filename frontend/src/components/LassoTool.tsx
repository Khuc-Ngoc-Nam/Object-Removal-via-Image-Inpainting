"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';

type Point = { x: number; y: number };

export default function LassoTool() {
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [allPoints, setAllPoints] = useState<Point[]>([]);
  const [frozenContours, setFrozenContours] = useState<Point[][]>([]);
  const [dynamicContour, setDynamicContour] = useState<Point[]>([]);
  const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);

  // Refs for stateless access inside callbacks
  const dynamicContourRef = useRef<Point[]>([]);
  const lastPtRef = useRef<Point | null>(null);
  const isWaitingRef = useRef(false);
  const pendingMoveRef = useRef<{x: number, y: number} | null>(null);

  // Constants
  const SNAP_MARGIN = 15;
  const COOLING_THRESHOLD = 60;
  const FREEZE_MARGIN = 30;

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        setImageObj(img);
      };
      img.src = url;
    }
  }, [file]);

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageObj) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageObj, 0, 0, canvas.width, canvas.height);

    if (frozenContours.length > 0 || dynamicContour.length > 0) {
      ctx.beginPath();
      
      let hasStarted = false;

      frozenContours.forEach(segment => {
        if (segment.length > 0) {
          if (!hasStarted) {
            ctx.moveTo(segment[0].x, segment[0].y);
            hasStarted = true;
          } else {
            ctx.lineTo(segment[0].x, segment[0].y);
          }
          for (let i = 1; i < segment.length; i++) {
            ctx.lineTo(segment[i].x, segment[i].y);
          }
        }
      });
      
      if (dynamicContour.length > 0 && isDrawing) {
        if (!hasStarted) {
          ctx.moveTo(dynamicContour[0].x, dynamicContour[0].y);
          hasStarted = true;
        } else {
          ctx.lineTo(dynamicContour[0].x, dynamicContour[0].y);
        }
        for (let i = 1; i < dynamicContour.length; i++) {
          ctx.lineTo(dynamicContour[i].x, dynamicContour[i].y);
        }
      } else if (!isDrawing && frozenContours.length > 0) {
        // Close the visual loop
        const firstSeg = frozenContours[0];
        if (firstSeg && firstSeg.length > 0) {
          ctx.lineTo(firstSeg[0].x, firstSeg[0].y);
        }
      }

      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if (allPoints.length > 0) {
      ctx.fillStyle = '#FF0000';
      allPoints.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      if (allPoints.length > 2 && isDrawing) {
        ctx.beginPath();
        ctx.arc(allPoints[0].x, allPoints[0].y, 15, 0, 2 * Math.PI);
        ctx.strokeStyle = '#0000FF';
        ctx.stroke();
      }
    }
  }, [allPoints, dynamicContour, frozenContours, imageObj, isDrawing]);

  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas, dimensions]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const res = await fetch("http://localhost:8000/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        setSessionId(data.session_id);
        setDimensions({ width: data.width, height: data.height });

        const ws = new WebSocket(`ws://localhost:8000/ws/lasso/${data.session_id}`);
        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          
          if (msg.type === "map_built") {
             // We could do something here
          } else if (msg.type === "contour") {
            isWaitingRef.current = false;
            
            // Check for pending move
            if (pendingMoveRef.current) {
              const { x, y } = pendingMoveRef.current;
              pendingMoveRef.current = null;
              sendMove(x, y);
            }

            const points = msg.points as Point[];
            
            if (points.length > COOLING_THRESHOLD) {
              const freezeIndex = points.length - FREEZE_MARGIN;
              const newSeed = points[freezeIndex];
              const frozenSegment = points.slice(0, freezeIndex + 1);
              
              setFrozenContours(prev => [...prev, frozenSegment]);
              
              setAllPoints(prev => {
                const nextPoints = [...prev, newSeed];
                lastPtRef.current = newSeed;
                ws.send(JSON.stringify({ type: "anchor", x: newSeed.x, y: newSeed.y }));
                return nextPoints;
              });
              
              const remaining = points.slice(freezeIndex);
              dynamicContourRef.current = remaining;
              setDynamicContour(remaining);
            } else {
              dynamicContourRef.current = points;
              setDynamicContour(points);
            }
          } else if (msg.type === "mask_generated") {
            setMaskUrl(msg.image);
            setIsDrawing(false);
            setDynamicContour([]);
            dynamicContourRef.current = [];
          }
        };
        wsRef.current = ws;
      } catch (err) {
        console.error("Upload failed", err);
      }
    }
  };

  const getMousePos = (evt: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !dimensions) return { x: 0, y: 0, x_snapped: false, y_snapped: false };
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let x = (evt.clientX - rect.left) * scaleX;
    let y = (evt.clientY - rect.top) * scaleY;

    let x_snapped = false;
    let y_snapped = false;

    if (x < SNAP_MARGIN) { x = 0; x_snapped = true; }
    else if (x > dimensions.width - 1 - SNAP_MARGIN) { x = dimensions.width - 1; x_snapped = true; }

    if (y < SNAP_MARGIN) { y = 0; y_snapped = true; }
    else if (y > dimensions.height - 1 - SNAP_MARGIN) { y = dimensions.height - 1; y_snapped = true; }

    return { x: Math.round(x), y: Math.round(y), x_snapped, y_snapped };
  };

  const sendMove = (x: number, y: number) => {
    if (!wsRef.current || !lastPtRef.current) return;
    isWaitingRef.current = true;
    wsRef.current.send(JSON.stringify({ 
      type: "move", 
      start_x: lastPtRef.current.x, 
      start_y: lastPtRef.current.y, 
      x, 
      y 
    }));
  };

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!wsRef.current || !dimensions) return;
    const { x, y } = getMousePos(e);

    if (!isDrawing) {
      setIsDrawing(true);
      const newPt = { x, y };
      setAllPoints([newPt]);
      lastPtRef.current = newPt;
      wsRef.current.send(JSON.stringify({ type: "anchor", x, y }));
    } else {
      const startPt = allPoints[0];
      const distToStart = Math.hypot(x - startPt.x, y - startPt.y);

      if (distToStart < 15 && allPoints.length > 2) {
        // Close loop
        const finalSegment = dynamicContourRef.current;
        setFrozenContours(prev => {
          const finalFrozen = [...prev, finalSegment];
          const flatPoints = finalFrozen.flat();
          wsRef.current?.send(JSON.stringify({ 
            type: "close", 
            points: flatPoints 
          }));
          return finalFrozen;
        });
        
        dynamicContourRef.current = [];
        setDynamicContour([]);
      } else {
        // Manual anchor
        const finalSegment = dynamicContourRef.current;
        setFrozenContours(prev => [...prev, finalSegment]);
        dynamicContourRef.current = [];
        setDynamicContour([]);

        const newPt = { x, y };
        setAllPoints(prev => {
          const next = [...prev, newPt];
          lastPtRef.current = newPt;
          return next;
        });
        wsRef.current.send(JSON.stringify({ type: "anchor", x, y }));
      }
    }
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !wsRef.current || allPoints.length === 0 || !dimensions) return;
    const { x, y, x_snapped, y_snapped } = getMousePos(e);
    const lastPt = lastPtRef.current;
    if (!lastPt) return;

    if (x_snapped || y_snapped) {
      let onSame = false;
      if ((x === 0 && lastPt.x === 0) || 
          (x === dimensions.width - 1 && lastPt.x === dimensions.width - 1) ||
          (y === 0 && lastPt.y === 0) || 
          (y === dimensions.height - 1 && lastPt.y === dimensions.height - 1)) {
        onSame = true;
      }
      
      if (!onSame) {
        // Force anchor at boundary intersection
        const finalSegment = dynamicContourRef.current;
        setFrozenContours(prev => [...prev, finalSegment]);
        dynamicContourRef.current = [];
        setDynamicContour([]);

        const newPt = { x, y };
        setAllPoints(prev => {
           const next = [...prev, newPt];
           lastPtRef.current = newPt;
           return next;
        });
        wsRef.current.send(JSON.stringify({ type: "anchor", x, y }));
        return;
      }
    }

    if (isWaitingRef.current) {
      pendingMoveRef.current = { x, y };
    } else {
      sendMove(x, y);
    }
  };

  const resetDrawing = () => {
    setMaskUrl(null); 
    setAllPoints([]);
    setFrozenContours([]);
    setDynamicContour([]);
    dynamicContourRef.current = [];
    lastPtRef.current = null;
    isWaitingRef.current = false;
    pendingMoveRef.current = null;
    setIsDrawing(false);
  };

  const resetAll = () => {
    setSessionId(null); 
    setFile(null); 
    resetDrawing();
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gray-50 min-h-screen text-gray-800">
      <h1 className="text-3xl font-bold mb-4 text-blue-600">Lasso Tool</h1>
      
      {!sessionId && (
        <div className="mb-8 p-6 bg-white rounded-lg shadow-md border border-gray-200">
          <label className="block mb-2 font-semibold">Upload an image to start:</label>
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleUpload}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
        </div>
      )}

      {dimensions && (
        <div className="relative border-4 border-gray-800 rounded-lg shadow-2xl bg-white" style={{ display: 'inline-block' }}>
          <canvas
            ref={canvasRef}
            width={dimensions.width}
            height={dimensions.height}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onContextMenu={(e) => e.preventDefault()}
            className="cursor-crosshair"
            style={{ 
              cursor: 'url("data:image/svg+xml,%3Csvg xmlns=\\\'http://www.w3.org/2000/svg\\\' width=\\\'24\\\' height=\\\'24\\\' viewBox=\\\'0 0 24 24\\\' fill=\\\'none\\\' stroke=\\\'%231d4ed8\\\' stroke-width=\\\'2\\\' stroke-linecap=\\\'round\\\' stroke-linejoin=\\\'round\\\'%3E%3Cpath d=\\\'M7 21S4 18 4 14c0-4 3-7 5-7s6 2 8 5c1.45 2.18 3 3 3 3\\\'/%3E%3Ccircle cx=\\\'10\\\' cy=\\\'5\\\' r=\\\'2\\\'/%3E%3C/svg%3E") 10 5, crosshair',
              maxWidth: "100%", maxHeight: "80vh", objectFit: "contain", display: "block" 
            }}
          ></canvas>
        </div>
      )}

      {isDrawing && (
        <div className="mt-4 text-gray-600 animate-pulse">
          <p>Drawing mode active. Trace the boundary. Click near the start point to close.</p>
        </div>
      )}

      {maskUrl && (
        <div className="mt-8 flex flex-col items-center">
          <h2 className="text-xl font-bold mb-4 text-green-600">Đã khép kín vòng!</h2>
          <p className="text-gray-500 mb-6 text-center max-w-lg">
            Đường bao vật thể đã được xác định. Bạn có thể xoá vật thể này hoặc vẽ lại nếu chưa ưng ý.
          </p>
          
          <div className="flex flex-row gap-4">
            <button 
              onClick={() => alert('Backend Module 2 (Inpainting) sẽ được gọi để xoá vật thể này. Tính năng đang được phát triển!')}
              className="px-6 py-3 bg-red-600 text-white font-bold rounded shadow hover:bg-red-700 transition"
            >
              Xoá vật thể này
            </button>
            <button 
              onClick={resetDrawing}
              className="px-6 py-3 bg-gray-200 text-gray-800 font-semibold rounded shadow hover:bg-gray-300 transition"
            >
              Xoá đường bao và vẽ lại
            </button>
          </div>

          <button 
            onClick={resetAll}
            className="mt-6 text-sm text-blue-600 underline hover:text-blue-800 transition"
          >
            Tải ảnh khác
          </button>
        </div>
      )}
    </div>
  );
}
