from insightface.app import FaceAnalysis

print("Initializing FaceAnalysis...")
try:
    app = FaceAnalysis(name="buffalo_l", allowed_modules=['recognition'], providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Models loaded:", app.models)
except Exception as e:
    import traceback
    traceback.print_exc()
