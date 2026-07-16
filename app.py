import gradio as gr
from ultralytics import YOLO
from PIL import Image

deploy_model = YOLO("best_model.pt")

def detect_damage(image, conf):
    results = deploy_model.predict(image, conf=conf, verbose=False)
    annotated = results[0].plot()[..., ::-1]
    names = results[0].names
    counts = {}
    for c in results[0].boxes.cls.tolist():
        label = names[int(c)]
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "No damage detected"
    return Image.fromarray(annotated), summary

demo = gr.Interface(
    fn=detect_damage,
    inputs=[gr.Image(type="pil", label="Upload Road Image"), gr.Slider(0.1, 0.9, value=0.25, label="Confidence Threshold")],
    outputs=[gr.Image(type="pil", label="Detected Damage"), gr.Textbox(label="Detected Classes")],
    title="Road Damage Detection",
)

if __name__ == "__main__":
    demo.launch(share=True)
