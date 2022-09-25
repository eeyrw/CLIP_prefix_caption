import gradio as gr
from predict_simple import Predictor,WEIGHTS_PATHS

def genTxtFromImage(x,modelList):
    if x and modelList!='':
        text = predictor.predict(x,modelList,True)
    else:
        text = ''
    return text

def refreshWeight(model_list):
    predictor.refreshWeight()
    model_list.choices = list(WEIGHTS_PATHS.keys())

predictor = Predictor()
predictor.setup()

with gr.Blocks() as demo:
    gr.Markdown("Image caption demo")
    with gr.TabItem("Image"):
        with gr.Row():
            image_input = gr.Image(type='pil')
            text_output = gr.Textbox()
        image_button = gr.Button("Refresh weights")
        model_list = gr.Dropdown(list(WEIGHTS_PATHS.keys()))


    image_button.click(refreshWeight, inputs=model_list, outputs=None)
    image_input.change(genTxtFromImage, inputs=[image_input,model_list], outputs=text_output)
    model_list.change(genTxtFromImage, inputs=[image_input,model_list], outputs=text_output)
    
demo.launch()