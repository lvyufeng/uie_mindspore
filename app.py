import gradio as gr
from uie_predictor import UIEPredictor

default_schema = ['时间', '选手', '赛事名称']
ie = UIEPredictor(model='lvyufeng/uie-base', schema_lang='en', schema=default_schema, engine="mindspore",
                  position_prob=0.5, max_seq_len=512, batch_size=1, split_sentence=False, use_fp16=True)

# UGC: Define the inference fn() for your models
def model_inference(schema, text):
    ie.set_schema(eval(schema))
    res = ie(text)
    json_out = {"text": text, "result": res}
    return json_out


def clear_all():
    return None, None, None

def fill_example_data(option, schema, text):
    if option == "比赛":
        schema = gr.Textbox.update("['时间', '选手', '赛事名称']")
        text = gr.Textbox.update("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
    elif option == "医疗":
        schema = gr.Textbox.update("['日期', '患者姓名', '病症']")
        text = gr.Textbox.update("2023年3月15日上午，王小明因发热、咳嗽等症状到医院就诊。")
    elif option == "法律":
        schema = gr.Textbox.update("['案件编号', '被告', '罪名']")
        text = gr.Textbox.update("2023年12月1日，张三因涉嫌盗窃被告上法庭。")
    elif option == "银行":
        schema = gr.Textbox.update("['日期', '账户名', '金额']")
        text = gr.Textbox.update("2023年7月10日，李四在工商银行开设了一张储蓄账户，存入了10000元。")
    elif option == "旅游":
        schema = gr.Textbox.update("['日期', '目的地', '景点']")
        text = gr.Textbox.update("2023年8月20日，小明一家人去北京旅游，参观了故宫和天安门广场。")
    return schema, text

with gr.Blocks() as demo:
    gr.Markdown("UIE")

    with gr.Row():
        choices = ["比赛", "医疗", "法律", "银行", "旅游"]
        option_dropdown = gr.Dropdown(choices=choices, label="Select an option:", value=choices[0])
        with gr.Column(scale=1, min_width=100):
            schema_box = gr.Textbox(
                placeholder="ex. ['时间', '选手', '赛事名称']",
                label="Type any schema you want:",
                lines=2,
                value="['时间', '选手', '赛事名称']")
            text_box = gr.Textbox(
                placeholder="ex. 2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！",
                label="Input Sequence:",
                lines=2,
                value="2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")

    with gr.Row():
        btn1 = gr.Button("Clear")
        btn2 = gr.Button("Submit")
    json_out = gr.JSON(label="Information Extraction Output")


    option_dropdown.change(fn=fill_example_data, inputs=[option_dropdown, schema_box, text_box], outputs=[schema_box, text_box])
    btn1.click(fn=clear_all, inputs=None, outputs=[schema_box, text_box, json_out])
    btn2.click(fn=model_inference, inputs=[schema_box, text_box], outputs=[json_out])

demo.launch()

