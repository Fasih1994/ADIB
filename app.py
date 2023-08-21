import time
import os 

from dotenv import load_dotenv
import gradio as gr

from model import get_agent, ask_agent

load_dotenv()



with gr.Blocks() as demo:
    with gr.Row(): 
        with gr.Column():
            img1 = gr.Image("images/adib.jpeg", shape=(10,10), show_label=False, show_download_button=False)
        with gr.Column():
            mrk = gr.Markdown("""
                # This is a ADIB AI
                ## It answers user's query against bank's data.
            """,)
        with gr.Column():pass
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear",variant='stop',size='sm')

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        agent = get_agent(chat_history=history)
        result = ask_agent(agent=agent,query=history[-1][0])
        bot_message = result
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.02)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch(server_name='0.0.0.0', server_port=int(os.environ['PORT']))