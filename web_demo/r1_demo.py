import json
import gradio as gr
from copy import deepcopy
from openai import OpenAI
from css import CSS
from format_output import format_response, inverse_response


def save_log(json_path, messages):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                messages,
                indent=4,
                ensure_ascii=False,
            )
        )


def clear_chat(messages_log, all_usr_message_log, r1=False):
    chatbot = []
    if messages_log:
        ## 保存clear之前的会话
        messages_log = inverse_response(messages_log) if r1 else messages_log
        print(messages_log)
        all_usr_message_log.append({messages_log[1]["content"]:messages_log})
        save_log(history_log_path, all_usr_message_log)

    return chatbot, all_usr_message_log

def user(user_message, system_message, history: list):
    if history:
        messages =  history + [{"role": "user", "content": user_message}]
    else:
        messages =  [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": user_message}]
    return "", messages

def bot(history: list, openai_base_url="", openai_api_key="", max_tokens=2048, temperature=0.95, top_p=0.7, stream=False, r1=False):
    infer_history = deepcopy(history)
    if r1:
        infer_history = inverse_response(infer_history)
     ## 保存当前页面
    save_log(current_log_path, [{infer_history[1]["content"]:infer_history}])
    history.append({"role": "assistant", "content": ""})
    print(infer_history)
    llm = OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url)
    model_name = llm.models.list().data[0].id
    response_llm = llm.chat.completions.create(
        model=model_name,
        messages=infer_history,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream
    )
    if stream:
        result = ""
        for chunk in response_llm:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
                response = format_response(result) if r1 else result
                history[-1] = {"role": "assistant", "content": response}
                yield history
    else:
        response_llm = response_llm.choices[0].message.content
        response = format_response(response_llm) if r1 else response_llm
        history[-1] = {"role": "assistant", "content": response}
        yield history

def webui():

    with gr.Blocks(css=CSS) as demo:

        with gr.Column():
            chatbot = gr.Chatbot(type="messages", height=500) 
            all_messages_log = gr.State([])
            with gr.Row():
                
                with gr.Column():
                    sys_msg = gr.Textbox("", lines=3, placeholder="Enter system message...", label="System Prompt...")
                    with gr.Row():
                        stream_output = gr.Checkbox(value=True, label="Stream Output", info="Whether to enable streaming output?")
                        reasoning_mode = gr.Checkbox(label="Reasoning Mode", info="Whether to start the inference mode (the model needs to support it)")

                    msg = gr.Textbox(placeholder="Enter user message...", label="User Input...", submit_btn=True)
                    clear = gr.Button(variant="primary", value="clear")
  
                with gr.Column():
                    
                    url = gr.Textbox("http://localhost:8001/v1", placeholder="Enter Openai Url", label="Please Enter Openai Url")
                    api_key = gr.Textbox("zjuici", placeholder="Enter Openai Key...", label="Please Enter Openai Key")
                        
                    max_new_tokens = gr.Slider(
                        minimum=100,
                        maximum=8192,
                        value=4096,
                        step=1,
                        label="Max_new_tokens",
                    )
                    top_p = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.7, step=0.01, label="Top_p"
                    )
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.5,
                        value=0.95,
                        step=0.01,
                        label="Temperature",
                    ) 
        msg.submit(user, [msg, sys_msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, url, api_key, max_new_tokens, top_p, temperature, stream_output, reasoning_mode], [chatbot]
        )
        clear.click(clear_chat, [chatbot, all_messages_log, reasoning_mode], [chatbot, all_messages_log], queue=False)

    return demo



if __name__ == "__main__":
    history_log_path = "./log_conversation_history.json"
    current_log_path = "./log_conversation_current.json"
     
    webui().launch(share=True)


# CUDA_VISIBLE_DEVICES=0 vllm serve /home/scb123/HuggingfaceWeight/Qwen2.5-7B-Instruct --dtype float16 --max-model-len 4096 --port 8001
# CUDA_VISIBLE_DEVICES=1 vllm serve /home/scb123/PyProject/LLaMA-Factory/qwen2.5_r1_o1_sft/checkpoint-3403 --dtype float16 --enable-chunked-prefill false --gpu-memo
# ry-utilization 0.9 --max-model-len 8192