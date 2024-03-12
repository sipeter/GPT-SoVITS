'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, sys
now_dir = os.getcwd()
sys.path.append(now_dir)

import os, re, logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import numpy as np
import librosa
import torchaudio
import time


infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
        
import gradio as gr
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}

cut_method = {
    i18n("不切"):"cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path
    
tts_pipline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path

def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket
              ):
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False
    }
    
    for item in tts_pipline.run(inputs):
        yield item
        
def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names

SoVITS_names, GPT_names = get_weights_names()

# 原代码基础上增加快捷选取参考音频功能和保存参考音频功能

### ---增加快捷选取参考音频功能  --开始--

    # 初始化引导音频列表

def replace_chinese(text):
    pattern = r'([\u4e00-\u9fa5]{10}).*'
    result = re.sub(pattern, r'\1...', text)
    return result

# 初始化引导音频列表
def init_wav_list(sovits_path):
    wav_path = "./output/slicer_opt"
    match = re.search(r'([a-zA-Z0-9\-_]+)_e\d+_s\d+\.pth', sovits_path)
    if match:
        result = match.group(1)
        wav_path = f"./logs/{result}/5-wav32k/"
    else:
        return ["无参考音频"], {}

    res_wavs = {}

    res_text = ["请选择参考音频"]

    # 读取文本
    text = ""
    try:
       with open(rf'./logs/{result}/2-name2text.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("无参考音频")
        return ["无参考音频"], {}
    with open(rf'./logs/{result}/2-name2text.txt', 'r', encoding='utf-8') as f:
        text = f.read()


    # 遍历目录
    # 增加计数功能，使得每次显示的音频不超过50个，防止页面卡顿   
    count = 0
    for file_path in os.listdir(wav_path):
        # 检查当前file_path是否为文件
        if os.path.isfile(os.path.join(wav_path, file_path)):
            # 将文件名添加到列表中
            match = re.search(rf'{file_path}\t(.+?)\t(.+?)\t(.+?)\n', text)
            if match:
                # 提取匹配到的内容
                extracted_text = match.group(3)
                # print(extracted_text)

                # 传入音频文件路径，获取音频数据和采样率
                audio_data, sample_rate = librosa.load(f'./logs/{result}/5-wav32k/{file_path}')
                # 使用librosa.get_duration函数计算音频文件的长度
                duration = librosa.get_duration(y=audio_data, sr=sample_rate)
                duration = int(duration)

                 # 只添加3到10秒之间的音频
                if 3 <= duration <= 10:
                    key = f"{replace_chinese(extracted_text)}_{duration}秒"
                    res_text.append(key)
                    res_wavs[key] = (f'./logs/{result}/5-wav32k/{file_path}', extracted_text)
                    count += 1
                    if count >= 50:
                        break


            else:
                print("No match found")

    return res_text, res_wavs

    # 切换参考音频


def change_wav(audio_name):
    first_key = list(reference_dict.keys())[0]

    try:
        value = reference_dict[audio_name]
        return value[0], value[1]
    except Exception as e:
        return reference_dict[first_key][0], reference_dict[first_key][1]

### ---增加快捷选取参考音频功能 --结束--
    
# 
reference_wavs,reference_dict = init_wav_list(sovits_path)

#添加音频历史记录--开始--
output_history =[]
history_max_num = 20

def sync_output_history_to_checkbox_audio():
    checkbox_result = []
    audio_result = []
    for item in output_history:
        label = item['label']
        if len(label)>15:
            label=label[:15]+'...'
        checkbox_result.append(gr.update(label=label,value=False))
        audio_result.append(gr.update(value=item['value']))
    for _ in range(len(audio_result),history_max_num):
        checkbox_result.append(gr.update(label="",value=False))
        audio_result.append(gr.update(value = None))
    return [*checkbox_result,*audio_result]

def add_to_history(audio,input_text):
    if(audio is None or audio[1] is not None):
        if len(output_history) == history_max_num:
            output_history.pop()
        output_history.insert(0,{'value':audio,'label':input_text})

    return [*sync_output_history_to_checkbox_audio()]

def clear_history():
    output_history = []
    checkbox_result = []
    audio_result = []
    for _ in range(history_max_num):
        checkbox_result.append(gr.update(label="",value=False))
        audio_result.append(gr.update(value = None))
    return [*checkbox_result,*audio_result]

def shown_audio_num_change(audio_num):
    audio_num = int(audio_num)
    audio_result = []
    checkbox_result = []
    for _ in range(audio_num):
        audio_result.append(gr.update(visible=True))
        checkbox_result.append(gr.update(visible=True))
    for _ in range(audio_num,history_max_num):
        audio_result.append(gr.update(visible=False))
        checkbox_result.append(gr.update(visible=False))
    return [*checkbox_result,*audio_result]

def delete_selected_history(*selected_list):
    for i in reversed(range(len(output_history))):
        if(selected_list[i]):
            output_history.pop(i)
    return [*sync_output_history_to_checkbox_audio()]

def merge_selected_history(*selected_list):
    m_list = []
    labels = ""
    for i in reversed(range(len(output_history))):
        if(selected_list[i]):
            labels+=" "+output_history[i]["label"]
            m_list.append(output_history[i]["value"][1])
    if(m_list):
        combined = np.hstack(m_list)
        delete_selected_history(*selected_list)       
        return add_to_history((32000, combined),labels)
    return [*sync_output_history_to_checkbox_audio()]

#添加音频历史记录 --结束--

# 添加根据模型路径切换参考音频功能--开始--
def change_sovits_weights(sovits_path):
    
    sovits_path = sovits_path.replace("SoVITS_weights/","")

    print(sovits_path)

    global reference_wavs,reference_dict
    reference_wavs,reference_dict = init_wav_list(sovits_path)


    return gr.update(choices=reference_wavs)

# 添加根据模型路径切换参考音频功能--结束--



with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    
    with gr.Column():
        # with gr.Group():
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True)
            SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path, interactive=True)
            #增加参考音频列表
            wavs_dropdown = gr.Dropdown(label=i18n("参考音频列表"), choices=reference_wavs,value="请选择参考音频",interactive=True)


            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
            
            # 增加wavs_dropdown部分，选择模型后，自动切换参考音频    
            SoVITS_dropdown.change(change_sovits_weights,[SoVITS_dropdown],[wavs_dropdown]).then(

                    tts_pipline.init_vits_weights, [SoVITS_dropdown], []

                )

            GPT_dropdown.change(tts_pipline.init_t2s_weights, [GPT_dropdown], [])
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(value=i18n("*请上传并填写参考信息"))
            inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath")
            prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="", lines=2)
            with gr.Row():
                prompt_language = gr.Dropdown(
                    label=i18n("参考音频的语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
                )
                with gr.Column():
                    ref_text_free = gr.Checkbox(label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"), value=False, interactive=True, show_label=True)
                    gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。"))
            # 更换参考音频列表，自动切换参考文本
            wavs_dropdown.change(change_wav,[wavs_dropdown],[inp_ref,prompt_text])
    
        with gr.Column():
            gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
            text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=16, max_lines=16)
            text_language = gr.Dropdown(
                label=i18n("需要合成的语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
            )

        
    with gr.Group():
        gr.Markdown(value=i18n("推理设置"))
        with gr.Row():

            with gr.Column():
                batch_size = gr.Slider(minimum=1,maximum=200,step=1,label=i18n("batch_size"),value=1,interactive=True)
                speed_factor = gr.Slider(minimum=0.25,maximum=4,step=0.05,label="speed_factor",value=1.0,interactive=True)
                top_k = gr.Slider(minimum=1,maximum=100,step=1,label=i18n("top_k"),value=5,interactive=True)
                top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("top_p"),value=1,interactive=True)
                temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("temperature"),value=1,interactive=True)
            with gr.Column():
                how_to_cut = gr.Radio(
                    label=i18n("怎么切"),
                    choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                    value=i18n("凑四句一切"),
                    interactive=True,
                )
                with gr.Row():
                    split_bucket = gr.Checkbox(label=i18n("数据分桶(可能会降低一点计算量，选就对了)"), value=True, interactive=True, show_label=True)
            # with gr.Column():
                output = gr.Audio(label=i18n("输出的语音"),show_download_button=True)
                with gr.Row():
                    inference_button = gr.Button(i18n("合成语音"), variant="primary")
                    stop_infer = gr.Button(i18n("终止合成"), variant="primary")
                
        
        inference_button.click(
            inference,
            [
                text,text_language, inp_ref, 
                prompt_text, prompt_language, 
                top_k, top_p, temperature, 
                how_to_cut, batch_size, 
                speed_factor, ref_text_free,
                split_bucket
             ],
            [output],
        )
        stop_infer.click(tts_pipline.stop, [], [])

    with gr.Group():
        gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="", lines=4)
            with gr.Column():
                _how_to_cut = gr.Radio(
                            label=i18n("怎么切"),
                            choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                            value=i18n("凑四句一切"),
                            interactive=True,
                        )
                cut_text= gr.Button(i18n("切分"), variant="primary")
            
            def to_cut(text_inp, how_to_cut):
                if len(text_inp.strip()) == 0 or text_inp==[]:
                    return ""
                method = get_method(cut_method[how_to_cut])
                return method(text_inp)
        
            text_opt = gr.Textbox(label=i18n("切分后文本"), value="", lines=4)
            cut_text.click(to_cut, [text_inp, _how_to_cut], [text_opt])
        gr.Markdown(value=i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

        # 输出音频历史记录 -- 开始--
        history_audio = []
        history_checkbox = []
        with gr.Accordion("生成历史"):
            with gr.Row():
                shown_audio_num = gr.Slider(1,20,history_max_num,step=1,interactive=True,label="记录显示数量")
                add_history_button = gr.Button("添加当前音频记录",variant="primary")
                merge_audio_button = gr.Button("合并选中音频",variant="primary")
                delete_select_history_button = gr.Button("删除选择的记录")
                clear_history_button = gr.Button("清空记录")
            index=0
            while(index<history_max_num):
                index+=5
                with gr.Row():
                    for _ in range(5):
                        with gr.Group():
                            history_checkbox.append(gr.Checkbox(interactive=True,show_label=False,label=""))
                            history_audio.append(gr.Audio(label="",show_download_button=True))

            shown_audio_num.change(shown_audio_num_change,[shown_audio_num],[*history_checkbox,*history_audio])
            add_history_button.click(add_to_history,[output,text],[*history_checkbox,*history_audio])
            merge_audio_button.click(merge_selected_history,[*history_checkbox],[*history_checkbox,*history_audio])
            delete_select_history_button.click(delete_selected_history,[*history_checkbox],[*history_checkbox,*history_audio])
            clear_history_button.click(clear_history,outputs=[*history_checkbox,*history_audio])

        # 输出音频历史记录 -- 结束--

app.queue().launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=is_share,
    server_port=infer_ttswebui,
    quiet=True,
)
