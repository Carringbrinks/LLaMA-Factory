
import copy
import random
import json


def load_json(data_path):
    """
    # 加载 json 文件
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data_path, data_list):
    """
    # 保存 json 文件
    """
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def format_human_data():
    human_input_path = r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\samples_human_knowledge_0814.json"
    datas = load_json(human_input_path)
    new_data = []
    for key in datas:
        single_table_data = []
        for data in datas[key]:
            query = data["query"]
            ans = data["answer"]
            if isinstance(query, str) and isinstance(ans, str):
                new_query = random.choice([query + ans, ans + query])
                single_table_data.append(new_query)
        new_data.append({"query": single_table_data, "id": key})
    save_json("./meta_data/samples_human_knowledge_0814_new.json", new_data)


def re_human_data():
    human_input_path = "./save_data/human_query_react_correct_result_0812.json"
    react_data_path = "./save_data/query_react_correct_result_0812.json"

    human_datas = load_json(human_input_path)
    react_data = load_json(react_data_path)
    new_react_data = {}
    count = 0
    other_count = 0
    unknow_count = 0
    for key in react_data:
        repeat_count = 0

        if key in human_datas:
            tem_data = []
            single_react_data = react_data[key]
            react_data_query = [
                react_query["query"] for react_query in single_react_data
            ]
            human_datas_query = [
                human_query["query"] for human_query in human_datas[key]
            ]
            for i, query_0 in enumerate(react_data_query):
                if not any(query_0 in element for element in human_datas_query):
                    count += 1
                    tem_data.append(react_data[key][i])
                new_react_data.update({key: tem_data})

            # for i, query_0 in enumerate(react_data_query):
            #     for j, query_1 in enumerate(human_datas_query):
            #         if query_0 in query_1:
            #             # single_react_data.pop(i-repeat_count)
            #             repeat_count += 1
            #             break
            #         elif j == len(human_datas_query)-1:
            #             # print(111)
            #             tem_data.append(single_react_data[i])
            # if len(human_datas_query) == 0:
            #     tem_data.append(single_react_data)
            # new_react_data.update({key: tem_data})

        else:
            other_count += len(react_data[key])
            new_react_data.update({key: react_data[key]})
    save_json("save_data/query_react_correct_result_0812_new.json", new_react_data)


def merge_json():
    json_path_list = [
        r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\ori_queries_general_single_qwen-72b_human_0902_0.json",
        r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\ori_queries_general_single_qwen-72b_human_0902_1.json",
        r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\ori_queries_general_single_qwen-72b_human_0902_2.json"

    ]
    all_json_data = {}
    for json_path in json_path_list:
        datas = load_json(json_path)

        all_json_data.update(datas)
    save_json(
        r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\ori_queries_general_single_qwen-72b_human_0902.json",
        all_json_data,
    )

def is_error(react):
    if react["output"] == "Agent stopped due to iteration limit or time limit.":
        return True
    if len(react["intermediate_steps"]) == 0:
        return True
    if "Action Input" in react["output"]:
        return True
    else:
        for msg in [
            "pd.read_csv",
            "pd.DataFrame",
            "Invalid Format",
            "AttributeError",
            "NameError",
            "ModuleNotFoundError",
            "ValueError",
            "KeyError",
            "TypeError",
        ]:
            last_ob = react["intermediate_steps"][
                str(int(len(react["intermediate_steps"]) - 1))
            ]["observation"]
            if msg in last_ob:
                return True
        return False


def filter_data():
    # json_path = r"./origin_data.json"
    json_path = r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\ori_queries_general_single_qwen-72b_human_0902.json"
    datas = load_json(json_path)

    datas_new = {}
    dup_count = 0
    count = 0
    for key in datas:
        pop_count = 0
        single_table_data_new = copy.deepcopy(datas[key])
        for i, data in enumerate(datas[key]):
            if is_error(data):
                single_table_data_new.pop(i - pop_count)
                pop_count += 1

        dup_count += pop_count
        count += len(single_table_data_new)
        datas_new.update({key: single_table_data_new})

    save_json(r"./meta_data\qwen_gen_qa_slm_prompt_0826\filter_ori_human_data_zg_0902.json", datas_new)
    print(count)
    print(dup_count)

def delete_python_code_visual_data():
    data_path = "F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\meta_python_code_data_0814.json"
    datas = load_json(data_path)
    new_data = []
    visual_words = ["趋势图","箱型图", "箱线图", "折线图", "分布图", "散点图", "柱状图", "图表", "时间序列图", "热力图", "饼图", "变化图", "直方图", "关系图"]
    for i, data in enumerate(datas):
        temp_query = []
        if len(data["query"]) != 2:
            for query in data["query"]:
                if not any(visual_word in query for visual_word in visual_words):
                    temp_query.append(query)
            data["query"] = temp_query
            if len(temp_query):
                new_data.append(data)

    save_json("F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\meta_python_code_data_de_plt_0814.json", new_data)

def delete_human_visual_data():
    data_path = "F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\meta_human_knowledge_data_0814.json"
    datas = load_json(data_path)
    new_data = []
    visual_words = ["趋势图","箱型图", "箱线图", "折线图", "分布图", "散点图", "柱状图", "图表", "时间序列图", "热力图", "饼图", "变化图", "直方图", "关系图"]
    for i, data in enumerate(datas):
        temp_query = []
        # if len(data["query"]) != 2:
        for query in data["query"]:
            if not any(visual_word in query for visual_word in visual_words):
                temp_query.append(query)
        data["query"] = temp_query
        if len(temp_query):
            new_data.append(data)

    save_json("F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\meta_human_knowledge_data_de_plt_0814.json", new_data)

def delete_ori_visual_data():
    data_path = r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\save_data\0814\filter_ori_human_data.json"
    datas = load_json(data_path)
    new_data = {}
    visual_words = ["趋势图", "箱型图", "箱线图", "折线图", "分布图", "散点图", "柱状图", "图表", "时间序列图",
                    "热力图", "饼图", "变化图", "直方图", "关系图"]
    for key in datas:
        temp_query = []
        for data in datas[key]:
            if not any(visual_word in data["query"] for visual_word in visual_words):
                temp_query.append(data)

        new_data.update({key:temp_query})


    save_json(r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\save_data\0814\filter_ori_human_data_de_plt.json",
              new_data)

def gen_meta_confuse_data(data_path, save_path, data_type):

    datas = load_json(data_path)
    new_datas = {}
    for data in datas:
        new_data = []
        for query in data["query"]:
            new_query = {}
            new_query.update(query)
            # new_query.update(dict(tool_type=data_type))
            new_data.append(new_query)
        table_id = data["id"]
        if table_id in new_datas:
            new_datas[table_id].extend(new_data)
        else:
            new_datas.update({table_id:new_data})

    save_json(save_path, new_datas)

def format_zg_human_data(data_path, save_path):
    datas = load_json(data_path)

    new_datas = []
    single_table_data = {"query": [], "id": "", "path":""}
    for data in datas:

        table_name = data["path"].split("/")[-1].split(".")[0]

        if table_name == single_table_data["id"]:
            single_table_data["query"].append(data)
        else:
            if len(single_table_data["query"]):
                new_datas.append(single_table_data)
            single_table_data = {"query": [], "id": "","path":""}
            single_table_data["id"] = table_name
            single_table_data["path"] = data["path"]
            single_table_data["query"].append(data)

    ## 添加最后一个数据
    new_datas.append(single_table_data)

    save_json(save_path, new_datas)

def merge_train_data():
    data_paths = ["./react_train_data_0902_no_huamn.json", r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\react_train_data_0902_huamn_zg.json"]
    save_path = "react_train_data_0902.json"

    all_data = []
    for path in data_paths:
        data = load_json(path)
        all_data.extend(data)
    print(len(all_data))
    save_json(save_path, all_data)

def jsonl_to_json(json_path, jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f, open(json_path, "w", encoding="utf-8") as f1:
        jsonl_data = load_json(jsonl_path)
        for line in jsonl_data:
            print(line)





if __name__ == "__main__":
    pass

    jsonl_to_json("sft_data_merge_v20_quality_filtered.json", "/home/scb123/PyProject/LLaMA-Factory/data/sft_data_merge_v20_quality_filtered.jsonl")
    # re_human_data()
    # filter_data()
    # merge_json()
    # re_human_data()
    # merge_train_data()

    # delete_python_code_visual_data()
    # delete_human_visual_data()
    # delete_ori_visual_data()

    # meta_json = r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\queries_general_single_qwen-72b_human_0902.json"
    # save_meta_json = r"F:\PythonPro\ReactFunctionCallData\gen_react_qa\meta_data\qwen_gen_qa_slm_prompt_0826\queries_general_single_qwen-72b_human_0902_confuse.json"
    # gen_meta_confuse_data(meta_json, save_meta_json, "human_input")


    # format_zg_human_data("F:\PythonPro\ReactFunctionCallData\gen_react_qa\human_input_20240902.json", "F:\PythonPro\ReactFunctionCallData\gen_react_qa\queries_general_single_qwen-72b_human_0902.json")