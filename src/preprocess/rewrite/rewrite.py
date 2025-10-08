from datetime import datetime
import os
import json
import re

from openai import OpenAI
import json
import json_repair
from tqdm import tqdm
from src.utils.utils import call_with_messages

TRAIN_BEFORE_REWRITE = "data/origin/seed42/train_data.json"
TEST_BEFORE_REWRITE = "data/origin/seed42/test_data.json"

TRAIN_REWRITE_OUTPUT = "data/rewrite/seed42/train_rewrite_data.json"
TEST_REWRITE_OUTPUT = "data/rewrite/seed42/test_rewrite_data.json"

# rewrite prompt
MASHUP_REWRITE_PROMPT = "src/prompt/rewrite_mashups.txt"
API_REWRITE_PROMPT = "src/prompt/rewrite_apis.txt"

# remote model
OPENAI_API_BASE = ""
OPENAI_API_KEY = ""
PRETRAINED_MODEL_PATH = ""
LOG_FILE = "logs/"

def process_rewrite_json(text, type):
    try:
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        json_objects = [json.loads(match) for match in matches]
        for match in json_objects:
            try:
                mashup = match
                description = mashup.get("enhanced_description", "")
                if description:
                    if type == "api":
                        tags = mashup.get("tags", [])
                        if len(tags) == 0:
                            return {}
                        else:
                            return {"description": description, "tags": tags}
                    elif type == "mashup":
                        categories = mashup.get("categories", [])
                        if len(categories) == 0:
                            return {}
                        else:
                            return {
                                "description": description,
                                "categories": categories,
                            }

            except Exception as e:
                print(f"JSON not found: {e}")
                return {}
    except Exception as e:
        print(f"JSON not found: {e}")
        return {}
    return {}


def rewrite_mashup(mashup_prompt, question, base_url, api_key, model_path):

    mashup_str = json.dumps(
        {
            "title": question["title"],
            "description": question["description"],
            "tags": question["tags"],
        }
    )

    mashup_prompt += """
<Now,My Input Is Follow>
{}
""".format(
        mashup_str
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": mashup_prompt},
    ]
    wrong_error = 0
    while True:
        rewrite_json = call_with_messages(messages, base_url, api_key, model_path)

        rewrite_json = process_rewrite_json(rewrite_json, "mashup")
        if rewrite_json == {} and wrong_error < 8:
            print("llm gets wrong answer")
            wrong_error += 1
        else:
            print("rewrite mashup after process:", rewrite_json)
            break
    return rewrite_json


def rewrite_api(api_prompt, api, base_url, api_key, model_path):
    api_str = json.dumps(
        {"title": api["title"], "description": api["description"], "tags": api["tags"]}
    )
    api_prompt += """
<Now,My Input Is Follow>
{}
""".format(
        api_str
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": api_prompt},
    ]
    wrong_error = 0
    while True:
        rewrite_json = call_with_messages(messages, base_url, api_key, model_path)
        rewrite_json = process_rewrite_json(rewrite_json, "api")

        if rewrite_json == {} and wrong_error < 8:
            print("llm gets wrong answer")
            wrong_error += 1
        else:
            print("api after process:", rewrite_json)
            break
    return rewrite_json


def rewrite_mashup_api(
    questions_origin,
    state_file,
    file_name,
    mashup_prompt,
    api_prompt,
    base_url,
    api_key,
    model_path,
):
    # check log file
    start_index = 0
    if os.path.exists(state_file):
        with open(state_file, "r") as state_f:
            try:
                state = json.load(state_f)
                start_index = state.get("last_index", 0)
            except json.JSONDecodeError:
                print("state log catch error")

    # if log exist, load content
    questions_rewrite = []
    output_file_path = file_name
    if os.path.exists(output_file_path) and start_index > 0:
        with open(output_file_path, "r", encoding="utf-8") as f:
            questions_rewrite = json.load(f)

    # start process
    pbar = tqdm(
        total=len(questions_origin) - start_index, desc="rewriting", colour="blue"
    )

    for index, question in enumerate(questions_origin[start_index:], start=start_index):
        pbar.update(1)
        mashup_json = rewrite_mashup(
            mashup_prompt, question, base_url, api_key, model_path
        )
        question["description"] = mashup_json.get(
            "description", question["description"]
        )
        question["categories"] = mashup_json.get("categories", question["categories"])
        if question["related_apis"] is not None:
            for api in question["related_apis"]:
                if api is not None:
                    api_json = rewrite_api(api_prompt, api, base_url, api_key, model_path)
                    if api_json["description"] is not None:
                        api["description"] = api_json.get("description", api["description"])
                    if api_json["tags"] is not None:
                        api["tags"] = api_json.get("tags", api["tags"])

        questions_rewrite.append(question)
        print("question after rewrite:", question)

        # save state
        with open(state_file, "w") as state_f:
            json.dump({"last_index": index + 1}, state_f)

        # save result
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(questions_rewrite, f, ensure_ascii=False, indent=4)

    return questions_rewrite


if __name__ == "__main__":
    train_mashup_path = TRAIN_BEFORE_REWRITE
    test_mashup_path = TEST_BEFORE_REWRITE

    with open(train_mashup_path, "r", encoding="utf-8") as file:
        train_set = json.load(file)
    with open(test_mashup_path, "r", encoding="utf-8") as file:
        questions_origin = json.load(file)
    dataset = train_set[:5]
    questions_origin = questions_origin[:5]
    
    train_log_name = "train_rewrite_log.json"
    train_log_path = LOG_FILE + train_log_name
    
    with open(MASHUP_REWRITE_PROMPT, "r", encoding="utf-8") as file:
        mashup_prompt = file.read()
    with open(API_REWRITE_PROMPT, "r", encoding="utf-8") as file:
        api_prompt = file.read()
    dataset = rewrite_mashup_api(
        train_set,
        train_log_path,
        TRAIN_REWRITE_OUTPUT,
        mashup_prompt,
        api_prompt,
        OPENAI_API_BASE,
        OPENAI_API_KEY,
        PRETRAINED_MODEL_PATH,
    )


    test_log_name = "test_rewrite_log.json"
    test_log_path = LOG_FILE + test_log_name
    
    questions_origin = rewrite_mashup_api(
        questions_origin,
        test_log_path,
        TRAIN_REWRITE_OUTPUT,
        mashup_prompt,
        api_prompt,
        OPENAI_API_BASE,
        OPENAI_API_KEY,
        PRETRAINED_MODEL_PATH,
    )
