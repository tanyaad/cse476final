#!/usr/bin/env python3

"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import requests
#api
API_KEY = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL = "bens_model"
INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type":  "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": prompt}],
        "temperature": temperature,
        "max_tokens": 256,
    }
    try:
        #print("DEBUG:", json.dumps(payload, indent=2))
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": resp.status_code}
        else:
            try: err_text = resp.json()
            except Exception: err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": resp.status_code}
    except requests.RequestException as e:
        print("RequestException:", e)
        return {"ok": False, "text": None, "raw": None, "status": -1}

def cleanoutput(text:str)->str: #gets only text
    if not text:
        return""
    out=text.strip()

    if out.startswith("```"): #remove
        out=out.strip("`")
        lines=out.splitlines()
        if lines and lines[0].lower().startswith("python"):
            out="\n".join(lines[1:]).strip()

    return out


def run_agent(question:str)->str:
    plan_prompt =(
    "Problem: "+question+
   " Before answering write a short plan on how you will solve it."
    "Don't include the final answer right now."
    )
    r1=call_model_chat_completions(plan_prompt, system="You are a helpful assistant. Produce only a short plan.", temperature=0.3)

    if not r1["ok"]:
        print("ERROR in planning stage:", r1)

        return "ERROR"
    
    plan = r1["text"]

    answer_prompt=(
    "Problem: "+question+
    "Here is the plan: "+plan+
    "Now follow this plan and only provide the final answer and no explaining."
    )

    r2= call_model_chat_completions(answer_prompt, system="Follow the plan exactly and return only the answer.", temperature=0.0)
    if not r2["ok"]:
        return "ERROR"
    final =cleanoutput(r2["text"])
    return final




def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for i, obj in enumerate(questions, start=1):
        q=obj["input"]
        print(f"Processing question {i}/{len(questions)}...")
        # Example: assume you have an agent loop that produces an answer string.
        # real_answer = agent_loop(question["input"])
        # answers.append({"output": real_answer})
        ans= run_agent(q)
        answers.append({"output": ans})
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()

