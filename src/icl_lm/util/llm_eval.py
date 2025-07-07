import os
import json
import statistics
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from .generator import Generator

SYSTEM_PROMPT = "You are a writing evaluator designed to assess student story completions. You will be provided children's stories written for a 3-4 year old audience. Your role is to provide constructive, fair, and detailed evaluations based on specific rubric criteria."

USER_PROMPT = """
In the following exercise, the student is given a pre-written beginning of a story. The student needs to complete this story. The exercise tests the student´s language abilities and creativity.

Here is the pre-written beginning:

<PROVIDED BEGINNING>
[STORY_BEGIN]
</PROVIDED BEGINNING>

Here is the students response:

<STUDENT RESPONSE>
[STORY_END]
</STUDENT_RESPONSE>

First, provide a concise qualitative assessment about the student's writing. Then, give the writing a grade out of 10. These assessments should be done for each of the following rubric items:

1. Grammar:
* Is the writing grammatically correct?
* Evaluate syntax, punctuation, and sentence structure.
2. Consistency:
* Is the student's writing consistent with the provided beginning of the story?
* How well does the student complete the final sentence of the prescribed beginning?
3. Plot:
* Does the plot of the student's writing make sense (regardless of the provided beginning)?
4. Creativity: 
* How creative is the student's writing?

Format your response as follows:

<GRAMMAR>
[Qualitative assessment of grammar]
</GRAMMAR>
<GRAMMAR_GRADE>
[Grade out of 10]
</GRAMMAR_GRADE>

<CONSISTENCY>
[Qualitative assessment of consistency]
</CONSISTENCY>
<CONSISTENCY_GRADE>
[Grade out of 10]
</CONSISTENCY_GRADE>

<PLOT>
[Qualitative assessment of plot]
</PLOT>
<PLOT_GRADE>
[Grade out of 10]
</PLOT_GRADE>

<CREATIVITY>
[Qualitative assessment of creativity]
</CREATIVITY>
<CREATIVITY_GRADE>
[Grade out of 10]
</CREATIVITY_GRADE>

Provide your assessment below:
"""

class Evaluator:
    def __init__(self, config, model, splits, tokenizer, checkpoint_dir, generation_dir, device):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = splits["test"]
        self.device = device
        self.dataset_name = splits["name"]

        self.model_name = model.name
        self.out_dir = os.path.join(generation_dir, self.dataset_name, self.model_name)
        os.makedirs(self.out_dir, exist_ok=True)

        self.input_path = os.path.join(self.out_dir, "input.jsonl")
        self.output_path = os.path.join(self.out_dir, "output.jsonl")
        self.info_path = os.path.join(self.out_dir, "info.json")
        self.results_path = os.path.join(self.out_dir, "results.json")

        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")
        self.client = OpenAI(api_key=self.api_key)

        self.generator = Generator(config, model, splits, tokenizer, checkpoint_dir, generation_dir, device)

    def generate_input_file(self):
        
        self.model.eval()
        self.model.to(self.device)
        
        input_items = []
        num_samples = self.config.num_samples
        max_len = self.config.max_length

        for i in tqdm(range(num_samples), desc="Preparing LLM evaluation prompts"):
            sample = self.test_data[i]
            input_tokens = sample[:max_len]
            
            if self.tokenizer.eos_token_id in input_tokens:
                    eos_index = (input_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
                    input_tokens = input_tokens[:eos_index]
            
            input_len = max(10, len(input_tokens) // 2)
            prompt_ids = input_tokens[:input_len].to(self.device)

            generated = self.generator.sample(prompt_ids)
            
            if self.tokenizer.eos_token_id in generated:
                    eos_index = (generated == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
                    generated = generated[:eos_index]

            if len(generated) <= input_len:
                continue  # Skip if generation is too short

            prompt_text = self.tokenizer.decode(prompt_ids.tolist())
            generation_text = self.tokenizer.decode(generated.tolist()[input_len:])

            full_prompt = USER_PROMPT.replace("[STORY_BEGIN]", prompt_text).replace("[STORY_END]", generation_text)

            input_items.append({
                "custom_id": f"{self.model_name}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt}
                    ],
                    "max_tokens": 1000
                }
            })

        with open(self.input_path, "w") as f:
            for item in input_items:
                f.write(json.dumps(item) + "\n")

    def create_batch(self):
        if os.path.exists(self.info_path):
            return

        if not os.path.exists(self.input_path):
            self.generate_input_file()

        file = self.client.files.create(file=open(self.input_path, "rb"), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{self.model_name} LLM eval"}
        )

        with open(self.info_path, "w") as f:
            json.dump({"batch_id": batch.id}, f)

    def save_output(self):
        if os.path.exists(self.output_path):
            return True

        with open(self.info_path, "r") as f:
            batch_id = json.load(f)["batch_id"]

        batch = self.client.batches.retrieve(batch_id)

        if batch.status == "completed":
            content = self.client.files.content(batch.output_file_id).text
            with open(self.output_path, "w") as f:
                f.write(content)
            return True
        elif batch.status == "failed":
            raise RuntimeError(f"Batch {batch_id} failed.")
        else:
            print(f"Batch {batch_id} is still processing: {batch.status}")
            return False

    def parse_output(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                return json.load(f)

        with open(self.output_path, "r") as f:
            lines = f.readlines()

        def extract_score(tag, text):
            try:
                value = text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()
                return int(value.split("/")[0].strip())
            except:
                return None

        scores = {"grammar": [], "consistency": [], "plot": [], "creativity": []}
        errors = 0

        for line in lines:
            response = json.loads(line)
            content = response["response"]["body"]["choices"][0]["message"]["content"]

            g = extract_score("GRAMMAR_GRADE", content)
            c = extract_score("CONSISTENCY_GRADE", content)
            p = extract_score("PLOT_GRADE", content)
            cr = extract_score("CREATIVITY_GRADE", content)

            if None in (g, c, p, cr):
                errors += 1
                continue

            scores["grammar"].append(g)
            scores["consistency"].append(c)
            scores["plot"].append(p)
            scores["creativity"].append(cr)

        results = {}
        for key in scores:
            s = scores[key]
            results[key] = {
                "mean": round(sum(s) / len(s), 2),
                "stdev": round(statistics.stdev(s), 2) if len(s) > 1 else 0.0
            }

        total_scores = [sum(x) / 4 for x in zip(*scores.values())]
        results["total"] = {
            "mean": round(sum(total_scores) / len(total_scores), 2),
            "stdev": round(statistics.stdev(total_scores), 2) if len(total_scores) > 1 else 0.0
        }

        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

        latex = (
            f"{results['total']['mean']} ({results['total']['stdev']}) & "
            f"{results['grammar']['mean']} ({results['grammar']['stdev']}) & "
            f"{results['consistency']['mean']} ({results['consistency']['stdev']}) & "
            f"{results['plot']['mean']} ({results['plot']['stdev']}) & "
            f"{results['creativity']['mean']} ({results['creativity']['stdev']})"
        )
        print("Latex format:")
        print(latex)

    def evaluate(self):
        self.create_batch()
        if self.save_output():
            self.parse_output()
