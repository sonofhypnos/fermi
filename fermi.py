
from openai import OpenAI
import os
import json
import math
from typing import List, Dict, Tuple
import subprocess

def get_openai_key_from_1password(item_identifier: str) -> str:
    """Fetch the OpenAI API key stored in 1Password using the 1Password CLI."""
    try:
        # The command to execute, split into parts for subprocess
        command = ["op", "item", "get", item_identifier, "--fields", "credential"]

        # Run the command and capture the output
        result = subprocess.run(command, text=True, capture_output=True, check=True)

        # The output will be in stdout
        openai_key = result.stdout.strip()

        return openai_key
    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch OpenAI API key: {e}")
        return ""



def calculate_log_loss(predicted_answer: str, actual_answer: str) -> float:
    """Calculate the log-scale loss between the predicted and actual answers."""
    try:
        predicted_value = float(predicted_answer)
        actual_value = float(actual_answer)
        return abs(math.log10(predicted_value) - math.log10(actual_value))
    except ValueError:
        # Handle cases where conversion to float fails
        return float("inf")


# def generate_prediction(question: str, prompt_template: str) -> str:
#     """Generate a prediction for a given question using a specified prompt template."""
#     prompt = prompt_template.format(question=question)
#     messages = [{"role": "user", "content": prompt}]

#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#         stop=None,
#         top_p=1.0,
#     )
#     return response.choices[0].message.content

# def generate_prediction(question: str, prompt_template: str) -> str:
#     """Generate a prediction for a given question using a specified prompt template and extract the final answer."""
#     prompt = prompt_template.format(question=question)
#     messages = [{"role": "user", "content": prompt}]
#     thread = client.beta.threads.create(
#                 messages=messages,
#                 )

#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#         stop=None,
#         top_p=1.0,
#     )
#     full_response = response.choices[0].message.content

#     # Extract the final answer using the specified format
#     start = full_response.find("[[") + 2  # Offset by 2 to skip the brackets themselves
#     end = full_response.find("]]", start)
#     final_answer = full_response[start:end].strip() if start > 1 and end > start else "NaN"

#     return final_answer, full_response
import io
from contextlib import redirect_stdout


def generate_prediction(question: str, prompt_template: str, client, model) -> str:
    """Generate a prediction for a given question using a specified prompt template, extract, and evaluate the final answer."""
    system_prompt = "You are an expert superforecaster and familiar with the work of Tetlock and others. Make a fermi estimate for the question below and think step by step."
    prompt = prompt_template.format(question=question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stop=None,
        top_p=1.0,
    )
    full_response = response.choices[0].message.content

    # Extract the final answer or program

    to_removes = ["python\n", "python" ]
    for to_remove in to_removes:
        final_expression = final_expression.replace(
            f"```{to_remove}", "```"
        )
    start = full_response.find("```") + 3  # Offset to skip the '```' itself
    end = full_response.find("```", start)
    if start > 1 and end > start:
        final_expression = full_response[start:end].strip()
        try:
            # TODO: we could also tell the functions in the prompt to not import the math module.
            def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name in ["math", "another_safe_module"]:
                    return __import__(name, globals, locals, fromlist, level)
                raise ImportError(f"Import of {name} is not allowed")

            safe_globals = {
                "__builtins__": {"__import__": safe_import},
                # Include any other built-ins you wish to allow
            }
            safe_locals = {"result": None}
            exec(final_expression, safe_globals, safe_locals)
            final_answer = float(safe_locals.get("result", "Number not found"))

        except Exception as e:
            print(f"Error evaluating the expression: {e}")
            final_answer = "NaN"
    else:
        final_answer = "NaN"

    return str(final_answer), full_response


# We could use the code interpreter to run the code and get the answer.
def compare_prompts(
    question: str, actual_answer: str, prompts: List[str], remainder: str, client, model
) -> List[Tuple[str, float]]:
    """Compare different prompts by generating predictions for each and calculating their log-scale loss."""
    results = []
    for prompt_template in prompts:
        predicted_answer, full_response = generate_prediction(
            question, prompt_template + "\n" + remainder, client, model
        )
        log_loss = calculate_log_loss(predicted_answer, actual_answer)

        print(
            f"Prompt: {prompt_template}\nFull Response: {full_response}\nLog Loss: {log_loss}\n"
        )
        results.append(
            {
                "prompt_template": prompt_template,
                "log_loss": log_loss,
                "predicted_answer": predicted_answer,
                "actual_answer": actual_answer,
                "full_response": full_response,
            }
        )
    return results


def extract_answer_and_unit(answer_string):
    response = answer_string.split(" ")
    if len(response) == 2:
        return response[0], response[1]
    if len(response) == 1:
        return response[0], "no units. It is dimensionless."
    else:
        raise Exception(f"Answer string is not in the expected format: {answer_string}")


def save_results_to_json(results: List[Dict], filename: str):
    """Save the results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def load_train_data(filepath: str) -> List[Dict]:
    """Load training data from a JSON file."""
    with open(filepath, "r") as file:
        train_data = json.load(file)
    return train_data


def print_results_and_average_loss(data):
    """Print the saved results and calculate the average loss per prompt."""
    for result in data:
        prompt = result["prompt"]
        responses = result["responses"]
        fp_score = result["mean_fp_score"]
        log_loss = result["mean_log_loss"]
        print(f"Prompt: {prompt}\nMean Log Loss: {log_loss}\nMean FP Score: {fp_score}\n")


def load_data_print_results(filename="gpt_prompt_results.json"):
    data = load_results(filename)
    print_results_and_average_loss(data)


def load_results(filename):
    """Load the JSON data from a file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def fp_score(predicted, actual):
    if predicted == "Nan":
        return 0
    predicted = float(predicted)
    actual = float(actual)
    if actual <= 0 or predicted <= 0:
        return 0
    log_difference = abs(math.log10(predicted) - math.log10(actual))
    score = max(0, 1 - (1 / 3) * log_difference)
    return score


def main():

    prompts = [
        "Address the question '{question}' by estimating a lower and an upper 5% bound for the possible answers. First list your consideration for arriving at those estimates. Finally, use the geometric mean of these bounds to compute the most likely answer.",
        "Consider a world where '{question}' has a straightforward answer. Describe the steps you would take to solve this problem in such a world, then apply this reasoning to our current problem.",
        "The question '{question}' involves complex issues. Break these down into simpler components, solve each component, and then synthesize these solutions to arrive at a final answer.",
        "Quantitatively estimate the answer to '{question}' by identifying key variables and their relationships. Detail your estimation process and final calculation.",
        #TODO: add that it shouldn't trust it's own calculations to the prompt.
    ]
    hints = ["While you have excellent knowledge, you should not trust your own calculations. Put as much of your calculation as possible in the code block", "Try two different approaches to the problem and compare the results. Submit a mean of the results that reflects how much you trust each approach."]

    model = "gpt-3.5-turbo-0125"



    train_data = load_train_data("./data/realFP/train_realfp.json")

    # Dictionary to accumulate prompt results
    prompt_results = {prompt: [] for prompt in prompts}
    sample_size = 10
    results_dir = "./results/"


    filename = f"gpt_prompt_results_{model}_{sample_size}.json"
    if os.path.exists(os.path.join(results_dir, filename)):
        load_data_print_results(os.path.join(results_dir, filename))
        exit()

    # Get openai key
    item_identifier = "yocqakmciuu7bidjgftbol57wy"
    openai_api_key = get_openai_key_from_1password(item_identifier)
    client = OpenAI(api_key=openai_api_key)

    # Iterate over each training example
    for i, example in enumerate(train_data):
        if i >= sample_size:
            break
        question = example["question"]
        # Extract and normalize the actual answer and its unit
        actual_answer, unit = extract_answer_and_unit(example["answer"])
        #        remainder = f"Give your answer in {unit}. Provide the final answer as a Python executable expression or final program within '```' that prints the final value."
        remainder = f"Give your answer in {unit}. Provide the final answer as a Python executable expression or final program within '```' as brackets that sets the `result` variable to the final value. Make sure that your final program doesn't have any free parameters left. Make best-guess estimates for them instead."
        # Compare prompts for the current example
        comparison_results = compare_prompts(
            question, actual_answer, prompts, remainder, client, model
        )

        # Accumulate results
        for results in comparison_results:
            prompt_results[results["prompt_template"]].append(results)

    # Compute mean log loss for each prompt and prepare results for JSON
    final_results = []
    for prompt, responses in prompt_results.items():
        fp_scores = [
            fp_score(response["predicted_answer"], response["actual_answer"])
            for response in responses
        ]
        mean_fp_score = sum(fp_scores) / len(fp_scores)
        log_losses = [
            calculate_log_loss(response["predicted_answer"], response["actual_answer"])
            for response in responses
        ]
        mean_log_loss = sum(log_losses) / len(log_losses)
        final_results.append(
            {
                "prompt": prompt,
                "mean_log_loss": mean_log_loss,
                "mean_fp_score": mean_fp_score,
                "responses": responses,
            }
        )

    # Save results to JSON
    save_results_to_json(final_results, f"{results_dir}gpt_prompt_results_{model}_{sample_size}.json")


if __name__ == "__main__":
    main()

