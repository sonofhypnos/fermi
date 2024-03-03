""" You are an expert superforecaster, familiar with the work of
            Tetlock and others. Make a fermi estimate for the following question. You MUST give a numeric answer UNDER
            ALL CIRCUMSTANCES. If for some reason you can’t answer, pick a base rate for questions of this form.
            Question: {question}
            # Question Background: {background}
            Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the
            decimal. Do not output anything else.
            Answer: {{ Insert answer here }}"""

prompt = """ You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can’t answer, pick the base rate, but return a number between 0 and 1. Question: {question} Question Background: {background} Resolution Criteria: {resolution_criteria} Today’s date: {date_begin} Question close date: {date_end} Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. Do not output anything else. Answer: {{ Insert answer here }} """

from openai import OpenAI
import json
import math
from typing import List, Dict, Tuple
import subprocess

def get_openai_key_from_1password(item_identifier: str) -> str:
    """Fetch the OpenAI API key stored in 1Password using the 1Password CLI."""
    try:
        # The command to execute, split into parts for subprocess
        command = ['op', 'item', 'get', item_identifier, '--fields', 'credential']
        
        # Run the command and capture the output
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        
        # The output will be in stdout
        openai_key = result.stdout.strip()
        
        return openai_key
    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch OpenAI API key: {e}")
        return ""

# Example usage
item_identifier = "yocqakmciuu7bidjgftbol57wy"
openai_api_key = get_openai_key_from_1password(item_identifier)
client = OpenAI(api_key=openai_api_key)


# result = subprocess.run(
#     "source /home/tassilo/.zshrc && export OPENAI_API_KEY=$(get_openai_key)",
#     stdout=subprocess.PIPE,
#     text=True,
#     executable="/bin/zsh",
# )  # See my dotfiles for get_openäi_key

# Below should
prompts = [
    "Given the question '{question}', break down your thought process into detailed steps. Explain each step clearly as you work towards the answer.",
    "For the question '{question}', first provide a reasonable lower and upper bound for the answer. Explain how you arrived at these bounds. Then, calculate the final answer within these limits.",
    "Address the question '{question}' by estimating a lower and an upper 5% bound for the possible answers. First list your consideration for arriving at those estimates. Finally, use the geometric mean of these bounds to compute the most likely answer.",
    "Consider the question '{question}'. Compare this situation to a historical or well-known similar problem. Use this comparison to guide your reasoning and provide an answer.",
    "Imagine you are an expert in the field relevant to '{question}'. As this expert, how would you approach solving this problem? Detail your expert analysis step by step.",
    "Solve the problem posed by '{question}' by drawing an analogy to a simpler but related problem. Use the solution of the simpler problem to guide your answer.",
    "Consider a world where '{question}' has a straightforward answer. Describe the steps you would take to solve this problem in such a world, then apply this reasoning to our current problem.",
    "The question '{question}' involves complex issues. Break these down into simpler components, solve each component, and then synthesize these solutions to arrive at a final answer.",
    "Using the Socratic method, ask and answer your own questions to explore and solve '{question}'. This iterative process should guide you to a reasoned conclusion.",
    "Approach '{question}' by considering insights from different fields (e.g., physics, economics, biology). How do these perspectives help you understand and solve the problem?",
    "Apply fundamental principles relevant to the question '{question}' to guide your reasoning. What principles are most relevant, and how do they lead to an answer?",
    "Quantitatively estimate the answer to '{question}' by identifying key variables and their relationships. Detail your estimation process and final calculation.",
    "Use logical reasoning and deduction to arrive at an answer for '{question}'. Start with broad logical statements and narrow down to specific deductions that lead to the answer.",
    "Apply creative thinking to solve '{question}'. Imagine unconventional solutions or approaches that could lead to an answer, explaining your creative process step by step.",
    "Solve '{question}' by first estimating the answer and then analyzing potential errors in your estimation. Adjust your initial estimate based on this analysis to provide a refined answer.",
]

model = "gpt-3.5-turbo"


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

def generate_prediction(question: str, prompt_template: str) -> str:
    """Generate a prediction for a given question using a specified prompt template, extract, and evaluate the final answer."""
    prompt = prompt_template.format(question=question)
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stop=None,
        top_p=1.0,
    )
    full_response = response.choices[0].message.content

    # Extract the final answer or program
    start = full_response.find("```") + 3  # Offset to skip the '```' itself
    end = full_response.find("```", start)
    if start > 1 and end > start:
        final_expression = full_response[start:end].strip()
        try:
            output_capture = io.StringIO()
            # Safely evaluate the expression
            final_expression = final_expression.replace("python\n", "") # TODO: fix this in the prompt
            # with redirect_stdout(output_capture):
            safe_globals = {"__builtins__": None, "math": __import__('math'), "print": print}
            safe_locals = {"result": None}
            exec(final_expression, safe_globals, safe_locals)
            # if output_capture.getvalue() != "":
            #     final_answer = output_capture.getvalue()
            final_answer = float(safe_locals["result"])

        except Exception as e:
            print(f"Error evaluating the expression: {e}")
            final_answer = "NaN"
    else:
        final_answer = "NaN"

    return str(final_answer), full_response


# We could use the code interpreter to run the code and get the answer. 
def compare_prompts(
    question: str, actual_answer: str, prompts: List[str], remainder: str
) -> List[Tuple[str, float]]:
    """Compare different prompts by generating predictions for each and calculating their log-scale loss."""
    results = []
    for prompt_template in prompts:
        predicted_answer, full_response = generate_prediction(question, prompt_template + "\n" + remainder)
        log_loss = calculate_log_loss(predicted_answer, actual_answer)
        print(f"Prompt: {prompt_template}\nFull Response: {full_response}\nLog Loss: {log_loss}\n")
        results.append((prompt_template, log_loss, full_response))
    return results


def main():
    # Example question and actual answer
    question = "If I were to take an English penny, melt it down, then somehow stretch it out into a perfect hollow sphere 1 atom thick, how large would this sphere be?"
    actual_answer = "2.3e+8"
    # TODO: seperate out units in answer
    unit = "in**2"
    remainder = f"Give your answer in {unit}. Provide the final answer as a Python executable expression or final program within '```' as brackets that sets the `result` variable to the final value."

    

    # Compare prompts
    comparison_results = compare_prompts(question, actual_answer, prompts, remainder)

    # Sort results by log loss, lower is better
    comparison_results.sort(key=lambda x: x[1])

    # Display results
    for prompt, loss, _ in comparison_results:
        print(f"Prompt: {prompt}\nLog Loss: {loss}\n")


if __name__ == "__main__":
    main()
