# Motivation:

**TLDR: I looked into how much it would take to fine-tune gpt-4 to do Fermi estimates better. If you liked** [**the post/paper**](https://www.lesswrong.com/posts/K2F9g2aQubd7kwEr3/approaching-human-level-forecasting-with-language-models-2) **on fine-tuning Language models to make predictions you might like reading this. I evaluated gpt-4 on the first dataset I found, but gpt-4 was already making better fermi estimates than the examples in the dataset, so I stopped there. See infos on how to replicate my results in the next section**

First problem I encountered: there is no public access to fine-tuning gpt-4 so far. Ok, we might as well just do gpt-3.5 I guess.

First, I found this [Fermi estimate dataset](https://github.com/allenai/fermi). (While doing this, I was thinking I should perhaps search more widely what kind of different AI benchmarks exist, since probably a dataset that is evaluating a similar capability is already out there, but I don't know its name.)

Next I looked at [this paper](https://ceur-ws.org/Vol-3551/paper9.pdf), where people used among other gpt-3.5 and gpt-4 on this benchmark. Clearly these people weren't even trying, though, because gpt-4 does worse than gpt-3.5. One of the main issues I saw was that they were trying to make the LLM output the answer as a program in the domain specific language used in that dataset. They couldn't even get the LLM to output valid programs more than 60% of the time (their metric compares on a log scale, if the answer by the LLM is within 3 orders of magnitude of the real answer. 1 is best 0 is more than 3 orders of magnitude away: fp-score(x) = max(0,1-1/3 * | log_10(prediction/answer)|)).

![image](https://i.imgur.com/sS6keul.png)

My conjecture was that just using python instead should give you better results.(This turned out to be true). I get a mean score of ~0.57 on 100 sample problems, so as good results with gpt-4-turbo as they get when they first provide “context” by giving the llm the values for the key variables needed to compute the answer (why would this task even still be hard at all?).

When gpt-4 turned out to get a worse fp-score than gpt-4-turbo on my 10 samples, I got suspicious. After looking at samples that gpt-4 got a bad score on, it became clear that this was to blame on bad quality of the dataset. 2 answers were flat-out not using the correct variables/confused, while gpt-4 was answering correctly. Once, the question didn't make clear what unit to use. For 2 other samples gpt-4 gave a better answer. Once, using a better approach (using geometry instead of wrong figures of how much energy the earth gets from the sun, to determine the fraction of sun energy that the earth receives). Once, by having better numbers, input estimates like how many car miles are driven in total in the US.

So on this dataset, gpt-4 seems to be already at the point of data-saturation. I was actually quite impressed how well it was doing. When I had tried using gpt-4 for this task, I had always felt like it was doing quite badly. One guess I have is this is because when I ask gpt-4 for an estimate, it is often a practical question, which is actually harder than these artificial questions. In addition, the reason I ask gpt-4 is that the question is hard, and I expect to need to employ a lot of cognitive labor to do it myself.

Another data point with respect to this was the “Thinking physics exercises”. Which [I tried with some of my friends](https://www.lesswrong.com/posts/PiPH4gkcMuvLALymK/exercise-solve-thinking-physics#MWbkF74caCaw39Abz). For that task, gpt-4 was better than people who were bad at this, but worse than people who were good at this (and given 5–10 minutes of thinking time) (although I did not rigorously evaluate that). GPT-4 is probably better than most humans at doing Fermi estimates given 10 minutes of time. Especially in domains one is unfamiliar with, since it has so much more breadth.

I would be interested to see what one would get out of actually making a high quality dataset by taking Fermi estimates from people I deem to produce high quality work in that area.


# Project Setup and Dependencies

This project uses Python to evaluate GPT models on Fermi estimation tasks. To replicate this project, you'll need:

1. Python =>3.10
2. OpenAI API key

## Dependencies
- OpenAI Python library

## Setup
1. Install the required Python packages:
   ```
   pip install openai
   ```

2. Set up your OpenAI API key in one of two ways:
   a. Set an environment variable named `OPENAI_API_KEY` with your API key.
   b. (Optional) Use 1Password CLI for secure API key storage.

Note: The code uses GPT-4 by default. Adjust the `model` variable in the `main()` function if you want to use a different model.

## Running the Project
Execute the main script to run the evaluation:

```
python fermi.py
```

This will generate results in the `./results/` directory.


