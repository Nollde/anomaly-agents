# Anomaly Agents - Hackathon Project Setup

## Hackathon Context

We are a team of three researchers participating in a hackathon at the **Center for Decoding the Universe at Stanford**. The invitation reads:

> As AI agents become central to scientific discovery at the Center for Decoding the Universe, we need systematic evidence on when they aid replication and when they fail to do so. We will attempt to reproduce key results and analyses from research papers using AI agents, as well as directly with domain-expert scientists, documenting what works and what doesn't.
>
> The day will consist of two morning keynotes, followed by hands-on hacking sessions, and will conclude with individual or team presentations.
>
> We encourage you to join together with your research group, as this activity is particularly beneficial for students and postdoctoral researchers. We especially encourage you to bring a paper to replicate (ideally with available data/code).
>
> Through this hackathon, we are also building community across KIPAC, SLAC, HAI/SDS, and CS while advancing our understanding of AI-enabled science.

## Project Goal

We want to reproduce the core result of a methods paper called **"Classifying Anomalies THrough Outer Density Estimation (CATHODE)"**.

We have put this paper and other related papers about the used data and methods in the directory `source`.

## Approach

Our reproduction relies majorly on an AI Agent (you) which helps us with the implementation and execution of the analysis.

Our approach to the challenge is **modular**, whereas the reimplementation of the main result of the paper is only the last module. The other modules lead up to the final results.

## Project Stages

We have four different stages, each containing one or multiple tasks:

### 1. Paper Understanding (BSc level)
We want to understand if the Agent can extract basic knowledge from the paper:
- **Easy task**: Extract a number from the text
- **Middle task**: Extract a number from a plot

### 2. Data Understanding (MSc level)
We want to see that the Agent can:
- Download and access the dataset of the challenge
- Create plots of key features
- Interpret those plots
- Come up with appropriate normalization strategies

### 3. Method Understanding (PhD Level)
We want the Agent to reproduce key methods in the analysis:
- **CATHODE method**: Conditional density estimation
- **CWOLA method**: Weakly supervised classifier

The results of those methods do not necessarily have to be done on the dataset of this study but can be employed on toy data.

### 4. Paper Reproduction (PostDoc Level)
We want the agent to basically reproduce the paper, most importantly:
- **Figure 7 (left)**: The dependence of the "Maximum Significance Improvement" in dependence of the fraction of the number of inclusively injected signal events and the number of background events

### 5. Future Outlook (Professor Level)
CATHODE is one of four anomaly detection methods which are compared and combined in the paper "The Interplay of Machine Learning–based Resonant Anomaly Detection Methods". The full reconstruction of all four methods should be denoted as "Professor Level" and be left for future investigation.

## Agent Instructions

We will give instructions to the agent in the form of a markdown file named `instructions.md`. These instructions shall include:

- **Role**: You are a very smart physicist with high coding skills, a great analytic understanding. You are creative but you follow the instructions meticulously. If you have an additional idea which goes beyond what we told you, you can ask to implement it if it is beneficial for the project.

- **Task Guidance**: To guide you, we have prepared many different tasks which lead up to the final result. Each task has a name, a difficulty, and a weight. We have prepared a specific prompt for every task in which we explain to you what to do. You can find the tasks in `tasks.md`. Afterwards we will score each task to see how good you have done.

- **Implementation**: Everything should be implemented in Python. You are running in a Docker container on a node which possesses a GPU. The necessary software is installed. If you need more than the installed software you may ask for it.

- **Code Quality**: We want you to write very clean code. An emphasis is on our understanding of what you do. For this, we want you to use the python package `law` (based on luigi) which is well used in data analysis in high energy physics. Use Mixin classes to factorize the law parameters across different tasks to make the tasks more legible. In the `requires(self)` function, use `.req` to forward the necessary parameters. Only use the local-scheduler, add this to the `law.cfg`.

- **Code Structure**:
  - The code shall be modular
  - All tasks should be in a file at the root level of the project called `tasks.py`
  - All necessary functions and other code should be in appropriate files in the directory `src`

- **Environment Setup**: Create and maintain a file `setup.sh` which sets up the necessary environment variables (e.g. puts the working directory in the PYTHONPATH) and others.

- **Code Formatting**: The code shall be formatted with black using a maximum line length of 100 characters

- **Version Control**: We want you to commit the code to git in between major changes:
  - Git messages for normal development should be prefixed by `[dev]`
  - Fixes should be prefixed by `[fix]`
  - Changes which only have to do with git should be prefixed `[git]`

- **Task Naming**: You might solve each task with exactly one or more law tasks. If you use more than one law task, use a very clear naming scheme so we can understand what you are working on.

- **Documentation**:
  - We want to hand in our whole conversation at the end, for this, we ask you to maintain a file `communication.md` which contains our entire communication
  - We also want you to maintain a file `progress.md` which contains our progress, including summarized sections of our communication with remarks such that it is best understandable by a human, the steps we take, including the reasons for those steps, and intermediate as well as end results

- **Notes**:
  - We have one file `notes.md` which is our file to maintain notes along the hackathon and especially along the execution of the different tasks. You might help to set up placeholders for our notes (general and task specific) but you might not add content.
  - We have one file `_.md` which is our personal file to keep track of things outside our communication, you might read the file, but you can not make edits

- **Resources**: All project sources in the form of papers can be found in the directory `source`

- **Communication Style**: Be very friendly and sympathetic, but also be efficient in your communication

## Additional Guidance from Initial Setup

During the initial project setup, the following clarifications were provided:

- **Pitch Slide**: Keep presentation materials concise and focused on a single slide with bullet points (8-10 words max per point). Use marketing language to make the project appealing. The pitch slide should NOT be committed to git.

- **Documentation Focus**: The core documentation files (README, setup, instructions, tasks, notes) form the backbone of the project and should be maintained in git. Personal presentation materials remain separate.

- **Primary Target**: The main hackathon goal is to reproduce **Figure 7 (left)** - the dependence of Maximum Significance Improvement on signal-to-background ratio. This is the critical deliverable.

## Project Structure

```
anomaly_agents/
├── README.md                 # Project overview
├── setup.md                  # This file - project setup instructions
├── instructions.md           # Instructions for the AI agent
├── tasks.md                  # Detailed task breakdown
├── notes.md                  # Team notes (editable by team only)
├── _.md                      # Personal notes (readable by agent, not editable)
├── communication.md          # Full conversation log
├── progress.md               # Progress summary and results
├── setup.sh                  # Environment setup script
├── law.cfg                   # Law configuration
├── tasks.py                  # Law tasks implementation
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── cathode/            # CATHODE implementation
│   ├── cwola/              # CWOLA implementation
│   ├── data/               # Data handling
│   ├── plotting/           # Visualization utilities
│   └── utils/              # General utilities
└── source/                   # Papers and references
    ├── 2109.00546v3.pdf     # CATHODE paper
    ├── 2307.11157v2.pdf     # Interplay paper
    └── methods/             # Method papers
        └── data/            # Data papers
```
