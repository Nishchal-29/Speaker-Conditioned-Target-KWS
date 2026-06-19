# Your Project Name

- **Problem Statement Number** - 
- **Problem Statement Title** - *(Must exactly match one of the 11 Samsung EnnovateX AX Hackathon Problem Statements)*
- **Team name** - *(Same as Phase 1 Team name)*
- **Team members (Names)** - *Member 1 Name*, *Member 2 Name*
- **Institute/College Name** - *Name*, *Campus Name & Address (In case the institute has multiple campuses)*
- **Final Presentation Google Drive Link** - *Upload the PDF presentation for your final submission on Google Drive (It should be openly accessible and not behind any login wall)*
- **Full Submission Demo Video Link** - *(Upload the Demo video on Youtube as a public or unlisted video and share the link. Google Drive uploads for video is not allowed.)*
- **Setup & Result Reproducibility Video Link** - *(Upload the Demo video on Youtube as a public or unlisted video and share the link. Google Drive uploads for video is not allowed.)*

### Project Artefacts

- **Technical Documentation** - Create a **docs** folder and add all technical details in markdown files inside this folder explaining the project Technical Stack, List of OSS libraries/projects used along with their links, the technical architecture of your solution, implementation details, installation instructions, user guide, salient features of the projects. Kindly add screenshots wherever possible.
- **[Important]** Create a file `docs/ax.md` whiere you explain in detail how you utilizes open weight models and/or agentic development tools to implement your solution. Explain in detail your  Agentic AI setup , Agentic workflows, Reasoning & planning pipelines, Tool use / tool chaining, Coding assistants, agents, harness, MCP servers, agents.md, skills, Memory / context handling, Multi-agent orchestration systems, etc. Please highlight from your experience - what worked and **what did not work**.
- **Source Code** - Create a **src** folder and add all developed project source codes (including training & benchmark evaluation codes) in the repo. The code must be capable of being successfully installed/executed and must run consistently on the intended platforms.
- **Models Used** - *(Hugging Face links to all models used in the project. You are permitted to use only open weight models.)*
- **Models Published** - *(In case you have developed a model as a part of your solution, kindly upload it on Hugging Face under appropriate open source license and add the link here.)*
## Datasets Used

### 1. VoxCeleb1 & VoxCeleb2
- **Source:** https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **Description:** VoxCeleb is a large-scale audio-visual speaker recognition dataset collected from real-world YouTube videos. It contains over one million utterances from thousands of speakers recorded under diverse acoustic conditions, including background chatter, overlapping speech, room reverberation, and varying recording devices.
- **Usage in Project:** Used to finetune the ECAPA-TDNN model on PCEN frontend.

### 2. MUSAN (Music, Speech, and Noise Corpus)
- **Source:** https://www.openslr.org/17/
- **Description:** MUSAN is a corpus containing music, speech, and a diverse collection of technical and non-technical noise recordings. It was specifically designed for tasks such as voice activity detection, speech/music discrimination, and robustness enhancement through noise augmentation. 
- **Usage in Project:** Used in training all the three Models, for injecting real world noises to match the required KPIs.

## Datasets Published

### TTS Corpus Dataset
- **Link:** https://huggingface.co/datasets/Nishchal-29/tts_corpus
- **Description:** A Text-to-Speech (TTS) corpus created for speech synthesis and speaker modeling experiments. The dataset contains recordings from **20 speakers**, with **4 utterances per speaker**, covering approximately **200 unique words**. It is designed to provide a compact, multi-speaker speech dataset suitable for prototyping, benchmarking, and educational research in TTS and speech processing.
- - **Usage in Project:** It is used for our custom keyword detection, This dataset consists of some custom keywords so that our model can generalize patterns for incoming new words.

#### Final Presentation

Unlike Phase 1 presentation, in Phase 2 you can freely decide the template, flow and content of your technical presentation. Ensure you cover all aspects of your solution - innovation, novelty, architecture, open datasets/models developed and used, final deliverable details, KPIs of your solution, AI/Agent use, any other details. 

#### Full Submission Demo Video

Create a high quality video demonstration your solution in real life and showcasing how it is actually solves the proposed AX Hackathon problem.

#### Setup & Result Reproducibility Video

To ensure reproducibility of results and to verify the presented KPIs, we require you to create a video demonstrating:
- Step by step project installation,
- Data/model download steps, 
- Execution of all required codes to train the developed models (if any)
- Execution of all evaluation codes to reproduce the presented results/KPIs 

### Attribution 

In case this project is built on top of an existing open source project, please provide the original project link here. Also, mention what new features were developed. Failing to attribute the source projects may lead to disqualification during the time of evaluation.
