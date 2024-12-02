# BPO

Implementation of Black-Box Prompt Optimization (BPO) for CAP6614

This project also requires API keys for the GPT and Claude models. You must set the following variables with appropriate API key:

| Model | Variable |
| ----- | -------- |
| GPT | `OPENAI_API_KEY` |
| Claude| `ANTHROPIC_API_KEY` |

To utlize Google cloud models like Text-Bison and Gemini, it requires getting set up with Google CLI. Start by downloading the SDK found here: https://cloud.google.com/sdk/docs/install

After following all steps in that guide, including initializing Google CLI with ``` gcloud init    ```, a user must log in and choose their unique project ID.

Additionally, a user must specify their project name in the ```generate_response_textbison()``` function in generate_responses.py.

Example:
```vertexai.init(project='bpo111', location="us-central1")```
