<div id="toc" align="right">
  <ul style="list-style: none">
    <summary>
      <h1>Language Model Sonification</h1>
    </summary>
  </ul>
</div>

*"What it is or what it does?"* 

This question was posed to me by Professor Ollivier Dyens during a discussion on language models earlier this year.

*"Aren't these two the same?"* I replied. 

*"... there's a difference."*

In 2026, interactions with language models are more accessible than ever before. Yet there is a growing divide between the user and the underlying model, often packaged under layers of chat-based or agentic abstraction. Inspired by the phenomenon of [semantic satiation](https://en.wikipedia.org/wiki/Semantic_satiation), this simple parameter-mapping sonification strips the model of "what it does" so that listeners may begin to explore "what it is."

This project was created as part of a fellowship at Building 21, McGill University in Winter 2026 ([building21.ca/scholars/barry-yu](https://www.building21.ca/scholars/barry-yu)). I would like to thank the Building 21 community as well as Andy S. Yu for their endlessly inspiring support. 

## Setup

*Note: these commands work on MacOS and probably Linux too.*

1. Fork the repo and install requirements:
```bash
git clone https://github.com/realBarry123/llama-sonification
cd llama-sonification
```
```bash
pip install -r requirements.txt
```

2. Optionally, create a `.env` file in the root that specifies a path in which to store the model cache (e.g. a hard drive). By default, cache is stored at `~/.cache/huggingface`.
```txt
CACHE_PATH="/Volumes/some-hard-drive/cache-file-name
```

3. Get permission to access [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and login to Hugging Face. 

4. Run the Pygame demo:
```bash
python display.py
```

## Method

During each forward pass, the hidden states at the last token position are taken (due to KV caching). Each activation level is mapped to a frequency via

$$\mathrm{freq}(x) := \frac{|x|}{2\sigma_x}(20000-20) + 20$$

where $x$ is the activation level and $\sigma_x$ is the standard deviation of $x$ across a single forward pass. 

Network layers are played one after another, with frequencies resulting from all activations in the same layer played simultaneously as sine waves. Each forward pass plays for 2 seconds. 

The model generates tokens with a periodically fluctuating temperature (approx. 1h, 1.25–2.75) and from a fixed-length context that is cropped with each new token. Since hidden states give rise to the output token, I have decided to show each token on screen **after** the sonification of its generating timestep has played. 

The model is [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), with 17 hidden layers of 2048 dimensions each. It's also pretty straightforward to go edit the first few lines of `model_wrapper.py` and change it into any other Hugging Face transformer you want. Keep in mind that bigger models take longer to run, and if inference and sonification combined exceeds 2 seconds, it's game over. 
