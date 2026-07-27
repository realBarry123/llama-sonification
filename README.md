<div id="toc" align="center">
  <ul style="list-style: none">
    <summary>
      <h1>Language Model Sonification</h1>
    </summary>
  </ul>
</div>

<video src="https://github.com/user-attachments/assets/7cfe200b-5b23-4b8e-80ff-67a344180ee0" width="100px"></video>

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

$$\mathrm{freq}(x) := \frac{|x|}{z\sigma_x}(f_{upper}-f_{lower}) + f_{lower}$$
$$z=2, \space\space f_{lower}=20, \space\space f_{upper}=20000$$

where $x$ is the activation level and $\sigma_x$ is the standard deviation of $x$ across a single forward pass. 

Network layers are played one after another, with frequencies resulting from all activations in the same layer played simultaneously as sine waves. Each forward pass plays for 2 seconds. 

The model generates tokens with a periodically fluctuating temperature (approx. 1h, 1.25–2.75) and from a fixed-length context that is cropped with each new token. Since hidden states give rise to the output token, I have decided to show each token on screen **after** the sonification of its generating timestep has played. 

The model is [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), with 17 hidden layers of 2048 dimensions each. It's also pretty straightforward to go edit `config.py` and change it into any other Hugging Face transformer you want. Keep in mind that bigger models take longer to run, and if inference and sonification combined exceeds 2 seconds, it's game over. 

## Empirical Results

It's not really the goal of this project to draw useful correlations between the model and its sound. That being said, there are observations worth noting that, if nothing else, show that the sonification does reflect certain properties of the forward pass. 

One observation is that the sonification eventually "settles" into the lower frequencies as the model trains. This is apparent when sonifying the same token across different checkpoints of Pythia 70M, at intervals of 20,000 steps: 

<p align="center">
  <img width="60%" alt="pythia_learning_l" src="https://github.com/user-attachments/assets/5653f3b5-0164-42d3-b95e-9e5b90bb9cfc" />
</p>

Another interesting phenomenon is that the first token after a `<|begin_of_text|>` always sounds significantly different from the other ones, as shown here with Llama 3.2 1B: 

<p align="center">
  <img width="70%" alt="llama1b_bot" src="https://github.com/user-attachments/assets/b190148e-95e3-4667-95bd-ccc8b6c403eb" />
</p>

A similar thing happens in the Pythia models (410M shown here) for the first token in a new sequence:

<p align="center">
  <img width="70%" alt="pythia_eot" src="https://github.com/user-attachments/assets/65c9cb61-93fe-49a5-930e-bd317184688e" />
</p>

## Links

An older, slightly different, more deadpan description on my B21 profile → https://www.building21.ca/scholars/barry-yu

Petition for Building 21, the community that made this project possible → https://c.org/HLfXLz2txk
