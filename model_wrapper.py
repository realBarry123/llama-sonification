import torch
import math

class ModelWrapper():

    def __init__(self):
        import os
        from dotenv import load_dotenv
        import config

        os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
        from huggingface_hub import login, get_token
        from transformers import AutoTokenizer, AutoModelForCausalLM

        load_dotenv()
        CACHE_PATH = os.getenv("CACHE_PATH")

        if get_token() is None:
            if os.getenv("HF_TOKEN") is not None: 
                login(os.getenv("HF_TOKEN"))
            else:
                raise RuntimeError("Please login to huggingface or set environment variable HF_TOKEN")
                
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_REPO, cache_dir=CACHE_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_REPO, cache_dir=CACHE_PATH)
        self.chunk_size = config.CHUNK_SIZE
        self.context_limit = config.CONTEXT_LIMIT
        self.next_seq_idx = -1  # lag behind state index
        self.next_state_idx = 0
        self.context = None
        self.history = ""
        self.hidden_states = None
        self.filler_token_id = self.tokenizer.encode("…", add_special_tokens=False)[0]
        self.trim_length = 1
        self.verbose = False
        self.seed_length = 0
        self.temp_mean = config.TEMP_MEAN
        self.temperature = self.temp_mean
        self.temp_period = config.TEMP_PERIOD  # every hour
        self.temp_amplitude = config.TEMP_AMPLITUDE
        self.count = 0
        # assert self.trim_length >= self.chunk_size
    
    def seed(self, text):
        if self.verbose: print(f"Called seed(\"{text}\")")
        self.context = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self.seed_length = len(self.context["input_ids"][0])
        self._generate(self.chunk_size)

    def _trim_context(self):
        if self.verbose: print(f"Called _trim_context")
        self.history += self.tokenizer.batch_decode(self.context["input_ids"][:, :self.trim_length])[0]
        self.context["input_ids"] = self.context["input_ids"][:, self.trim_length:]

        self.next_seq_idx -= self.trim_length

    def _generate(self, tokens):
        if self.verbose: print(f"Called _generate")
        self._update_temperature()
        with torch.no_grad():
            output = self.model.generate(
                **self.context,
                max_new_tokens=tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=False,
                temperature=self.temperature,
                eos_token_id=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        self.context["input_ids"] = output.sequences

        self.context["attention_mask"] = torch.ones_like(output.sequences)
        self.hidden_states = output.hidden_states

        if self.verbose: print(f"New lengths: sequence={len(output.sequences[0])}, hidden_states={len(self.hidden_states)}")
    
    def next(self):
        if self.context == None:
            raise RuntimeError("model context not initialized. Did you call `seed`?")

        if len(self.context["input_ids"][0]) >= self.context_limit:
            self._trim_context()
        
        if self.next_state_idx == self.chunk_size:
            self._generate(self.chunk_size)
            self.next_state_idx = 0
        
        if self.next_seq_idx >= 0:  # prevent -1 from indexing
            next_token = self.context["input_ids"][0][self.next_seq_idx + self.seed_length]
            next_token = self.tokenizer.decode(next_token)
            next_token = self._process_newlines(next_token)
        else:
            next_token = " "

        # Slice from the hidden states (rather than indexing)
        next_state = self.hidden_states[self.next_state_idx : self.next_state_idx+1]

        self.next_seq_idx += 1
        self.next_state_idx += 1
        self.count += 1
        return next_token, self._format_hidden_states(next_state), self.get_context()
    
    def _update_temperature(self):
        self.temperature = self.temp_amplitude * math.sin(2 * math.pi / self.temp_period * self.count) + self.temp_mean
        if self.verbose: print(f"Temperature: {self.temperature}")

    @staticmethod
    def _format_hidden_states(hidden_states):
    
        steps = []

        for step in hidden_states:
            layers = torch.stack(
                [layer.squeeze(0)[-1] for layer in step]
            )
            steps.append(layers)

        return torch.stack(steps).to("cpu") # (time, layers, hidden)

    @staticmethod
    def _process_newlines(txt):
        txt = txt.replace("\n", "")
        if txt == "":
            return " "
        return txt
    
    def _get_context(self):
        return self.tokenizer.batch_decode(self.context["input_ids"])[0]
    
    def get_context(self):
        # print(self.tokenizer.batch_decode(self.context["input_ids"][:self.next_seq_idx])[0])
        return self.tokenizer.batch_decode(
            self.context["input_ids"][:, :self.next_seq_idx + self.seed_length]
        )[0]

    def write_context(self, file_path: str):
        text = self._get_context()
        if text[:17] == "<|begin_of_text|>":
            text = "\n" + text
        with open(file_path, "a") as f:
            f.write(text)

    def write_history(self, file_path: str):
        text = self.history + self._get_context()
        if text[:17] == "<|begin_of_text|>":
            text = "\n" + text
        with open(file_path, "a") as f:
            f.write(text)