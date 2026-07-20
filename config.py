HF_ENDPOINT = "https://huggingface.co"
# If running from China: 
# HF_ENDPOINT = "https://hf-mirror.com"

MODEL_REPO = "meta-llama/Llama-3.2-1B"

# Token generation
SEED = "Hello, world" # starting sequence for generation
CHUNK_SIZE = 4 # how many tokens to generate in advance
CONTEXT_LIMIT = 28 # how many tokens to keep before cropping
WRITE_HISTORY = True # if True, saves token history in context.txt

# Temperature oscillator controls
TEMP_MEAN = 2.0
TEMP_PERIOD = 30 * 60 # unit: tokens
TEMP_AMPLITUDE = 1.5

# Frequency mapping
FREQ_LOWER = 20
FREQ_UPPER = 20000
MAX_Z = 2

# Sonification settings
FS = 44100
GAIN = 0.4
CHANNEL_SPLIT = ("-b", "-f")
NOTE_LENGTH = 2/17

# Display settings
VW = 1512
VH = 982
FULL_SCREEN = True