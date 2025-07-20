import data.symbolic_music_tokenizer as Tokenizer

sm_tokenizer = Tokenizer.SymbolicMusicTokenizer()

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 320#128
DEFAULT_NUM_MEL_BINS = 512
MEL_LENGTH = 512 #256#512
# DEFAULT_CLIP_TIME = DEFAULT_HOP_WIDTH*MEL_LENGTH/DEFAULT_SAMPLE_RATE
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
MEL_FMIN = 20.0
MEL_FMAX = 7600.0

MIDI_MIN = 21 # 21
MIDI_MAX = 108 # 108

SEQUENCE_POOLING_SIZE = sm_tokenizer.compound_word_size
TOKEN_END = sm_tokenizer.EOS
TOKEN_PAD = sm_tokenizer.PAD
TOKEN_START = sm_tokenizer.BOS
TOKEN_BLANK = sm_tokenizer.BLANK

VOCAB_SIZE = sm_tokenizer.vocab_size