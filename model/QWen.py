
import sys
import os
work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2Config, Qwen2ForCausalLM

from collections import namedtuple
from data.constants import sm_tokenizer


class DecoderConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.encoder(x)  # (B, hidden_dim, H, W)
        B, D, H, W = feats.size()
        return feats.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, Seq_len, D)


# Wrapper Module
class QwenDecoder(nn.Module):
    def __init__(self, decoder_config = None): # , encoder_hidden=768
        super().__init__()
        
        
        if decoder_config is None:
            decoder_config = Qwen2Config(
                vocab_size=1024,
                hidden_size=1024,
                intermediate_size=2048,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=8,
                max_position_embeddings=2048,
                attention_dropout=0.1,
                use_cache=False,
                is_decoder=True,
                bos_token_id=sm_tokenizer.BOS,
                eos_token_id=sm_tokenizer.EOS,
                pad_token_id=sm_tokenizer.PAD,
                # tie_word_embeddings=True,
                # tie_encoder_decoder_embeddings=True,
                use_flash_attention=True,
                # use_rotary_embeddings=True,
                # rotary_embedding_base=10000,
                # rotary_embedding_base_power=1.0,
                # rotary_embedding_dim=64,
                # rotary_embedding_max_position_embeddings=2048,
                # rotary_embedding_type="standard",
                # rotary_embedding_use_cache=False,
                # hidden_act="relu"
            )
        
        # self.encoder = CNNEncoder(in_channels=1, hidden_dim=encoder_hidden)
        
        # self.decoder = AutoModelForCausalLM.from_pretrained(qwen_model_name)
        self.decoder = AutoModelForCausalLM.from_config(decoder_config)
        # AutoModelForCausalLM.register("QwenDecoder", self.decoder.config)
        
        # Ensure encoder dim matches decoder input embeddings
        # decoder_embed_dim = self.decoder.transformer.hidden_size
        # if encoder_hidden != decoder_embed_dim:
        #     self.encoder_proj = nn.Linear(encoder_hidden, decoder_embed_dim)
        # else:
        #     self.encoder_proj = nn.Identity()

    def forward(self, input_ids = None, inputs_embeds=None,  attention_mask=None, labels=None):
        #  image_input, decoder_embeddings, decoder_input_ids
        # Encode image
        # encoder_feats = self.encoder(image_input)  # (B, Seq_len, H)
        # encoder_feats = self.encoder_proj(encoder_feats)
        
        # encoder_feats = image_input

        # Get decoder input embeddings
        # if decoder_input_ids is not None:
        #     assert decoder_embeddings is None, "Provide either decoder_input_ids or decoder_embeddings, not both."
        #     decoder_embeddings = self.decoder.model.embed_tokens(decoder_input_ids)

        # Concatenate encoder feats + decoder embeddings
        # full_input = torch.cat([encoder_feats, decoder_embeddings], dim=1)

        # Construct attention mask (1 for real tokens, 0 for padding)
        # encoder_len = encoder_feats.size(1)
        # decoder_len = decoder_input_ids.size(1)
        # attention_mask = torch.ones((image_input.size(0), encoder_len + decoder_len), dtype=torch.long).to(image_input.device)

        # Call decoder model
        outputs = self.decoder(input_ids = input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs # [:, encoder_len:, :]  # Return only decoder token logits
    
    
if __name__ == "__main__":
    
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1-0-0.5B")
    decoder_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    decoder_config = decoder_config.to_dict()
    # decoder_config["bos_token_id"] =  1
    # decoder_config["eos_token_id"] =  2
    
    # decoder_config["intermediate_size"] = 4096
    # decoder_config["num_hidden_layers"] = 8
    # decoder_config["num_attention_heads"] = 8
    # decoder_config["vocab_size"] = 1024
    # decoder_config["max_position_embeddings"] = 2048
    # decoder_config["hidden_size"] = 1024
    # decoder_config["attention_dropout"] = 0.1
    
    # decoder_config = DecoderConfig(**decoder_config) #namedtuple("DecoderConfig", decoder_config.keys())(*decoder_config.values())
    
    decoder_config = Qwen2Config(
        vocab_size=1024,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        attention_dropout=0.1,
    )
    
    # AutoConfig.register("QwenDecoder", decoder_config)
    # Example decoder config for a smaller Qwen model
    
    
    # decoder_config = {
    #     'model_name': "Qwen/Qwen1.5-0.5B",  # Small Qwen model
    #     'vocab_size': 1024,
    #     'hidden_size': 1024,
    #     'num_hidden_layers': 6,
    #     'num_attention_heads': 8,
    #     'pad_token_id': 0
    # }
    model = QwenDecoder(decoder_config)
    
    print(model)
    
    # Example input
    image_input = torch.randn(2, 1, 224, 224)  # Batch of 2 images
    text_input = "Hello, how are you?"
    # decoder_input_ids = tokenizer(text_input, return_tensors="pt").input_ids
    full_input = torch.randn(2, 10, 1024)  # Example full input embeddings
    
    attention_mask = torch.ones((10, 10))  # Example attention mask
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)
    
    # Forward pass
    logits = model(full_input, attention_mask=attention_mask)  # Example attention mask
    print(logits.shape)  # Should be (batch_size, seq_len, vocab_size)
    exit()