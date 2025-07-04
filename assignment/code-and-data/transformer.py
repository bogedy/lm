from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, dropout_rate = 0.1):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        if self.with_residuals:
            # add residuals support.
            x1 = inputs
            x = self.layer_norm_1(x1)
            x2 = self.dropout(self.causal_attention(x)) + x1
            x = self.layer_norm_2(x2)
            x = self.mlp(x) + x2
            return x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.dropout(self.causal_attention(x))
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size) #  set the right values
        self.position_embeddings = nn.Embedding(max_context_len, embed_size) #  set the right values
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        B, N = x.size()
        tok_embeddings = self.token_embeddings(x)
        positions = torch.arange(N, device=x.device).expand(B, N)
        pos_embeddings = self.position_embeddings(positions)

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            init_method: str,
            dropout_rate: float = 0.1,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, dropout_rate) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        self.init_method = init_method

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        init_method = self.init_method # Default, will be overridden by the argument passed to __init__
        
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                if init_method == 'xavier':
                    torch.nn.init.xavier_uniform_(p.weight)
                elif init_method == 'kaiming':
                    torch.nn.init.kaiming_uniform_(p.weight, mode='fan_in', nonlinearity='relu')
                else:  # normal
                    torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, device=None) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                feed_tensor = torch.tensor([feed_to_lm], dtype=torch.int32)
                if device:
                    feed_tensor = feed_tensor.to(device)
                logits = self(feed_tensor)
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        raise Exception("Not implemented")
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.

