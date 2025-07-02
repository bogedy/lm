import torch
import attention
import transformer

def test_attention_scores():
    # case 1
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    b = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])  # (1, 2, 2)
    expected_output = torch.tensor([[[17.0, 23.0], [39.0, 53.0]]]) / (2.0 ** 0.5)

    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output)

    # case 2 (scaled)
    a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])  # (1, 2, 2)
    b = torch.tensor([[[0.5, 0.6], [0.7, 0.8]]])  # (1, 2, 2)
    expected_output = torch.tensor([[[0.17, 0.23], [0.39, 0.53]]])  / (2.0 ** 0.5)

    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output, atol=1e-6)

    print("Attention scores done :)")

def test_self_attention():
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    A = torch.tensor([[[1.0, float('-inf')], [0.0, 0.0]]])   # (1, 2, 2) attention scores
    # the softmax of these two attention scores is [1., 0.] and [0.5, 0.5]
    
    # Expected output:
    # For first token: 1.0*[1,2] + 0.0*[3,4] = [1,2]
    # For second token: 0.5*[1,2] + 0.5*[3,4] = [2,3]
    expected_output = torch.tensor([[[1.0, 2.0], [2.0, 3.0]]])
    
    sa = attention.self_attention(v, A)
    if not torch.allclose(sa, expected_output):
        print(sa, expected_output)
        assert False


    # causal masking test
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
    A = torch.ones(1, 3, 3) * 0.5  # uniform attention scores
    mask = attention.create_causal_mask(0, 0, 3)  # args don't matter for this mask
    
    # With causal mask, future tokens should be masked (-inf)
    # Expected softmax:
    # [[1.0, 0.0, 0.0],
    #  [0.5, 0.5, 0.0],
    #  [0.33, 0.33, 0.33]]
    expected_output = torch.tensor([
        [[1.0, 2.0], 
         [2.0, 3.0], 
         [3.0, 4.0]]
    ])
    
    sa = attention.self_attention(v, A, mask)
    if not torch.allclose(sa, expected_output, atol=0.01):
        print(sa, expected_output)
        assert False
    
    print("self_attention test passed :)")

test_attention_scores()
test_self_attention()

def test_multi_head_attention():
    # Test multi-head attention with 2 heads
    embed_dim = 4
    n_heads = 2
    max_context_len = 3
    
    # Create input tensor (batch_size=1, seq_len=2, embed_dim=4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    
    # Initialize multi-head attention
    mha = attention.CausalSelfAttention(embed_dim, n_heads, max_context_len)
    
    # Forward pass
    output = mha(x)
    
    # Basic checks
    assert output.shape == x.shape  # shape should be preserved
    assert not torch.allclose(output, x)  # output should be different from input
    
    print("multi_head_attention test passed :)")

def test_causal_mask():
    # Test the full implementation with multiple layers
    embed_dim = 8
    n_heads = 2
    max_context_len = 5
    
    # Create random input
    x = torch.randn(2, 4, embed_dim)
    
    # Initialize attention
    attn = attention.CausalSelfAttention(embed_dim, n_heads, max_context_len)
    
    # Forward pass
    output = attn(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check causal property - later positions shouldn't affect earlier ones
    x_clone = x.clone()
    x_clone[:, 2:, :] = torch.randn(2, 2, embed_dim)  # change later positions
    output_clone = attn(x_clone)
    assert torch.allclose(output[:, :2, :], output_clone[:, :2, :])  # first 2 positions should be same
    assert not torch.allclose(output[:, 2:, :], output_clone[:, 2:, :])
    print("causal_mask test passed :)")

# Add these to the test execution
test_multi_head_attention()
test_causal_mask()

def test_embed():
    vocab_size = 100
    embed_size = 8
    max_context_len = 10
    
    embed = transformer.Embed(vocab_size, embed_size, max_context_len)
    x = torch.randint(0, vocab_size, (2, 3))
    output = embed(x)
    
    assert output.shape == (2, 3, embed_size)
    
    # check that different tokens produce different embeddings
    x2 = x.clone()
    x2[0, 0] = (x[0, 0] + 1) % vocab_size  # change one token
    output2 = embed(x2)
    assert not torch.allclose(output[0,0], output2[0,0])
    
    # check position embeddings are different
    x3 = torch.tensor([[1, 1, 1], [1, 1, 1]])  # same token at all positions
    output3 = embed(x3)
    assert not torch.allclose(output3[0,0], output3[0,1])  # different positions should differ
    
    print("Embed test passed :)")

def test_transformer_decoder_block():
    n_heads = 2
    embed_size = 8
    mlp_hidden_size = 16
    max_context_len = 10
    
    decoder_block = transformer.TransformerDecoderBlock(
        n_heads, embed_size, mlp_hidden_size, max_context_len
    )
    
    # random input (batch_size=2, seq_len=4, embed_size=8)
    x = torch.randn(2, 4, embed_size)
    output = decoder_block(x)
    
    # check output shape
    assert output.shape == x.shape
    
    # test with residuals
    decoder_block_res = transformer.TransformerDecoderBlock(
        n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals=True
    )
    output_res = decoder_block_res(x)
    assert output_res.shape == x.shape
    
    # check residuals make a difference
    assert not torch.allclose(output, output_res)
    
    print("TransformerDecoderBlock test passed :)")

# Add these to the test execution
test_embed()
test_transformer_decoder_block()