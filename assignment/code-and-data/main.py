from __future__ import annotations
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_path', type=str, default="data/", help='Path to the data')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--embed_size', type=int, default=192, help='Embedding size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--num_batches_to_train', type=int, default=50000, help='Number of batches for training')
    parser.add_argument('--gradient_clipping', type=float, default=None, help='Gradient clipping value')
    parser.add_argument('--init_method', type=str, default='normal', 
                        choices=['normal', 'xavier', 'kaiming'],
                        help='Weight initialization method (normal/xavier/kaiming)')
    parser.add_argument('--with_residuals', action='store_true', 
                    help='Whether to use residual connections in the transformer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, 
                      help='Dropout rate (0 to disable dropout)')
    return parser.parse_args()


if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from transformer import TransformerLM
    import data
    import lm
    from datetime import datetime
    import random
    import os
    import csv

    args = parse_args()

    seq_len = args.seq_len
    batch_size = args.batch_size
    data_path = args.data_path
    n_layers = args.n_layers
    n_heads = args.n_heads
    embed_size = args.embed_size
    mlp_hidden_size = embed_size * 4
    learning_rate = args.learning_rate
    gradient_clipping = args.gradient_clipping
    num_batches_to_train = args.num_batches_to_train

    def generate_output_dir(base_dir="model_outputs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_str = f"layers_{n_layers}_heads_{n_heads}_embed_{embed_size}"
        output_dir = os.path.join(base_dir, f"{params_str}_{timestamp}_{random.randint(0, 9999)}") # add a random int because im paranoid that ill overwrite something lol
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    output_dir = generate_output_dir()

    # initialize loss CSV file
    with open(os.path.join(output_dir, 'losses.csv'), mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Batch', 'Train Loss', 'Test Loss'])

    tokenizer, train_data, test_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.

    tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))
    
    train_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
    test_iter = iter(data.RandomOrderDataIterator(test_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = args.with_residuals,
            init_method = args.init_method,
            dropout_rate = args.dropout_rate
        )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    model.train()

    loss_data = []
    num_batches = 0
    while True:
        for batch in data.batch_items(train_iter, batch_size):
            if num_batches >= num_batches_to_train: break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y, ignore_index=tokenizer.pad_id())

            # parameters update
            model.zero_grad()
            loss.backward()
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            train_loss = loss.item()
            loss_data.append((num_batches, train_loss, None))

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    model.eval()
                    test_loss_total = 0
                    test_batches = 0
                    n_iter = 10 # there's too much test data to test it all
                    with torch.no_grad():
                        for test_batch in data.batch_items(test_iter, batch_size):
                            test_batches += 1
                            test_x, test_y = lm.batch_to_labeled_samples(test_batch)
                            test_x = test_x.to(device)
                            test_y = test_y.to(device)

                            test_logits = model(test_x)
                            test_loss = lm.compute_loss(test_logits, test_y, ignore_index=tokenizer.pad_id())
                            test_loss_total += test_loss.item()
                            if test_batches == n_iter: break

                    avg_test_loss = test_loss_total / test_batches
                    loss_data[-1] = (num_batches, train_loss, avg_test_loss)  # Update last entry with test loss

                    print(f"Average test loss: {avg_test_loss}")

                    sampled = ""
                    for _ in range(1):
                        start_token = tokenizer.tokenize("Hello")
                        sampled = tokenizer.detokenize(model.sample_continuation(start_token, 500, device=device))
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")

                if num_batches % 500 == 0:
                    torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint_{num_batches}.pth'))
                    with open(os.path.join(output_dir, f'sample_{num_batches}.txt'), 'w') as sample_file:
                        sample_file.write(sampled)
                    
                    with open(os.path.join(output_dir, 'losses.csv'), mode='w', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(['batch', 'train', 'test'])
                        csv_writer.writerows(loss_data)
