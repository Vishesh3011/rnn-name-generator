import torch
import torch.nn as nn
import numpy as np
import string
import json
import os


def get_vocab():
    # Construct the character 'vocabulary'
    # we allow lowercase, uppercase and digits only, along with special characters:
    # "" - Empty string used to denote elements for the RNN to ignore
    # "<bos>" - Beginning of sequence token for the input the the RNN
    # "." - End of sequence token
    vocab = ["", "<bos>", "."] + list(string.ascii_lowercase + string.ascii_uppercase + string.digits + " ")
    id_to_char = {i: v for i, v in enumerate(vocab)} # maps from ids to characters
    char_to_id = {v: i for i, v in enumerate(vocab)} # maps from characters to ids
    return vocab, id_to_char, char_to_id


def load_data(filename):
    # read in the list of names
    data = json.load(open(filename, "r"))
    # append the end of sequence token to each name
    data = [v+'.' for v in data]
    return data


def seqs_to_ids(seqs, char_to_id, max_len=20):
    """Takes a list of names and turns them into a list of tokens ids.
    Responsible for padding sequences shorter than max_len with 0 so that all sequences are max_len.
    Also truncates names that are longer than max_len.
    Should also skip empty sequences if there are any.

    Args:
        seqs (list(str)): A list of names as strings.
        char_to_id (dict(str : int)): The mapping for characters to token ids
        max_len (int, optional): The maximum length of the ouput sequence. Defaults to 20.

    Returns:
        np.array: the names represented using token ids as 2d numpy array, 
            where each row corresponds to a name. The size of the array should be N * max_len
            where N is the number of non-empty sequences input. Padded with zeros if needed.
    """
    all_seqs = []
    for seq in seqs:
        if len(seq) == 0:
            continue
        char_ids = [0] * max_len
        for i, char in enumerate(seq):
            if i < max_len and char in char_to_id:
                char_ids[i] = char_to_id[char]
        all_seqs.append(char_ids)
    
    return np.array(all_seqs)


class RNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size = 32, gru_size=32):
        super(RNNLM, self).__init__()

        # store layer sizes
        self.emb_size = emb_size
        self.gru_size = gru_size

        # for embedding characters (ignores those with value 0: the padded values)
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # GRU layer
        self.gru = nn.GRU(emb_size, gru_size, batch_first=True)
        # linear layer for output
        self.linear = nn.Linear(gru_size, vocab_size)
    
    def forward(self, x, h_last=None):
        """Takes a batch of names/sequences expressed as token ids and passes them through the GRU.
            The output is the predicted (un-normalized) probabilities of
            the next token for all prefixes of the input sequences.

        Args:
            x (torch.tensor): A 2d tensor of longs giving the token ids for each batch. Shape B * S
                where B is the batch size (any batch size >= 1 is permitted), S is the length of the sequence.
            h_last (torch.tensor, optional): A 3d float tensor of size 1 * B * G where B is the batch size and G
                is the dimensionality of the GRU hidden state. The hidden state from the previous step, provide only if
                generating sequences iteratively. Defaults to None.

        Returns:
            tuple(torch.tensor, torch.tensor): first element of the tuple is the B * S * V where V is the vocabulary size.
                This is the logit output of the RNNLM. The second element is the hidden state of the final step
                of the GRU it should be 1 * B * G dimensional.
        """
        # Embedding layer
        x_emb = self.emb(x)

        # Hidden state check
        if h_last is not None:
            if h_last.dim() == 2:
                h_last = h_last.unsqueeze(0)

        # GRU Layer + Hidden state update
        gru_out, h = self.gru(x_emb, h_last)

        h = h.squeeze(0)

        # Linear layer
        out = self.linear(gru_out)
        return out, h


def train_model(model, Xtrain, Ytrain, Xval, Yval, id_to_char, max_epoch):
    """Train the RNNLM model using the Xtrain and Ytrain examples.
    Uses batch stochastic gradient descent with the Adam optimizer on 
    the mean cross entropy loss. Prints out the validation loss
    after each epoch using calc_val_loss.

    Args:
        model (RNNLM): the RNNLM model.
        Xtrain (torch.tensor): The training data input sequence of size N * S.
            N is the total number of training instances.
            S is the sequence length.
            The sequences always start with the <bos> token id.
            The rest of the sequence is just Ytrain shifted to the right one position.
            The sequence is zero padded.
        Ytrain (torch.tensor): The expected output sequence of size N * S. Does not start with the <bos> token.
        Xval (torch.tensor): The validation data input sequence of size M * S.
            M is the number of validation instances.
            S is the sequence length.
            The sequences always start with the <bos> token id.
            The rest of sequence is just Yval shifted to the right one position.
            The sequence is zero padded.
        Yval (torch.tensor): The expected output sequence for the validation data of size M * S.
            Does not start with the <bos> token. Is zero padded.
        id_to_char (dict(int : str)): A mapping from ids to tokens.
        max_epoch (int): the maximum number of epochs to train for.
    """
    # construct the adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # construct the cross-entropy loss function
    # we want to ignore padding cells with value == 0
    lossfn = nn.CrossEntropyLoss(ignore_index=0)

    # calculate number of batches
    batch_size = 32
    num_batches = int(Xtrain.shape[0] / batch_size)

    # run the main training loop over many epochs
    for e in range(max_epoch):
        model.train()

        # Compute total loss for the epoch
        total_loss = 0.0

        # Run mini-batch SGD
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size

            xb = Xtrain[start:end]
            yb = Ytrain[start:end]

            out, _ = model(xb)

            out_flat = out.view(-1, out.shape[-1])  # (B*S, V)
            yb_flat = yb.view(-1)

            # Compute loss
            loss = lossfn(out_flat, yb_flat)

            # Zero gradients
            optim.zero_grad()

            # Apply backpropagation
            loss.backward()

            # Update parameters
            optim.step()

            # Gather loss
            total_loss += loss.item()

        avg_loss = total_loss / num_batches

        model.eval()

        with torch.no_grad():
            val_loss = calc_val_loss(model, Xval, Yval)

        print(f"Epoch {e + 1}/{max_epoch} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        


def gen_string(model, id_to_char, max_len=20, sample=True):
    """Generate a name using the model. The generation process should finish once
    the end token is seen. We either sample from the model, where the next token is
    chosen randomly according to the categorical probability distribution produced by softmax,
    or we use argmax decoding where the most likely token is chosen at every generation step.

    Args:
        model (RNNLM): The trained RNNLM model.
        id_to_char (dict(int, str)): A mapping from token ids to token strings.
        max_len (int, optional): The maximum length of the output senquence. Defaults to 20.
        sample (bool, optional): If True then generate samples. If False then use argmax decoding. 
            Defaults to True.

    Returns:
        str: The generated name as a string.
    """
    # put the model into eval mode because we don't need gradients
    model.eval()

    # setup the initial input to the network
    # we will use a batch size of one for generation
    h = torch.zeros((1,1,model.gru_size), dtype=torch.float)  # (1,1,G)
    x = torch.ones((1, 1), dtype=torch.long) # x is the <bos> token id which = 1
    out_str = ""

    for i in range(max_len):
        out, h = model(x, h)

        logits = out[0, -1, :]

        probs = torch.softmax(logits, dim=-1)

        if sample:
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = torch.argmax(probs).item()

        if next_id == 2:
            break

        out_str += id_to_char[next_id]

        x = torch.tensor([[next_id]], dtype=torch.long)

    # set the model back to training mode in case we need gradients later
    model.train()

    return out_str


def calc_val_loss(model, Xval, Yval):
    """Calculates the validation loss in average nats per character.

    Args:
        model (RNNLM): the RNNLM model.
        Xval (torch.tensor): The validation data input sequence of size B * S. 
            B is the batch size, S is the sequence length. The sequences always start with the <bos> token id.
            The rest of sequence is just Yval shifted to the right one position.
            The sequence is zero padded.
        Yval (torch.tensor): The expected output sequence for the validation data of size B * S. 
            Does not start with the <bos> token. Is zero padded.

    Returns:
        float: validation loss in average nats per character.
    """

    # use cross entropy loss
    lossfn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # put the model into eval mode because we don't need gradients
    model.eval()

    # calculate number of batches, we need to be precise this time
    batch_size = 32
    num_batches = int(Xval.shape[0] / batch_size)
    if Xval.shape[0] % batch_size != 0:
        num_batches += 1

    # sum up the total loss
    total_loss = 0
    total_chars = 0
    for n in range(num_batches):

        # calculate batch start end idxs 
        s = n * batch_size
        e = (n+1)*batch_size
        if e > Xval.shape[0]:
            e = Xval.shape[0]

        # compute output of model        
        out,_ = model(Xval[s:e])

        # compute loss and store
        loss = lossfn(out.permute(0, 2, 1), Yval[s:e]).detach().cpu().numpy()
        total_loss += loss

        char_count = torch.count_nonzero(Yval[s:e].flatten())
        total_chars += char_count.detach().cpu().numpy()

    # compute average loss per character
    total_loss /= total_chars
    
    # set the model back to training mode in case we need gradients later
    model.train()

    return total_loss


def main():

    # load the data from disk
    data = load_data(os.path.join("data", "names_small.json"))

    # get the letter 'vocabulary'
    vocab, id_to_char, char_to_id = get_vocab()
    vocab_size = len(vocab)

    # convert the data into a sequence of ids which will be the target for our RNN
    # o/p: [cat, dog] -> [[c, a, t, .], [d, o, g, .]]
    Y = seqs_to_ids(data, char_to_id)

    # the input needs to be shifted by 1 and have the <bos> tokenid prepended to it
    # this also means we have to remove the last element of the sequence to keep the length constant
    # o/p: [[c, a, t, .], [d, o, g, .]] -> [[<bos>, c, a, t], [<bos>, d, o, g]]
    X = np.concatenate([np.ones((Y.shape[0], 1)), Y[:, :-1]], axis=1)

    # split the data int training and validation
    # convert the data into torch tensors
    train_frac = 0.9
    num_train = int(X.shape[0]*train_frac)
    Xtrain = torch.tensor(X[:num_train], dtype=torch.long)
    Ytrain = torch.tensor(Y[:num_train], dtype=torch.long)
    Xval = torch.tensor(X[num_train:], dtype=torch.long)
    Yval = torch.tensor(Y[num_train:], dtype=torch.long)

    # train the model
    model = RNNLM(vocab_size)
    train_model(model, Xtrain, Ytrain, Xval, Yval, id_to_char, max_epoch=10)

    # use the model to generate and print some names
    print("Argmax: ", gen_string(model, id_to_char, sample=False))
    print("Random:")
    for i in range(10):
        gstr = gen_string(model, id_to_char)
        print(gstr)

if __name__ == "__main__":
    vocab, id_to_char, char_to_id = get_vocab()
    main()

