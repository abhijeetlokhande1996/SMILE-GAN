from discriminator import Discriminator
import torch
import torch.nn as nn
from utils import read_csv, smile_to_one_hot_3D
from generator import Generator
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def G_train(G, D, bs, z_dim, criterion, G_optimizer):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def D_train(D, G, x, bs, z_dim, real_ip_dim, D_optimizer, criterion):

    # real_ip_dim = np.prod(real_ip_dim[1:])
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    # x_real, y_real = x.view(-1, real_ip_dim), torch.ones(bs, 1)
    x_real, y_real = x, torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def main():

    file_path = "./250k_rndm_zinc_drugs_clean_3.csv"
    smiles_arr = read_csv(file_path, 1000, ["smiles"])
    smiles_arr = smiles_arr["smiles"].tolist()
    smiles_arr = [smile_string.strip() for smile_string in smiles_arr]
    # shuffle done
    # dont't shuffle further
    # np.random.shuffle(smiles_arr)
    data_one_hot, MAX_LENGTH, NCHARS, idx_to_char_dict = smile_to_one_hot_3D(
        smiles_arr)
    # print(data_one_hot[0])
    # print(idx_to_char_dict)
    batch_size = 128
    hidden_dim = 200
    LATENT_DIM = 196
    train_loader = torch.utils.data.DataLoader(
        dataset=data_one_hot, batch_size=batch_size, shuffle=False, drop_last=True)
    G = Generator(latent_dim=LATENT_DIM,
                  hidden_dim=hidden_dim, max_len=MAX_LENGTH)
    D = Discriminator(max_len=MAX_LENGTH, nchars=NCHARS)

    # optimizer
    lr = 0.001
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    z = Variable(torch.randn(batch_size, LATENT_DIM).to(device))

    criterion = nn.BCELoss()
    n_epoch = 15
    for epoch in range(1, n_epoch+1):
        D_losses, G_losses = [], []
        for batch_idx, x_ip in enumerate(tqdm(train_loader)):
            D_losses.append(D_train(D, G, x_ip, batch_size, LATENT_DIM,
                                    data_one_hot.shape, D_optimizer, criterion))

            G_losses.append(G_train(G, D, batch_size,
                            LATENT_DIM, criterion, G_optimizer))

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch), n_epoch,
              torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    test_results = []
    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, LATENT_DIM).to(device))
        x_hat = G(test_z.type(torch.float32))
        test_results.append(x_hat)
    # print(test_results)
    strings = []
    for arr in test_results[0]:
        s_str = ''
        for str_1 in arr:
            idx11 = str_1[:, None].argmax(axis=0)[0]
            char = idx_to_char_dict[idx11.item()]
            if char and not char.isspace():
                s_str += char
    strings.append(s_str)
    print("len(strings): ", len(strings))
    print(strings)


if __name__ == '__main__':
    main()
