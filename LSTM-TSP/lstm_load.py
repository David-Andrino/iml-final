from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

layer_size = 51
epochs = 15
noise_amplitude = 1 

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, layer_size)
        self.lstm2 = nn.LSTMCell(layer_size, layer_size)
        self.linear = nn.Linear(layer_size, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), layer_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), layer_size, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), layer_size, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), layer_size, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class Pred:
    def predict(x, future, target):
        seq = Sequence()
        seq.double()
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
        # load data
        seq.load_state_dict(torch.load('traindata_15epochs_1noise.pt'))
        test_input = test_input = torch.from_numpy(x[:1, :-1])
        test_target = torch.from_numpy(target[:1, 1:])
        with torch.no_grad():
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
            return y


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--steps', type=int, default=5, help='steps to run')
    # opt = parser.parse_args()
    # set random seed to 0
    # np.random.seed(0)
    # torch.manual_seed(0)
    # load data and make training set
    # np.random.seed(2)

    T = 20
    L = 1000
    N = 100
    

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) #+ np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    # ideal = np.sin(x / 1.0 / T).astype('float64')
    # triangular_signal = noise_amplitude * np.abs(2 * (x / T - np.floor(0.5 + x / T)))
    data = np.sin(x / 1.0 / T).astype('float64') + np.random.uniform(-0.1*noise_amplitude/2,0.1*noise_amplitude/2)
    # print("New data generated")

    # data = torch.load('traindata.pt')
    input = torch.from_numpy(data[1:, :-1])
    target = torch.from_numpy(data[1:, 1:])
    test_input = torch.from_numpy(data[:1, :-1])
    test_target = torch.from_numpy(data[:1, 1:])
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # load data
    seq.load_state_dict(torch.load('traindata_15epochs_1noise.pt'))

    with torch.no_grad():
        future = 1000
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
    # draw the result
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    draw(y[0], 'r')
    
    # draw(y[1], 'g')
    # draw(y[2], 'b')
    plt.show()
    # plt.savefig('predict%d_noise%f.pdf'%(epochs,noise_amplitude))
    plt.close()

    # torch.save(seq.state_dict(), open(f"traindata_{epochs}epochs_{noise_amplitude}noise.pt", 'wb'))