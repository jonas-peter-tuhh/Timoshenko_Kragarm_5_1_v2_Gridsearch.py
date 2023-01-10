"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
##
import sys
import os
import time
#insert path of parent folder to import helpers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = True

class Net(nn.Module):
    def __init__(self, num_layers, layers_size):
        super(Net, self).__init__()
        assert num_layers == len(layers_size)
        self.linears = nn.ModuleList([nn.Linear(1, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i])
                            for i in range(1, num_layers)])
        self.linears.append(nn.Linear(layers_size[-1], 3))

    def forward(self, x):  # ,p,px):
        # torch.cat([x,p,px],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        x = torch.unsqueeze(x, 1)
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output.reshape(-1, 1)
##
# Hyperparameter

learning_rate = 0.01
mse_cost_function = torch.nn.MSELoss()  # Mean squared error


# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21
K = 5/6
G = 80
A = 100

#Normierungsfaktor (siehe Kapitel 10.3)
normfactor = 10/((11*Lb**5)/(120*EI))

# ODE als Loss-Funktion, Streckenlast
Ln = 0 #float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
Lq = Lb # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
s = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x):
    return eval(s)


#Netzwerk System 1
def f(x, net):
    net_out = net(x)
    phi = net_out[0::3]
    _, _, phi_xxx = deriv(phi, x, 3)
    ode = phi_xxx + (h(x - Ln))/EI
    return ode


def g(x, net):
    net_out = net(x)
    gamma = net_out[1::3]
    gamma_x = deriv(gamma, x, 1)[0]
    ode = gamma_x - (h(x - Ln))/(K*A*G)
    return ode


def t(x,net):
    net_out = net(x)
    ode = 0
    phi = net_out[0::3]
    gamma = net_out[1::3]
    v = net_out[2::3]
    v_x = deriv(v, x, 1)[0]
    ode += phi + gamma - v_x
    return ode


x = np.linspace(0, Lb, 1000)
qx = h(x) * (x <= (Ln + Lq)) * (x >= Ln)

Q0 = integrate.cumtrapz(qx, x, initial=0)

qxx = qx * x

M0 = integrate.cumtrapz(qxx, x, initial=0)
def gridSearch(num_layers, layers_size):
    start = time.time()
    net = Net(num_layers, layers_size)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        mse_cost_function = torch.nn.MSELoss()  # Mean squared error
        y1 = net(myconverter(x, False))
        fig = plt.figure()
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-10, 0])
        net_out_plot = myconverter(y1)
        line1, = ax1.plot(x, net_out_plot[2::3])
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.show(block=False)
        pt_x = myconverter(x)
        f_anal = (-1 / 120 * normfactor * x ** 5 + 1 / 6 * Q0[-1] * x ** 3 - M0[-1] / 2 * x ** 2) / EI + (
                1 / 6 * normfactor * x ** 3 - Q0[-1] * x) / (K * A * G)

    iterations = 1000000
    for epoch in range(iterations):
        if not train: break
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(myconverter(x_bc), 1)

        x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = torch.unsqueeze(myconverter(x_collocation), 1)
        f_out_phi = f(pt_x_collocation, net)
        f_out_gamma = g(pt_x_collocation, net)
        f_out_v = t(pt_x_collocation, net)

        # Randbedingungen
        net_bc_out = net(pt_x_bc)

        # Netzwerkausgabewerte berechnen
        phi = net_bc_out[0::3]
        gamma = net_bc_out[1::3]
        v = net_bc_out[2::3]

        # für phi:
        phi_x, phi_xx = deriv(phi, pt_x_bc, 2)

        BC6 = phi_xx[0] - Q0[-1] / EI
        BC7 = phi_xx[-1]
        BC8 = phi_x[0] + M0[-1] / EI
        BC9 = phi_x[-1]
        BC10 = phi[0]

        # für gamma:
        BC4 = gamma[0] + (Q0[-1]) / (K * A * G)
        BC5 = gamma[-1]

        # für v:
        BC1 = v[0]

        mse_Gamma_phi = errsum(mse_cost_function, 1 / normfactor * BC6, BC7, 1 / normfactor * BC8, BC9, BC10)
        mse_Gamma_gamma = errsum(mse_cost_function, 1 / normfactor * BC4, BC5)
        mse_Gamma_v = errsum(mse_cost_function, BC1)
        mse_Omega_phi = errsum(mse_cost_function, f_out_phi)
        mse_Omega_v = errsum(mse_cost_function, f_out_v)
        mse_Omega_gamma = errsum(mse_cost_function, f_out_gamma)

        loss = mse_Gamma_phi + mse_Gamma_v + mse_Gamma_gamma + mse_Omega_v + mse_Omega_phi + 1 / normfactor * mse_Omega_gamma
        loss.backward()

        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out = myconverter(net(pt_x))
                net_out_v = net_out[2::3]
                err = np.linalg.norm(net_out_v - f_anal, 2)
                print(f'Error = {err}')
                if err < 0.1 * Lb:
                    print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    break
                line1.set_ydata(net_out_v)
                fig.canvas.draw()
                fig.canvas.flush_events()
    ##

# GridSearch
time_elapsed = []
for num_layers in range(4, 5):
    for _ in range(10):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [np.random.randint(30, 300) for _ in range(num_layers)]
        time_elapsed.append(
            (num_layers, layers_size, gridSearch(num_layers, layers_size)))
        plt.close()

with open(r'random14m4s.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')

##
if choice_load == 'n':
    choice_save = input("Möchtest du die Netzwerkparameter abspeichern? y/n")
    if choice_save == 'y':
        filename = input("Wie soll das State_Dict heißen?")
        torch.save(net.state_dict(),'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1_v2\\saved_data\\'+filename)
