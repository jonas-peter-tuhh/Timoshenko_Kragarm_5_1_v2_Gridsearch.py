"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
##
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import torch.autograd
from scipy.optimize import curve_fit

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
        inputs = x
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output.reshape(-1, 1)
##
choice_load = input("Möchtest du ein State_Dict laden? (y/n):")
if choice_load == 'y':
    train=False
    filename = input("Welches State_Dict möchtest du laden?")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load('C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1_v2\\saved_data\\'+filename))
    net.eval()
##
# Hyperparameter
learning_rate = 0.01

# Definition der Parameter des statischen Ersatzsystems

Lb = float(input('Länge des Kragarms [m]: '))
E = 21#float(input('E-Modul des Balkens [10^6 kNcm²]: '))
h = 10#float(input('Querschnittshöhe des Balkens [cm]: '))
b = 10#float(input('Querschnittsbreite des Balkens [cm]: '))
A = h*b
I = (b*h**3)/12
EI = E*I*10**-3
G = 80#float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1#int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
normfactor = 10/(Lb**3/(K*A*G)+(11*Lb**5)/(120*EI))

for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    Ln[i] = 0#float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = Lb#float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    s[i] = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')


def h(x, j):
    return eval(s[j])


#Netzwerk System 1
def f(x, net):
    net_out = net(x)
    phi = net_out[0::3]
    phi_x = torch.autograd.grad(phi, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    phi_xx = torch.autograd.grad(phi_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    phi_xxx = torch.autograd.grad(phi_xx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    ode = 0
    for i in range(LFS):
        ode += phi_xxx + h(x - Ln[i], i)/EI * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode


def g(x, net):
    net_out = net(x)
    ode = 0
    gamma = net_out[1::3]
    gamma_x = torch.autograd.grad(gamma, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(gamma))[0]
    for i in range(LFS):
        ode += gamma_x - h(x - Ln[i], i)/(K*A*G) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode


def t(x,net):
    net_out = net(x)
    ode = 0
    phi = net_out[0::3]
    gamma = net_out[1::3]
    v = net_out[2::3]
    v_x = torch.autograd.grad(v, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v))[0]
    ode += phi+gamma-v_x
    return ode


x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)
#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten
def gridSearch(num_layers, layers_size):
    start = time.time()
    net = Net(num_layers, layers_size)
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        # + net_S(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
        y1 = net(torch.unsqueeze(Variable(torch.from_numpy(
            x).float(), requires_grad=False).to(device), 1))
        fig = plt.figure()
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-20, 0])
        # ax2.set_
        net_out_plot = y1.cpu().detach().numpy()
        line1, = ax1.plot(x, net_out_plot[2::3])
        plt.show(block=False)
        f_anal=(-1/120 * normfactor * pt_x**5 + 1/6 * Q0[-1] * pt_x**3 - M0[-1]/2 *pt_x**2)/EI + (1/6 * normfactor * (pt_x)**3 - Q0[-1]*pt_x)/(K*A*G)

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(
            x_bc).float(), requires_grad=True).to(device), 1)
        # unsqueeze wegen Kompatibilität
        pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(),
                           requires_grad=False).to(device)

        x_collocation = np.random.uniform(
            low=0.0, high=Lb, size=(250 * int(Lb), 1))
        #x_collocation = np.linspace(0, Lb, 1000*int(Lb))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
        ode_phi = f(pt_x_collocation, net)
        ode_gamma = g(pt_x_collocation, net)
        ode_v = t(pt_x_collocation, net)

        # Randbedingungen
        net_bc_out = net(pt_x_bc)

        # Netzwerkausgabewerte berechnen
        phi = net_bc_out[0::3]
        gamma = net_bc_out[1::3]
        v = net_bc_out[2::3]

        # für phi:
        phi_x = torch.autograd.grad(phi, pt_x_bc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(phi))[0]
        phi_xx = torch.autograd.grad(phi_x, pt_x_bc, create_graph=True, retain_graph=True,
                                     grad_outputs=torch.ones_like(phi))[0]
        phi_bc1 = phi_x[0] + M0[-1] / EI
        phi_bc2 = phi[0]
        phi_bc3 = phi_xx[0] - Q0[-1] / EI
        phi_bc4 = phi_xx[-1]
        phi_bc5 = phi_x[-1]

        # für gamma:
        gamma_bc1 = gamma[0] + (Q0[-1]) / (K * A * G)
        gamma_bc2 = gamma[-1]

        # für v:
        v_bc1 = v[0]

        # Alle e's werden gegen 0-Vektor (pt_zero) optimiert.
        # BC Error
        mse_bc_phi = 1 / normfactor * mse_cost_function(phi_bc1, pt_zero) + mse_cost_function(phi_bc2, pt_zero) + (
                    Lb / 5) ** 5 * mse_cost_function(phi_bc3, pt_zero) + mse_cost_function(phi_bc4,
                                                                                           pt_zero) + mse_cost_function(
            phi_bc5, pt_zero)
        mse_bc_gamma = 1 / normfactor * mse_cost_function(gamma_bc1, pt_zero) + mse_cost_function(gamma_bc2, pt_zero)
        mse_bc_v = mse_cost_function(v_bc1, pt_zero)
        # ODE Error
        mse_ode_phi = 1 / normfactor * mse_cost_function(ode_phi, pt_all_zeros)
        mse_ode_gamma = 1 / normfactor * mse_cost_function(ode_gamma, pt_all_zeros)
        mse_ode_v = mse_cost_function(ode_v, pt_all_zeros)

        loss = mse_bc_phi + mse_ode_phi + mse_bc_gamma + mse_ode_gamma + mse_bc_v + mse_ode_v
        loss = 1 / normfactor * loss
        loss.backward()
        scheduler.step(loss)
        optimizer.step()
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out_v = net(pt_x)[2::3]
                #net_out_v = net_out.cpu().detach().numpy()
                err = torch.norm(net_out_v - f_anal, 2)
                print(f'Error = {err}')
                if time.time()-start > 200:
                    return 'kacke'
                if err < 0.1 * Lb:
                    print(
                        f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    end = time.time()
                    return end-start
                line1.set_ydata(net_out_v.cpu().detach().numpy())
                fig.canvas.draw()
                fig.canvas.flush_events()
    ##


# GridSearch
time_elapsed = []
for num_layers in range(2, 5):
    for _ in range(20):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [np.random.randint(8, 15) for _ in range(num_layers)]
        time_elapsed.append(
            (num_layers, layers_size, gridSearch(num_layers, layers_size)))
        plt.close()

with open(r'timing2.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')

##
if choice_load == 'n':
    choice_save = input("Möchtest du die Netzwerkparameter abspeichern? y/n")
    if choice_save == 'y':
        filename = input("Wie soll das State_Dict heißen?")
        torch.save(net.state_dict(),'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1_v2\\saved_data\\'+filename)
