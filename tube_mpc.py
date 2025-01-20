from utils import FiniteHorizon, LTI
from minkowski import plot_polytope, minkowski_sum, pontryagin_diff, matrix_polytope_mult
import numpy as np
import numpy.typing as nptyping
import polytope as pc
import matplotlib.pyplot as plt
import scipy as sp

# %% System properties
# System matrices
A = np.array([[1.5, 1], [0, 0.5]])
B = np.array([ [1], [0]])

x0 = np.array([2.1, -11]).reshape(2,1)

# System Constraints
ulim = 11
Ax = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
bx = np.array([[10,14,14,10]]).T
Au = np.array([[1],[-1]])
bu = np.array([[ulim, ulim]]).T

# (Symmetric) Noise Bounds
wbounds = np.array([[0.15, 0.3]]).T

# MPC Params
N = 50


# %% Tracking controller Weights - Also used for the MPC
R = np.array([[20]])
Q = np.array([[5,0],[0,1]])
P = sp.linalg.solve_discrete_are(A,B,Q,R)
L = np.linalg.inv(R+B.T@P@B)@(B.T@P@A)

# %% Calculating mRPI and tightened constaints
X = pc.Polytope(Ax, bx)
U = pc.Polytope(Au, bu)

W = pc.box2poly(np.concatenate([-wbounds,wbounds],1)) # Noise bound set

def mRPI_iterative(AL:nptyping.NDArray, W:pc.Polytope, max_iter=1000):
    F = W.copy()
    for i in range(max_iter):
        # ax = F.plot()
        # ax.autoscale()
        # plt.show()
        # Terminate if A**k matrix is close to zero => Little new information
        if np.allclose(np.linalg.matrix_power(AL,i),0,1E-9,1E-9):
            break
        Wnew = matrix_polytope_mult(W,np.linalg.matrix_power(AL,i), output="ConvexHull")
        Fnew = minkowski_sum(F,Wnew)
        # Terminate if change in volume is small => algo is converging
        if np.abs(Fnew.volume - F.volume) < 1E-9:
            F = Fnew
            break
        F = Fnew

    return F

E = mRPI_iterative((A-B@L),W)
X_bar = pontryagin_diff(X, E)
U_bar = pontryagin_diff(U, matrix_polytope_mult(E,-L))

setfig, setaxes = plt.subplots(2,2)
plot_polytope(W,setaxes[0,0],color='yellow')
setaxes[0,0].text(0,0,r"$\mathbb{W}$",size=32,weight='black',c='k')
plot_polytope(E,setaxes[0,1], color='gold')
setaxes[0,1].text(0,0,r"$\mathbb{E}$",size=32,weight='black',c='k')
plot_polytope(X,setaxes[1,0], color='navy')
setaxes[1,0].text(*pc.extreme(X)[1],r"$\mathbb{X}$",size=32,weight='black',c='k')
plot_polytope(X_bar,setaxes[1,0],color='blueviolet')
setaxes[1,0].text(0,0,r"$\mathbb{\bar{X}}$",size=32,weight='black',c='k')
setaxes[1,1].plot(pc.extreme(U),[0,0],linewidth=4)
setaxes[1,1].text(pc.extreme(U)[0],0,r"$\mathbb{U}$",size=32,weight='black',c='k')
setaxes[1,1].plot(pc.extreme(U_bar),[0,0],linewidth=3)
setaxes[1,1].text(0,0,r"$\mathbb{\bar{U}}$",size=32,weight='black',c='k')

# %% Running simulations
wbounds = np.concatenate([-wbounds,wbounds],1)

# Create a function for I.I.T additive noise on the states
wfunc = lambda :np.array([[np.random.uniform(*wb) for wb in wbounds]]).reshape((-1,1))

x = x0 + wfunc()
x_bar = x
x_bars = []
Es = [pc.Polytope(E.A, E.b + np.squeeze(E.A@x_bar))]

MPC = FiniteHorizon(A,B,Q,R,P,N,X_bar.A,X_bar.b.reshape((-1,1)),U_bar.A,U_bar.b.reshape((-1,1)), E.A, E.b.reshape((-1,1)))
sys = LTI(A,B,wfunc,x,ulim)


for t in range(20):
    Uopt, Xopt = MPC.solve(x)
    x_bar = Xopt[:,[0]]
    Es.append(pc.Polytope(E.A, E.b + np.squeeze(E.A@x_bar)))
    x_bars.append(x_bar)
    u = Uopt[:,[0]] - L@(x-x_bar)
    x = sys.step(u)


x_bars = np.concatenate(x_bars, 1)
print(np.concatenate([sys.X[:,:-1].T,x_bars.T],1))
fig, ax = plt.subplots()
plot_polytope(X,ax, color='navy')
plot_polytope(X_bar,ax, color='blueviolet')
for p in Es:
    plot_polytope(p, ax, alpha=0.5)
ax.plot(*sys.X, label=r"$x$")
ax.scatter(*sys.X)

ax.plot(*x_bars, label=r"$\bar{x}$", linestyle='--')
ax.scatter(*x_bars, marker='o')
ax.legend()
plt.show()
