import numpy as np
import polytope as pc
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy.typing as nptyping
from scipy.spatial import ConvexHull

def plot_polytope(p:pc.Polytope, ax=None,color=None, alpha=1):
    ax = p.plot(ax=ax,color=color,alpha=alpha)
    ax.axvline(0,color='k')
    ax.axhline(0,color='k')
    ax.set_xmargin(0.05)
    ax.set_ymargin(0.05)
    ax.autoscale()
    ax.grid()
    # ax.set_yticks(np.arange(np.floor(np.min(ax.get_ylim())), np.ceil(np.max(ax.get_ylim())), 1))
    # ax.set_xticks(np.arange(np.floor(np.min(ax.get_xlim())), np.ceil(np.max(ax.get_xlim())), 1))
    return ax

def pontryagin_diff(p1:pc.Polytope, p2:pc.Polytope):
    Aret = np.asarray(p1.A, copy=True)
    bret = np.asarray(p1.b, copy=True)
    dim = Aret.shape[1]

    v = cp.Parameter((1,dim))
    X = cp.Variable((dim,1))
    criterion = cp.Maximize(v@X)
    prob = cp.Problem(criterion, [p2.A@X <= p2.b.reshape((-1,1))])

    for i,r in enumerate(Aret):
        v.value = r.reshape((1,-1))
        prob.solve()
        bret[i] = bret[i] - np.dot(r,X.value).item()

    return pc.Polytope(Aret, bret)

def matrix_polytope_mult(p:pc.Polytope, M:nptyping.NDArray, output="Polytope"):
    extremes = pc.extreme(p)
    ## Special case if M has output dimension 1
    if M.shape[0] == 1:
        Me = (M@extremes.T).T
        A = np.array([[-1,1]]).T
        b = np.array([[-np.min(Me), np.max(Me)]]).T
        return pc.Polytope(A,b)
    hull = ConvexHull((M@extremes.T).T)
    A,b = hull.equations[:,:-1], hull.equations[:,-1]
    if output == "Polytope":
        return pc.Polytope(A,-b)
    if output == "ConvexHull":
        return hull
    else:
        raise ValueError("Invalid output option.")

def minkowski_sum(p1:pc.Polytope, p2:pc.Polytope):
    """Just a convex hull approximation."""
    if type(p1) == pc.Polytope:
        p1vertices = pc.extreme(p1)
    elif type(p1) == ConvexHull:
        p1vertices = p1.points[p1.vertices]
    else:
        raise ValueError("p1 is of invalid type.")
    
    if type(p2) == pc.Polytope:
        p2vertices = pc.extreme(p2)
    elif type(p2) == ConvexHull:
        p2vertices = p2.points[p2.vertices]
    else:
        raise ValueError("p2 is of invalid type.")
    
    p2s = [p2vertices+v for v in p1vertices]
    hull = ConvexHull(np.concat(p2s,0))
    A,b = hull.equations[:,:-1], hull.equations[:,-1]
    ret = pc.Polytope(A,-b)
    return ret

if __name__ == '__main__':
        
    Ax = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    bx = np.array([10,7,9,8])
    X = pc.Polytope(Ax,bx)
    Aw= np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    bw = np.array([3,4,1,2])
    E = pc.Polytope(Aw,bw)

    XmW = pontryagin_diff(X,E)
    XpW = minkowski_sum(X, E)

    fig, axes = plt.subplots(2,2)
    plot_polytope(X,axes[0,0])
    axes[0,0].set_title(r"$\mathbb{X}$")
    plot_polytope(E,axes[0,1])
    axes[0,1].set_title(r"$\mathbb{E}$")
    plot_polytope(XmW,axes[1,0])
    axes[1,0].set_title(r"$\mathbb{X} \ominus \mathbb{E}$")
    plot_polytope(XpW,axes[1,1])
    axes[1,1].set_title(r"$\mathbb{X} \oplus \mathbb{E}$")


    plt.show()