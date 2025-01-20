import cvxpy as cp
import numpy as np
import polytope as pc


class FiniteHorizon:
    def __init__(self, A, B, Q, R, P, N, Ax=None, bx=None, Au=None, bu=None, Ae=None, be=None ,c=None):
        self.N = N
        n, m = B.shape

        # Setting up the optimization variables
        self.U = cp.Variable((m, N))  # [ u_0, ..., u_{N-1} ]
        self.X = cp.Variable((n, N + 1))  # [ x_0, ..., x_N ]
        self.x0 = cp.Parameter(
            (n,1)
        )  # Placeholder for the initial state; it is set when calling solve.

        # Setting up the finite-horizon criterion
        criterion = cp.quad_form(self.X[:, N], P)
        for k in range(N):
            criterion += cp.quad_form(self.X[:, k], Q) + cp.quad_form(self.U[:, k], R)

        # Setting up constraints
        constraints = []
        constraints.append(
            self.X[:, 1 : N + 1] == A @ self.X[:, 0:N] + B @ self.U
        )  # x_{k+1} = A x_k + B u_k, k=0, ..., N-1

        # If polytopic state constraints are provided, add them to the constraints list
        if Ax is not None and bx is not None:
            constraints.append(Ax @ self.X <= bx)  # Ax @ x <= bx

        # If polytopic input constraints are provided, add them to the constraints list
        if Au is not None and bu is not None:
            constraints.append(Au @ self.U <= bu)

        # If polytopic tube mRPI constraints are provided, add them to the constraints list
        if Ae is not None and be is not None:
            constraints.append(Ae @ self.x0 <= be + Ae@self.X[:,[0]])
        else:
            constraints.append(self.X[:, 0] == self.x0)  # x0 == x(0)

        # If the quadratic terminal constraint is provided, add it to the constraints list
        if c is not None:
            constraints.append(cp.quad_form(self.X[:, [-1]], P) <= c)

        # Minimization problem
        self.prob = cp.Problem(cp.Minimize(criterion), constraints)

    def solve(self, x0, verbose=False):
        self.x0.value = x0
        self.prob.solve(verbose=verbose, solver=cp.CLARABEL)

        if self.prob.status == cp.INFEASIBLE:
            raise ValueError("The problem is infeasible")

        return self.U.value, self.X.value


class LTI:
    def __init__(self, A, B, wfunc, x0, umax=np.inf):
        self.A = A
        self.B = B
        self.w = wfunc
        self.n, self.m = B.shape
        self.umax = umax
        x0 = x0.reshape(self.n, 1)
        self.X = x0
        self.U = np.empty((self.m, 0))

    def step(self, u):
        # Imposte consraints on the input
        u = u.reshape(self.m, 1)
        u[u > self.umax] = self.umax
        u[u < -self.umax] = -self.umax

        # Take one step
        x = self.A @ self.X[:, [-1]] + self.B @ u + self.w()
        self.X = np.append(self.X, x, axis=1)
        self.U = np.append(self.U, u, axis=1)
        return self.X[:, [-1]]
