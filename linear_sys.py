try:
    from sympy import *
except ImportError:
    from sympy import *
import sys    
class rangeSpace :
    def __init__(self, A, y) :
        self.mat_A = A
        self.Aug_A = A.row_join(y)
        self.vect_y = y
        self.nullSpace_A = self.mat_A.nullspace()
        self.nullity_A = len(self.nullSpace_A)
        self.rangeSpaceBasis = self.mat_A.columnspace()
        self.rank_A = len(self.mat_A.columnspace())
        self.vect_xp = [ ]
        self.allSolution = [ ]
        self.freevarIndex = [ ]
        return
    def solutionSpace(self) :
        self.allSolution, params, self.freevarIndex = \
            self.mat_A.gauss_jordan_solve(self.vect_y, freevar=True)
        params_zeroes = { tau:0 for tau in params }
        self.vect_xp = self.allSolution.xreplace(params_zeroes)
        return self.vect_xp, self.nullity_A, self.nullSpace_A, \
            self.rank_A, self.rangeSpaceBasis 
    def printGeneralSolution(self) :
        self.parameter = symbols('alpha0:%d'%(self.nullity_A))
        print('Ax=y  (The solution x is in transpose form)' )
        x, T = symbols("x, T")
        pprint(x**T)
        print('   = ', end="")
        pprint(self.vect_xp.T)
        for i in range(self.nullity_A):
            print('   +', end=" ")
            Buff = factor(self.parameter[i]*self.nullSpace_A[i].T)
            g = gcd(tuple(Buff))
            h = MatMul(g,(Buff/g),evaluate = False)
            pprint(h)
        sys.stdout.flush()
        return
class similarTrnsfrm :
    def __init__(self, A) :
        self.mat_A = A
        self.eigenVectors = []
        self.EigenSpace = {}
        return
    def eigenSpace(self) :
        self.EigenSpace = {}
        self.eigenDict = self.mat_A
        if(self.mat_A.is_diagonalizable()) :
            eigenVectors = self.mat_A.eigenvects() 
            for i in range (len(eigenVectors)) :
                self.EigenSpace[(i+1, eigenVectors[i][0],eigenVectors[i][1])]= []
                self.EigenSpace[(i+1, eigenVectors[i][0],eigenVectors[i][1])]\
                    .append(eigenVectors[i][2])
            Q, DJ =  self.mat_A.diagonalize()
        else:
            Q, blocks = self.mat_A.jordan_cells() 
            basis = [Q[:,i] for i in range(Q.shape[1])] #Q dimension (4,4)
            n = 0
            self.EigenSpace = {}
            index = 0
            for r in blocks:
                index+=1
                eigval = r[0, 0]
                size = r.shape[0]
                self.EigenSpace[(index, eigval, size)] = []    
                self.EigenSpace[(index, eigval, size)].append(basis[n:n+size])
                n += size
            Q, DJ = self.mat_A.jordan_form()
        return self.EigenSpace, Q, DJ
    def printEigenSpace(self) :
        Keys = list(self.EigenSpace.keys())
        Values = list(self.EigenSpace.values())
        T = symbols("T")
        Vector = symbols('q1:%d'%(len(Keys)+1))
        print('Q = ', end = '')
        pprint(Matrix(Vector).T)
        for i in range(len(Keys)):
                print(Keys[i])
                pprint(Vector[i]**T)
                pprint(Matrix(Values[i]).T) 
        return
