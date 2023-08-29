try:
    from sympy import *
except ImportError:
    from sympy import *
class rangeSpace :
    def __init__(self, A, y) :
        self.mat_A = A
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
        try :
            self.allSolution, params, self.freevarIndex = \
                self.mat_A.gauss_jordan_solve(self.vect_y, freevar=True)
        except :
            self.allSolution = 'There is no solution'
            return 0, 0, 0, 0, 0
        params_zeroes = { tau:0 for tau in params }
        self.vect_xp = self.allSolution.xreplace(params_zeroes)
        return self.vect_xp, self.nullity_A, self.nullSpace_A, \
            self.rank_A, self.rangeSpaceBasis 
    def printGeneralSolution(self) :
        from IPython.display import display, Latex
        import re
        x = MatrixSymbol('x', shape(self.mat_A)[1],1)
        MatEqn = "$${}\\mathbf{{x}} = {}$$".format(latex(self.mat_A),latex(self.vect_y))
        display(Latex(MatEqn))
        if(self.allSolution == 'There is no solution') :
            print('\nThe system of linear equations has no solution.\n')
            return 0;
        self.parameter = symbols('alpha0:%d'%(self.nullity_A))
        E = []
        Xp = self.vect_xp
        for i in reversed(range(self.nullity_A)):
            Buff = factor(self.parameter[i]*self.nullSpace_A[i])
            g = gcd(tuple(Buff))
            h = MatMul(g,(Buff/g),evaluate = False)
            if i == (self.nullity_A) - 1 :
                E = h 
            else :
                E =MatAdd(E,h, evaluate = False)
        if self.nullity_A :
            if Xp.norm() :
                X = "$$\\mathbf{{x}}={} + {}$$".format(latex(Xp),latex(E))
            else :
                X = "$$\\mathbf{{x}}={}$$".format(latex(E))
        else :
            X = "$$\\mathbf{{x}}={}$$".format(latex(Xp))
        X= re.sub(r'\\left\(|\\right\)', '', X)
        display(Latex(X)) 
        return E
    
class similarTrnsfrm :
    def __init__(self, A) :
        self.mat_A = A
        self.eigenVectors = []
        self.EigenSpace = {}
        return
    def eigenSpace(self) :
        self.EigenSpace = {}
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
        from IPython.display import display, Latex
        import re
        MatEqn = "$$\\mathbf{{A}} = {}$$".format(latex(self.mat_A))
        display(Latex(MatEqn))
        Keys = list(self.EigenSpace.keys())
        Values = list(self.EigenSpace.values())
        Q, T= symbols("Q T")
        Vector = symbols('q1:%d'%(len(Keys)+1))
        Q = "$$\\mathbf{{Q}}={}$$".format(latex(Matrix(Vector).T))
        Q = re.sub(r'q', r'\\mathbf{q}', Q)
        display(Latex(Q))
        lamda = symbols('lamda1:%d'%(len(Keys)+1))
        for i in range(len(Keys)):
                ID = "$${}={}, mutiplicity ={}$$".format(latex(lamda[i]), \
                        latex(Keys[i][1]), latex(Keys[i][2]))
                display(Latex(ID))
                V = latex(Vector[i]**T)
                V = re.sub(r'q', r'\\mathbf{q}', V)
                V = "$${}={}$$".format(V, latex(Matrix(Values[i]).T)) 
                display(Latex(V)) 
        return
