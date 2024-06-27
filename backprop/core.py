
class Value(object):
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0

        
        # set for efficiency?
        # _underscore because internal
        # children are the things combined to make this thing
        self._prev = set(_prev)
        self._op = _op
        # set as dumby function for starters
        # when a node is combined to an output, the _output_ 
        # is told how to accumulate to that previous node's gradient
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out


    def __pow__(self, exp):
        out = Value(self.data**exp, (self,), f'**{exp}')
        
        def _backward():
            self.grad += out.grad * exp * self.data**(exp-1)
        out._backward = _backward
        
        return out

    

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        
        return out

    # public function, not to be confused with a given node's
    # _backward
    # topological sort of all predecessors: a node's backward
    # is never called before all it's later nodes are called (later in the forward comp graph)
    def backward(self):
        topo = []
        visited = set()
        # this will go as deep as possible to the start of the comp graph
        # then start adding in a forward-like pass. Something is only added when it has nothing before it unadded :)
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        
        # we want to proceed backwards in the comp graph
        self.grad = 1.0 # don't forget this! grad wrt to self
        for n in reversed(topo):
            n._backward()
