from binary import bin_parity, to_bin, to_set


class Spin:
    def __init__(self, m, us, ds):
        # us, ds expected to be in binary representation
        self.m = m
        self.us = us
        self.ds = ds
    def __repr__(self):
        return "<%d, u: %s, d: %s>" % (self.m, to_set(self.us), to_set(self.ds))
    
    @property
    def sites(self):
        return self.us | self.ds

def get_new_spin_ind(_spins):
    keys = _spins.keys()
    for i in range(62):
        if i not in keys:
            return i
    raise RuntimeError("Too many spins to keep track of.")

class Kron_Part:
    def __init__(self, freq, sign, us, ds):
        # us, ds expected to be in binary representation
        # If sign is -1, then the component represents 
        # 1./sqrt(2) * (|up = us, down = ds> - |up = ds, down = us> )
        self.freq: float = freq
        self.sign: int   = sign
        self.us  : int   = us
        self.ds  : int   = ds
        
    def __repr__(self):
        return "<%.10e, %d, %s, %s>" % (self.freq, self.sign, to_set(self.us), to_set(self.ds))

class Term:
    def __init__(self, *args):
        # Ordering: X...X Y...Y
        # Represent frozenset by int.
        if len(args) == 2:
            self.Xs = args[0]
            self.Ys = args[1]
            self.val = 1.
        elif len(args) == 1:   # used to construct identity or zero term. val is int in this case.
            self.Xs = 0
            self.Ys = 0
            self.val = args[0]

    @property
    def key(self):
        return (self.Xs, self.Ys)
    
    def __repr__(self):
        # Convert frozenset to set for more compact output
        if abs(self.val) < 1e-7:
            return f"{self.val:+.4e}<Xs:{to_set(self.Xs)} Ys:{to_set(self.Ys)}>"
        return f"{self.val:+.8f}<Xs:{to_set(self.Xs)} Ys:{to_set(self.Ys)}>"
    def __eq__(self, other):
        return self.Xs == other.Xs and self.Ys == other.Ys
    def __abs__(self):
        return abs(self.val)
    def __mul__(self, c):
        ans = self.copy()
        ans.val *= c
        return ans
    __rmul__ = __mul__
    def __imul__(self, c):
        if self.val == 0:
            return self
        self.val *= c
        return self
    def __iadd__(self, other):
        if self.val == 0:
            return other.copy()
        if self != other:
            return NotImplementedError
        self.val += other.val
        return self
    def copy(self):
        # WARNING: fatal error may occur if the Term is modified after a hash value is assigned.
        ans = Term(self.Xs, self.Ys)
        ans.val = self.val
        return ans
    def is_overlap(self, sites):
        return (self.Xs | self.Ys) & sites != 0
    def is_supported(self, sites):
        return self.Xs | self.Ys | sites == sites

    def is_commute(self, other):
        return bin_parity(self.Xs & other.Ys) ^ bin_parity(self.Ys & other.Xs) == 0

    def unit(self):
        if self.val != 0 and self.val != 1:
            self.val = 1.
        return self
    def spread(self):
        return (self.Xs | self.Ys).bit_count()

class Operator:
    def __init__(self, terms):
        self.terms = {}
        if isinstance(terms, int):
            terms = [Term(terms)]
        for term in terms:
            if term is not None:
                self.terms[term.key] = term
        if len(self.terms) == 0:
            terms = [Term(0)]
    def append(self, term):
        k = term.key
        if k in self.terms:
            self.terms[k].val += term.val
            # As a result of this construction, append has to copy, for consistency
        else:
            self.terms[k] = term.copy()
    def append_nocopy(self, term):
        # Only use this out of performance concerns.
        k = term.key
        if k in self.terms:
            self.terms[k].val += term.val
        else:
            self.terms[k] = term

    def __iand__(self, other):
        for term in other.terms.values():
            self.append(term)
        return self
    def extend(self, terms):
        # for a list of terms
        for term in terms:
            self.append(term)
    def copy(self):
        # Performance may be improved
        ans = Operator([])
        for term in self.terms.values():
            ans.terms[term.key] = term.copy()
        return ans
    def __add__(self, other):
        if len(self) >= len(other):
            ans = self.copy()
            for term in other.terms.values():
                ans.append(term.copy())
            return ans
        else:
            ans = other.copy()
            for term in self.terms.values():
                ans.append(term.copy())
            return ans
    
    def truncate(self, max_spread = None, max_len = None, tol: float = 1e-9):
        # terms_copy = self.terms.copy()
        # if max_spread is None and max_len is None:
        #     for i in terms_copy:
        #         if abs(self.terms[i].val) < tol:
        #             self.terms.pop(i)
        ans = Operator([])
        if max_spread is None and max_len is None:
            for term in self.terms.values():
                if abs(term.val) > tol:
                    ans.terms[term.key] = term.copy()
        elif max_spread is not None:
            raise NotImplementedError
            for i in terms_copy:
                if self.terms[i].spread() > max_spread:
                    self.terms.pop(i)
                    continue
                if abs(self.terms[i].val) < tol:
                    self.terms.pop(i)
        else:
            raise NotImplementedError
            if len(self) <= MAX_LENH:
                self.truncate()
                return
            _vals = np.array([abs(term.val) for term in self.terms.values()])
            _pval = np.partition(_vals, len(self) - MAX_LENH)[-MAX_LENH]
            for i in terms_copy:
                if abs(self.terms[i].val) + tol < _pval:
                    self.terms.pop(i)
                    continue
                if abs(self.terms[i].val) < tol:
                    self.terms.pop(i)
        self.terms = ans.terms

    def __repr__(self):
        if len(self.terms) == 0:
            return "0"
        str = ""
        for term in self.terms.values():
            str += f"{term} "
        return str
    def __len__(self):
        return len(self.terms)
    def __mul__(self, x):
        
        if isinstance(x, Operator):
            return self._opmul(x)
        ans = self.copy()
        for term in ans.terms.values():
            term.val *= x
        return ans

    def _opmul(self, other):
        
        ans = Operator([])

        for a in self.terms.values():
            for b in other.terms.values():
                
                val = a.val * b.val
                term = Term(a.Xs ^b.Xs, a.Ys ^ b.Ys)
                term.val = val
                
                term.val *= 1. - 2. * bin_parity(a.Ys & b.Xs)
                ans.append_nocopy(term)
        
        return ans
    
    def __rmul__(self, x):
        ans = self.copy()
        for term in ans.terms.values():
            term.val *= x
        return ans
    def __imul__(self, c):
        if isinstance(c, Operator):
            return NotImplementedError
        for term in self.terms.values():
            term.val *= c
        return self
    def find_term(self, term):
        return self.terms.get(term.key)
    def find_val(self, x, tol: float = 1e-9):
        # returns an operator that contains all terms whose val equals x or -x.
        x = abs(x)
        ans = Operator([])
        for term in self.terms.values():
            if abs(abs(term.val) - x) < tol:
                ans.append(term)
        return ans
    def max_term(self, counts_classical = False, max_spread = None):
        norm = 0.
        ans = Term(0)
        if counts_classical == False:
            for term in self.terms.values():
                if abs(term.val) > norm and term.Xs | term.Ys != 0:
                    if (max_spread is not None) and (term.spread() > max_spread):
                        continue
                    norm = abs(term.val)
                    ans = term
        else:
            for term in self.terms.values():
                if abs(term.val) > norm:
                    if (max_spread is not None) and (term.spread() > max_spread):
                            continue
                    norm = abs(term.val)
                    ans = term
        return ans.copy()
    def collect_overlap_terms(self, sites):
        if isinstance(sites, int) == False:
            sites = to_bin(sites)
        ans = Operator([])
        for term in self.terms.values():
            if term.is_overlap(sites):
                ans.append(term)
        return ans
    def collect_supported_terms(self, sites):
        if isinstance(sites, int) == False:
            sites = to_bin(sites)
        ans = Operator([])
        for term in self.terms.values():
            if term.is_supported(sites):
                ans.append(term)
        return ans
    def get_magnitudes(self, max_spread_stat, tol: float = 1e-9):
        ans = [[] for _ in range(max_spread_stat)]
        for term in self.terms.values():
            _s = term.spread()
            if _s <= max_spread_stat and _s > 0:
                ans[_s-1].append(abs(term.val))
        for i in range(len(ans)):
            ans[i].sort()
            new = []
            for _p in range(len(ans[i])):
                if _p != len(ans[i])-1 and abs(ans[i][_p] - ans[i][_p+1]) < tol:
                    continue
                new.append(ans[i][_p])
            ans[i] = new.copy()
        return ans
    def RG_violation(self, leading, norm, tol: float = 1e-9):
        norm = abs(norm)
        for term in self.terms.values():
            if abs(term) > norm:
                op = self.find_val(abs(term), tol = tol)
                comm = op * leading + (-1.) * leading * op
                comm.truncate(tol = tol)
                if len(comm) > 0:
                    return op
        return None
    def RG_quality(self, leading_sites, gap):
        norm = 0
        for term in self.terms.values():
            if abs(term) > norm and term.is_overlap(leading_sites):
                norm = abs(term)
        return abs(norm / gap)

def Tr(op, sites):
    ans = Operator([])
    for term in op.terms.values():
        if (term.Xs | term.Ys) & sites == 0:
            ans.append(term)
    if len(ans) == 0:
        return Operator(0)
    ans *= 2. ** len(to_set(sites))
    
    return ans

def tensor_append(A, T):
    # returns the tensor product of operator A and term T, assuming they do not overlap
    ans = Operator([])
    for a in A.terms.values():
        term = Term(a.Xs | T.Xs, a.Ys | T.Ys)
        term.val = a.val * T.val
        ans.append(term)
    return ans