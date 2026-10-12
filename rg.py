from dataclasses import dataclass, field

import numpy as np
from binary import bin_parity, to_bin, to_set
from config import Thresholds
from op import Kron_Part, Operator, Spin, Term, Tr, get_new_spin_ind, tensor_append


@dataclass
class RGResult:
    sample_id: int

    pair_freq: np.ndarray | None = None

    spin_stat: str         = ""
    quals: list[float]     = field(default_factory = list)
    energy: float | None   = None
    state: list[Kron_Part] = field(default_factory = list)

    warnings: list[str] = field(default_factory = list)
    error: str | None   = None

def RSRG(sample_id, H, spins, pos, sec_choices, thres: Thresholds):

    step = 0
    pair_freq = np.zeros_like(pos)
    spin_stat = ""
    quals = []
    state = []

    H = H.copy()

    lenH_max = 0

    res = RGResult(sample_id)



    while True:
        lenH_max = max(len(H), lenH_max)
        
        _max = H.max_term(max_spread = 2)
        if _max is None or _max.val == 0.:
            # End of RSRG.

            # Identify the energy term
            id_term = H.find_term(Term(1))
            if id_term is None:
                _en = 0.
            else:
                _en = id_term.val

            res.pair_freq = pair_freq
            res.energy    = _en
            res.state     = state
            res.quals     = quals
            res.spin_stat = spin_stat
            
            return res
        
        sec = 0
        # "0" is a spaceholder to be filled
        if step < len(sec_choices):
            # If step < len(sec_choices), then the branching is dictated by secs[step]
            sec = sec_choices[step]
        else:
            # Iterating over all the branching choices
            raise NotImplementedError

        spin_stat += "#%d f: " % step
        # The "f" is a placeholder for projection type, default to be "task failed"
        m_cnt = np.zeros(len(pos)+1, dtype = int)
        for _s in spins.values():
            m_cnt[_s.m] += 1
        _first = True
        for _m in range(len(m_cnt)):
            if m_cnt[_m] != 0:
                if _first == False:
                    spin_stat += ", "
                spin_stat += "%d %d" % (_m, m_cnt[_m])
                _first = False
        spin_stat += " "

        sites = _max.Xs | _max.Ys

        

        if len(to_set(sites)) == 1: 
            i = to_set(sites).pop()
            _i = to_bin({i})
            H0 = Operator([_max])
            H_eff  = H + (-1.) * H0
            H_eff.truncate(tol = thres.IR_cutoff)
            H1 = H_eff.collect_overlap_terms(sites)
            H_eff  = H_eff + (-1.) * H1
            H_eff.truncate(tol = thres.IR_cutoff)
            
            _violation = H1.RG_violation(Operator([_max]), abs(_max), tol = thres.IR_cutoff)
            if _violation is not None:
                _violation_message  = str(list(pos)) + "\n"
                _violation_message += str(sec_choices[:step]) + "\n"
                _violation_message += str(spins) + "\n"
                _violation_message += "WARNING: non-commuting larger terms exist: " + str(_violation) + "\n"
                _violation_message += "... traceback: leading    = " + str(H0) + "\n"
                res.warnings.append(_violation_message)

            if _max.Xs == _i and _max.Ys == _i: # H0 is of form Z.
                _message  = str(list(pos)) + "\n"
                _message += str(sec_choices[:step]) + "\n"
                _message += str(spins) + "\n"
                _message += "EXCEPTION: dangling Z term present: " + str(_max)
                res.warnings.append(_message)
                raise RuntimeError("Dangling Z term present: " + str(_max))
            elif _max.Xs == _i and _max.Ys == 0: # H0 is of form X.
                J0 = _max.val
                _qual = H1.RG_quality(sites, abs(J0))

                for ind in to_set(spins[i].sites):
                    pair_freq[ind] = abs(2. * J0)
                

                if sec == 1 or sec == 3:
                    P = .5 * (Operator(1) +       Operator([Term(_i,0)]))
                    K = Operator(1) + (-1.)*Operator([Term(_i,0)])
                    H_new1 = Tr(P*H1, sites)
                    H_new2 = (1./(4.*J0)) * Tr(P * compute_HKH(H1, K, J0, tol = thres.IR_cutoff), sites)

                    H = H_eff + J0 * Operator(1) + H_new1 + H_new2
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)
                    
                    state.append(Kron_Part(2*J0, 1, spins[i].us, spins[i].ds))
                    spins.pop(i)
                    spin_stat = spin_stat.replace("f", "X%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1
                    continue

                if sec == 2:
                    P = .5 * (Operator(1) + (-1.)*Operator([Term(_i,0)]))
                    K =  Operator(1) +       Operator([Term(_i,0)])
                    H_new1 = Tr(P*H1, sites)
                    H_new2 = (-1./(4.*J0)) * Tr(P * compute_HKH(H1, K, J0, tol = thres.IR_cutoff), sites)

                    H = H_eff + (-J0) * Operator(1) + H_new1 + H_new2
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)

                    state.append(Kron_Part(2*J0, -1, spins[i].us, spins[i].ds))
                    spins.pop(i)
                    spin_stat = spin_stat.replace("f", "X%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1
                    continue
                
            else:
                _message  = str(list(pos)) + "\n"
                _message += str(sec_choices[:step]) + "\n"
                _message += str(spins) + "\n"
                _message += "EXCEPTION: unexpected single-site term: " + str(_max)
                res.warnings.append(_message)
                raise RuntimeError("Unexpected single-site term: " + str(_max))

        else:
            i, j = to_set(sites)
            _ij = to_bin({i, j})
            _i  = to_bin({i})
            _j  = to_bin({j})

            XX = H.find_term(Term(_ij, 0  ))
            YY = H.find_term(Term(0  , _ij))
            ZZ = H.find_term(Term(_ij, _ij))

            if (ZZ is not None and XX is not None and abs(ZZ.val) > abs(XX.val)) or XX is None:
                # ZZ projection
                H0 = Operator([ZZ])
                J0 = -ZZ.val
                H_eff = H + (-1.0)*H0
                H_eff.truncate(tol = thres.IR_cutoff)
                H1 = H_eff.collect_overlap_terms(sites)
                H_eff  = H_eff + (-1.0) * H1
                H_eff.truncate(tol = thres.IR_cutoff)
                # Now H contains only the non-overlapping terms

                _violation = H1.RG_violation(Operator([ZZ]), abs(J0), tol = thres.IR_cutoff)
                if _violation is not None:
                    _violation_message  = str(list(pos)) + "\n"
                    _violation_message += str(sec_choices[:step]) + "\n"
                    _violation_message += str(spins) + "\n"
                    _violation_message += "WARNING: non-commuting larger terms exist: " + str(_violation) + "\n"
                    _violation_message += "... traceback: leading    = " + str(H0) + "\n"
                    res.warnings.append(_violation_message)

                # _qual = H1.RG_quality(Operator([ZZ]), abs(J0))
                _qual = H1.RG_quality(sites, abs(J0))

                if sec == 2:
                    K =  Operator([Term(1), Term(_ij, _ij)])
                    P_I  = .25 * Operator([Term(1)       , (-1.)*Term(_ij, _ij)])
                    P_iZ = .25 * Operator([Term(_i, _i),       Term(_j, _j)])
                    P_X  = .25 * Operator([Term(_ij,0), (-1.)*Term(0, _ij)])
                    P_Y  = .25 * Operator([Term(_i,_j) ,       Term(_j, _i)])

                    HKH = (1./(4*J0)) * compute_HKH(H1, K, J0, tol = thres.IR_cutoff)
                    HKH.truncate(tol = thres.IR_cutoff)

                    H_new_I  = Tr(P_I  * HKH, sites)
                    H_new_iZ = Tr(P_iZ * HKH, sites)
                    H_new_X  = Tr(P_X  * HKH, sites)
                    H_new_Y  = Tr(P_Y  * HKH, sites)

                    k = get_new_spin_ind(spins)
                    _k = to_bin({k})

                    H_new1 = Tr(P_I*H1, sites) + tensor_append(Tr(P_iZ*H1, sites), (-1.)*Term(_k, _k)) \
                                + tensor_append(Tr(P_X*H1, sites), Term(_k, 0)) \
                                + tensor_append(Tr(P_Y*H1, sites), Term(0, _k))
                    H_new2 = H_new_I + tensor_append(H_new_iZ, (-1.)*Term(_k, _k)) \
                            + tensor_append(H_new_X, Term(_k, 0)) + tensor_append(H_new_Y, Term(0, _k))
                    H_new1.truncate(tol = thres.IR_cutoff)
                    H_new2.truncate(tol = thres.IR_cutoff)

                    H = H_eff + J0*Operator(1) + H_new1 + H_new2
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)

                    _s1, _s2 = (spins[i], spins[j])
                    spins[k] = Spin(_s1.m + _s2.m, _s1.us | _s2.us, _s1.ds | _s2.ds)
                    spins.pop(i)
                    spins.pop(j)
                    spin_stat = spin_stat.replace("f", "ZZ%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1
                    continue

                if sec == 1 or sec == 3:
                    K = Operator([Term(1), (-1.)*Term(_ij, _ij)])
                    P_I  = .25 * Operator([Term(1)       ,       Term(_ij, _ij)])
                    P_iZ = .25 * Operator([Term(_i, _i), (-1.)*Term(_j, _j)])
                    P_X  = .25 * Operator([Term(_ij,0),       Term(0, _ij)])
                    P_Y  = .25 * Operator([Term(_j, _i), (-1.)*Term(_i, _j)])

                    _s1 = spins[i]
                    _s2 = spins[j]
                    if _s1.m < _s2.m:
                        # We always want |up> to have non-negative magnetization,
                        # so if the magnetization of site s2 is larger than that of s1,
                        # we need to swap their notation. This is done by adding negative
                        # signs to P_Z and P_Y.
                        P_iZ *= (-1.)
                        P_Y  *= (-1.)

                    HKH = (-1./(4*J0)) * compute_HKH(H1,  K, J0, tol = thres.IR_cutoff)
                    HKH.truncate(tol = thres.IR_cutoff)
                    
                    H_new_I  = Tr(P_I  * HKH, sites)
                    H_new_iZ = Tr(P_iZ * HKH, sites)
                    H_new_X  = Tr(P_X  * HKH, sites)
                    H_new_Y  = Tr(P_Y  * HKH, sites)

                    k = get_new_spin_ind(spins)
                    _k = to_bin({k})

                    H_new1 = Tr(P_I*H1, sites) + tensor_append(Tr(P_iZ*H1, sites), (-1.)*Term(_k, _k)) \
                                + tensor_append(Tr(P_X*H1, sites), Term(_k, 0)) \
                                + tensor_append(Tr(P_Y*H1, sites), Term(0, _k))
                    H_new2 = H_new_I + tensor_append(H_new_iZ, (-1.)*Term(_k, _k)) \
                            + tensor_append(H_new_X, Term(_k, 0)) + tensor_append(H_new_Y, Term(0, _k))
                    H_new1.truncate(tol = thres.IR_cutoff)
                    H_new2.truncate(tol = thres.IR_cutoff)

                    H = H_eff + (-J0)*Operator(1) + H_new1 + H_new2
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)

                    _s1, _s2 = (spins[i], spins[j])
                    if _s1.m > _s2.m:
                        spins[k] = Spin(_s1.m - _s2.m, _s1.us | _s2.ds, _s1.ds | _s2.us)
                    else:
                        spins[k] = Spin(_s2.m - _s1.m, _s1.ds | _s2.us, _s1.us | _s2.ds)
                    spins.pop(i)
                    spins.pop(j)
                    spin_stat = spin_stat.replace("f", "ZZ%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1
                    continue
            else:
                # XX+YY projection
                H0 = Operator([XX, YY])
                J0 = XX.val
                H_eff = H + (-1.0)*H0
                H_eff.truncate(tol = thres.IR_cutoff)
                H1 = H_eff.collect_overlap_terms(sites)
                H_eff  = H_eff + (-1.0) * H1
                H_eff.truncate(tol = thres.IR_cutoff)
                # Now H contains only the non-overlapping terms
                
                _anisotropy = abs(XX.val - YY.val)
                if _anisotropy / min(abs(XX.val), abs(YY.val)) > thres.anisotropy and _anisotropy > 100.*thres.IR_cutoff:
                    _message  = str(list(pos)) + "\n"
                    _message += str(sec_choices[:step]) + "\n"
                    _message += str(spins) + "\n"
                    _message += f"WARNING: anisotropic max term, ratio = {_anisotropy:.3e}" + "\n" + \
                            "         XX = " + str(XX) + ", YY = " + str(YY) + ", ZZ = " + str(ZZ)
                    res.warnings.append(_message)
                    _mean = (XX.val + YY.val) / 2.
                    XX.val = _mean
                    YY.val = _mean

                J0 = XX.val
                
                _violation = H1.RG_violation(Operator([XX, YY]), abs(J0), tol = thres.IR_cutoff)
                if _violation is not None:
                    _violation_message  = str(list(pos)) + "\n"
                    _violation_message += str(sec_choices[:step]) + "\n"
                    _violation_message += str(spins) + "\n"
                    _violation_message += "WARNING: non-commuting larger terms exist: " + str(_violation) + "\n"
                    _violation_message += "... traceback: leading    = " + str(H0) + "\n"
                    res.warnings.append(_violation_message)

                _qual = H1.RG_quality(sites, abs(J0))
                

                if sec == 2: # project onto m = +/- 1
                    K    = Operator([Term(_ij, 0), Term(0, _ij)])
                    P_I  = .25 * Operator([Term(1)       , (-1.)*Term(_ij, _ij)])
                    P_iZ = .25 * Operator([Term(_i, _i),       Term(_j, _j)])
                    P_X  = .25 * Operator([Term(_ij,0), (-1.)*Term(0, _ij)])
                    P_Y  = .25 * Operator([Term(_i, _j),       Term(_j, _i)])
                    HKH = (-1./(4*J0)) * compute_HKH(H1, K, J0, tol = thres.IR_cutoff)
                    HKH.truncate(tol = thres.IR_cutoff)

                    H_new_I  = Tr(P_I  * HKH, sites)
                    H_new_iZ = Tr(P_iZ * HKH, sites)
                    H_new_X  = Tr(P_X  * HKH, sites)
                    H_new_Y  = Tr(P_Y  * HKH, sites)
                    
                    k = get_new_spin_ind(spins)
                    _k = to_bin({k})
                    H_new1 = Tr(P_I*H1, sites) + tensor_append(Tr(P_iZ*H1, sites), (-1.)*Term(_k, _k)) \
                                + tensor_append(Tr(P_X*H1, sites), Term(_k, 0)) \
                                + tensor_append(Tr(P_Y*H1, sites), Term(0, _k))
                    H_new2 = H_new_I + tensor_append(H_new_iZ, (-1.)*Term(_k, _k)) \
                            + tensor_append(H_new_X, Term(_k, 0)) + tensor_append(H_new_Y, Term(0, _k))
                    H_new1.truncate(tol = thres.IR_cutoff)
                    H_new2.truncate(tol = thres.IR_cutoff)
                    H = H_eff + H_new1 + H_new2
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)

                    _s1, _s2 = (spins[i], spins[j])
                    spins[k] = Spin(_s1.m + _s2.m, _s1.us | _s2.us, _s1.ds | _s2.ds)
                    spins.pop(i)
                    spins.pop(j)
                    spin_stat = spin_stat.replace("f", "XX%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1
                    continue

                if sec == 1: # project onto m = 0 singlet
                    K = Operator([Term(1), (-1.)*Term(_ij, _ij)])
                    P = .25 * Operator([Term(1), Term(_ij, _ij), (-1.)*Term(_ij,0), (-1.)*Term(0, _ij)])
                    H_new1 = Tr(P*H1, sites)
                    H_new2 = (-1./(4*J0)) * Tr(P * compute_HKH(H1, K, J0, tol = thres.IR_cutoff), sites)
                    H_new1.truncate(tol = thres.IR_cutoff)
                    H_new2.truncate(tol = thres.IR_cutoff)

                    H  = H_eff + (-2.*J0)*Operator(1) + H_new1 + H_new2
                    K =  Operator([Term(1), Term(_ij, _ij), Term(_ij,0), Term(0, _ij)])
                    H_jump = (-1./(16*J0)) * Tr(P * compute_HKH(H1, K, J0, tol = thres.IR_cutoff), sites)
                    H = H + H_jump
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)

                    _s1, _s2 = (spins[i], spins[j])
                    for ind in to_set(_s1.sites | _s2.sites):
                        pair_freq[ind] = abs(4. * J0)
                    state.append(Kron_Part(4*J0, -1, _s1.us | _s2.ds, _s1.ds | _s2.us))
                    spins.pop(i)
                    spins.pop(j)
                    spin_stat = spin_stat.replace("f", "XX%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1
                    continue
                
                if sec == 3:
                    
                    K = Operator([Term(1), (-1.)*Term(_ij, _ij)])
                    P = .25 * Operator([Term(1), Term(_ij, _ij), Term(_ij,0), Term(0, _ij)])

                    H_new1 = Tr(P*H1, sites)
                    H_new2 = (1./(4*J0)) * Tr(P * compute_HKH(H1, K, J0, tol = thres.IR_cutoff), sites)
                    H_new1.truncate(tol = thres.IR_cutoff)
                    H_new2.truncate(tol = thres.IR_cutoff)

                    
                    
                    H  = H_eff + (2.*J0)*Operator(1) + H_new1 + H_new2
                    K =  Operator([Term(1), Term(_ij, _ij), (-1.)*Term(_ij,0), (-1.)*Term(0, _ij)])
                    H_jump = (1./(16*J0)) * Tr(P * compute_HKH(H1, K, J0, tol = thres.IR_cutoff), sites)
                    H = H + H_jump
                    H.truncate(tol = thres.IR_cutoff, max_spread = thres.max_spread, max_len = thres.max_lenH)
                    
                    _s1, _s2 = (spins[i], spins[j])
                    for ind in to_set(_s1.sites | _s2.sites):
                        pair_freq[ind] = abs(4. * J0)
                    state.append(Kron_Part(4*J0, 1, _s1.us | _s2.ds, _s1.ds | _s2.us))
                    spins.pop(i)
                    spins.pop(j)
                    spin_stat = spin_stat.replace("f", "XX%d")%sec_choices[step]
                    quals.append(_qual)
                    step += 1

                    
                    continue



def compute_HKH(H: Operator, K: Operator, denom: float, tol: float = 1e-9):
    # denom provides the energy scale of the to-be denominator, as a reference for the truncation
    KH = K * H

    ans = Operator([])
    H_it  =  H.terms.values()
    KH_it = KH.terms.values()

    for a in H_it:
        # We can safely throw away terms such that a.val * b.val / denom < NUM_THRES
        if a.val == 0.:
            continue
        thres = abs(denom * tol / a.val)
        for b in KH_it:
            if abs(b.val) < thres:
                continue
            _val = a.val * b.val
            term = Term(a.Xs ^b.Xs, a.Ys ^ b.Ys)
            term.val = _val
            term.val *= 1. - 2. * bin_parity(a.Ys & b.Xs)
            ans.append_nocopy(term)
    return ans