import matplotlib.pyplot as plt
import scipy
import scipy.integrate
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import numpy as np

ml_per_sec_to_L_per_min = 60/1000

def gaussian_elastance(self, tau):
    return np.exp(-self.e_std**(-2)*(tau-self.t_peak)**2)

def stergiopolous_elastance(self, t):
    """
    Computes the normalized elastance at time t, according to the shape parameters given by
    Stergiopolus 1994.
    """
    # Note changing these may result in a non-nomralized elastance curve
    a1 = 0.708 * self.t_peak/self.T
    a2 = 1.677 * a1
    n1 = 1.32
    n2 = 21.9
    alpha = 1.672
    shapeFunction1 = (t/(a1*self.T))**n1 / (1.0 + (t / (a1*self.T)) ** n1)
    shapeFunction2 = (1.0 + (t/(a2*self.T))**n2)
    e = alpha * shapeFunction1/shapeFunction2
    return e


class VaryingElastance():
    def __init__(self):
        self.E_max = 3.0
        self.E_min = 0.3
        self.t_peak = 0.4
        self.T = 60/73
        self.Z_ao = 0.1
        self.C_ao = 1
        self.R_sys = 1
        self.C_sv = 15
        self.R_mv = 0.006
        self.V_tot = 100 + 80 + 100
        self.par_dict = self.__dict__.copy()
        #self.par_dict.pop("self")
        self.pleural_pressure_func = lambda t: -4
        self.elastance_fcn = stergiopolous_elastance

        self.age = 54 #yrs
        self.wt = 82.1 #kg
        self.ht = 181.6 #cm
        self.sex = "M"
        

    

    def print_params(self):
        print("Shifted parameters")
        print(list(self.par_dict.keys()))
        print(list(self.par_dict.values()))
        return list(self.par_dict.values())   

    def reset_exercise_shift(self):
        # Store initial values
        self.E_max = self.E_max_ux
        self.T = self.T_ux
        self.C_ao = self.C_ao_ux
        self.R_sys = self.R_sys_ux

    def elastance(self, tau):
        return self.elastance_fcn(self, tau)

    

    def set_pars(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
                self.par_dict[key] = val
            else:
                print("Warning: object has no attribute %s" % key)
        #self.set_vol()

    def set_subject(self, **subjectkwargs):
        for key, val in subjectkwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, val)
            else:
                print("Warning: object has no attribute %s" % key)


    def calc_consistent_initial_values(self, V_lv_0=100, P_ao_0=100):
        V_ao_0 = self.C_ao*P_ao_0
        P_sv_0 = (self.V_tot - V_lv_0 - V_ao_0)/self.C_sv
        u0 = (V_lv_0, P_ao_0, P_sv_0)
        return u0


    def rhs(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        P_sv = u[2]
        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv + self.pleural_pressure_func(t)
        Q_lvao = (P_lv > P_ao)*(P_lv - P_ao)/self.Z_ao
        Q_aosv = (P_ao - P_sv)/self.R_sys
        Q_svlv = (P_sv > P_lv)*(P_sv - P_lv)/self.R_mv

        der_V_lv = Q_svlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_P_sv = (Q_aosv - Q_svlv)/self.C_sv
        der_u = [der_V_lv, der_P_ao, der_P_sv]
        return der_u


    def calc_all(self, t, u):
        V_lv = u[0]
        P_ao = u[1]
        P_sv = u[2]
        tau = np.mod(t, self.T)
        e_t = self.elastance(tau)
        E = (self.E_max-self.E_min)*e_t + self.E_min
        P_lv = E * V_lv + self.pleural_pressure_func(t)
        Q_lvao = (P_lv - P_ao)/self.Z_ao * (P_lv > P_ao)
        Q_aosv = (P_ao-P_sv)/self.R_sys
        Q_svlv = (P_sv - P_lv)/self.R_mv * (P_sv > P_lv)
        P_meas = np.maximum(P_lv, P_ao)
        P_ao = P_meas
        P_sys = np.max(P_ao)
        P_dia = np.min(P_ao)
        PP = P_sys - P_dia
        V_sys = np.max(V_lv)
        V_dia = np.min(V_lv)
        SV = V_sys - V_dia
        MVP = np.mean(P_sv)

        all_vars = locals()
        del all_vars["self"]
        del all_vars["u"]

        der_V_lv = Q_svlv - Q_lvao
        der_P_ao = (Q_lvao - Q_aosv)/self.C_ao
        der_P_vc = (Q_aosv - Q_svlv)/self.C_sv
        der_u = [der_V_lv, der_P_ao, der_P_vc]
        return der_u, all_vars



def poly_func(x, a, b, c):
    return a*(np.array(x)**2) + b*np.array(x) + c

def exp_func(x, a, b, c, d):
    return a*(np.exp(-(x-c)/b)) + d

def plot_pv_loop(var_dict):
    plt.figure()
    plt.plot(var_dict["V_lv"], var_dict["P_lv"])
    plt.ylabel("Pressure")
    plt.xlabel("Volume")
    
    
def plot_pressures(var_dict, ax=None):
    t = var_dict["t"]
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.plot(t, var_dict["P_lv"], label="P_lv")
    #plt.plot(t, var_dict["P_ao"], label="P_ao")
    ax.plot(t, np.maximum(var_dict["P_ao"], var_dict["P_lv"]), label="P_ao")
    ax.plot(t, var_dict["P_sv"], label="P_sv")
    ax.set_ylabel("Pressure [mmHg]")
    return ax


def plot_flows(var_dict):
    t = var_dict["t"]
    plt.figure()
    plt.plot(t, var_dict["Q_lvao"], label="Q_av")
    plt.plot(t, var_dict["Q_aosv"], label="Q_sys")
    plt.plot(t, var_dict["Q_svlv"], label="Q_mv")
    plt.ylabel("Flow [ml/s]")


def plot_all(var_dict):
    t = var_dict["t"]
    for var, values in var_dict.items():
        plt.figure()
        plt.plot(t, values)
        plt.ylabel(var)


def calc_summary(var_dict):
    P_ao = np.maximum(var_dict["P_ao"], var_dict["P_lv"])
    P_sys = np.max(P_ao)
    P_dia = np.min(P_ao)
    P_map = np.mean(P_ao)
    Q_max = np.max(var_dict["Q_lvao"])
    PP = P_sys - P_dia
    V_sys = np.max(var_dict["V_lv"])
    V_dia = np.min(var_dict["V_lv"])
    SV = V_sys - V_dia
    CO = ml_per_sec_to_L_per_min*SV/(var_dict["t"][-1] - var_dict["t"][0])
    MVP = np.mean(var_dict["P_sv"])
    stroke_work_1 = P_map*SV
    #stroke_work_int = np.trapz(var_dict["Q_lvao"]*var_dict["P_lv"], x=var_dict["t"])
    ret_dict = locals()
    del ret_dict["var_dict"]
    del ret_dict["P_ao"]
    return ret_dict


def test_set_pars():
    vewk3 = VaryingElastance()
    cool_pars = dict(E_maxs=0.5, E_max=2.9)
    vewk3.set_pars(E_maxs=0.5, E_max=2.9)


def solve_to_steady_state(model, n_cycles=5, n_eval_pts=100):
    u0 = model.calc_consistent_initial_values()
    t_span = (0, model.T*n_cycles)
    t_eval = model.T*np.linspace(n_cycles-1, n_cycles, n_eval_pts)
    sol = scipy.integrate.solve_ivp(model.rhs, t_span, u0, dense_output=True, method="RK45",
            atol=1e-10, rtol=1e-9)
    u_eval = sol.sol(t_eval)
    _, all_vars = model.calc_all(t_eval, u_eval)
    return all_vars, t_eval, sol


def test():
    vewk3 = VaryingElastance()
    vewk3.exercise_shift(0.5)
    vewk3.print_params()
    tmin = 0
    tmax = 10
    t_eval_vals = np.linspace(tmin, tmax, 1000)
    sol = solve_ivp(vewk3.rhs, (tmin, tmax), [100, 100, 10], max_step=0.05, t_eval=t_eval_vals)
    _, all_vars = vewk3.calc_all(sol.t, sol.y)


    for var, vals in all_vars.items():
        print(var)
        plt.figure()
        plt.plot(sol.t, vals, label=var)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
