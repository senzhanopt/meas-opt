import pybamm

class Battery:
    
    def __init__(self, p_max = 15.0, e_max = 10.0, soc = 0.5, eta_ch = 0.99, 
                 eta_disc = 0.99, temp = 25, cell_model = "dfn"):
        self.p_max = p_max
        self.e_max = e_max
        self.soc = soc
        self.eta_ch = eta_ch
        self.eta_disc = eta_disc
        self.temp = temp
        self.cell_model = cell_model
        
        
    def charge(self, p_ch = 0.0, length_t = 0.25):
        p_ch_abs = max(p_ch, 0.0)
        p_dis_abs = max(-p_ch, 0.0)
        self.soc += (p_ch_abs*self.eta_ch-p_dis_abs/self.eta_disc)\
            *length_t/self.e_max
             
            
        
if __name__ == "__main__":
    bat1 = Battery()
    