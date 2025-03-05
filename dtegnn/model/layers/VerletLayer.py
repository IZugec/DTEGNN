import torch.nn as nn

class VerletC(nn.Module):
    def __init__(self):
        super(VerletC, self).__init__()
        
    def accel(self, masses, force):
        ev_A_to_N = 1.602176565e-9
        u_to_kg = 1.66053904e-27
        m_to_A = 1e10
        s_to_fs = 1e15

        masses = masses * u_to_kg
        force = force * ev_A_to_N
        acc = force/masses[:,None]
        acc = acc * m_to_A/pow(s_to_fs,2)
        
        return acc
    
    def forward(self, force, pos, vel, potim, masses, cell):
        a = self.accel(masses, force)
        pos_new = pos + vel * potim[:,None] + 0.5 * a * pow(potim[:,None],2)
        
        return pos_new

class VerletV(nn.Module):
    def __init__(self):
        super(VerletV, self).__init__()

    def accel(self,masses, force):
        ev_A_to_N = 1.602176565e-9
        u_to_kg = 1.66053904e-27
        m_to_A = 1e10
        s_to_fs = 1e15

        masses = masses * u_to_kg
        force = force * ev_A_to_N
        acc = force / masses[:,None]  # Acc in m/s^2
        acc = acc * m_to_A / pow(s_to_fs, 2)

        return acc

    def forward(self, force, force_old, vel_old, potim, masses):
        a_old = self.accel(masses, force_old)
        a = self.accel(masses, force)
        v_new = vel_old + 0.5 * (a_old + a) * potim[:,None]
        
        return v_new
