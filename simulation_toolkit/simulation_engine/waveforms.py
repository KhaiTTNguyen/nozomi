import numpy as np
from scipy.optimize import linprog

gamma = 267.513; # rad/ms/mT

dt0 = 0.0001 # default time step is 0.0001 ms

def grad_lobe_by_area(dt,grad_area,grad_lim,slew_lim,accel_lim,bp=False):
    '''
    generates a minimum duration gradient waveform with a given gradient area 

    dt = time step, ms
    grad_area = area of the gradient waveform, mT/m * ms
    grad_lim = maximum gradient strength, mT/m
    slew_lim = slew rate limit, mT/m/ms
    accel_lim = slew rate limit, mT/m/ms/s
    bp = False, is the waveform bipolar (or unipolar)
    '''
    nt = np.ceil(3*grad_area/grad_lim/dt)

    f = np.abs(np.arange(-nt/2,nt/2,1))
    if bp:
        f = np.abs(np.arange(0,nt,1))

    # Equality constraints 
    Aeq = np.zeros((3,nt))
    beq = np.zeros((3,))
    # sum(gradient) = area
    Aeq[0,:] = np.ones(nt)
    beq[0] = grad_area/dt
    # G[0] and G[-1] = 0
    Aeq[1,0] = 1
    beq[1] = 0
    Aeq[2,-1] = 1
    beq[2] = 0

    # Inequality constraints
    # gradient must be between 0 and grad_lim
    A1 = np.eye(nt)
    b1 = grad_lim*np.ones(nt)
    A2 = -np.eye(nt)
    b2 = np.zeros(b1.size)

    # max gradient slew rate
    Adiff1 = np.diag(np.ones(nt)) - np.diag(np.ones(nt-1),1)
    Adiff1 = Adiff1[0:-1,:]
    bdiff1 = slew_lim*np.ones(nt-1)*dt
    Adiff2 = -Adiff1
    bdiff2 = bdiff1

    # max gradient acceleration limit
    Aacc1 = np.diag(np.ones(nt-1),-1) - 2*np.diag(np.ones(nt),0) + np.diag(np.ones(nt-1),1)
    Aacc1 = Aacc1[1:-1,:]
    bacc1 = accel_lim*np.ones(nt-2)*dt*dt
    Aacc2 = -Aacc1
    bacc2 = bacc1

    # Combine all inequality constraints
    A = np.concatenate((A1,A2,Adiff1,Adiff2,Aacc1,Aacc2),axis=0)
    b = np.concatenate((b1,b2,bdiff1,bdiff2,bacc1,bacc2),axis=0)

    # generate shaped gradient waveform
    grad = linprog(f, A, b, Aeq, beq)

    if grad.success is False:
        raise RuntimeError("Optimization failed")
    
    gropt = grad.x
    # Truncate waveform
    gropt = gropt[np.abs(gropt)>1e-4]

    # duplicate, if the waveform is bipolar
    if bp:
        gropt = np.concatenate((np.array((0,)),-gropt[-1::-1],np.array((0,)),gropt,np.array((0,))))

    return gropt

def grad_lobe_by_dur(dt,nt,grad_lim,slew_lim,accel_lim):
    '''
    generates a maximum area gradient waveform with a given duration

    dt = time step, ms
    nt = number of samples in the waveform
    grad_lim = maximum gradient strength, mT/m
    slew_lim = slew rate limit, mT/m/ms
    accel_lim = slew rate limit, mT/m/ms/s
    '''

    f = np.ones([nt,])

    # Equality constraints 
    Aeq = np.zeros((2,nt))
    beq = np.zeros((2,))
    # G[0] and G[-1] = 0
    Aeq[0,0] = 1
    beq[0] = 0
    Aeq[1,-1] = 1
    beq[1] = 0

    # Inequality constraints
    # gradient must be between 0 and grad_lim
    A1 = np.eye(nt)
    b1 = grad_lim*np.ones(nt)
    A2 = -np.eye(nt)
    b2 = np.zeros(b1.size)

    # max gradient slew rate
    Adiff1 = np.diag(np.ones(nt)) - np.diag(np.ones(nt-1),1)
    Adiff1 = Adiff1[0:-1,:]
    bdiff1 = slew_lim*np.ones(nt-1)*dt
    Adiff2 = -Adiff1
    bdiff2 = bdiff1

    # max gradient acceleration limit
    Aacc1 = np.diag(np.ones(nt-1),-1) - 2*np.diag(np.ones(nt),0) + np.diag(np.ones(nt-1),1)
    Aacc1 = Aacc1[1:-1,:]
    bacc1 = accel_lim*np.ones(nt-2)*dt*dt
    Aacc2 = -Aacc1
    bacc2 = bacc1

    # Combine all inequality constraints
    A = np.concatenate((A1,A2,Adiff1,Adiff2,Aacc1,Aacc2),axis=0)
    b = np.concatenate((b1,b2,bdiff1,bdiff2,bacc1,bacc2),axis=0)

    # generate shaped gradient waveform
    grad = linprog(f, A, b, Aeq, beq)

    if grad.success is False:
        raise RuntimeError("Optimization failed")
    
    gropt = grad.x

    return gropt


class DiffGradWaveform:

    def calculate_bvalue(self,gmax=None):
        ''' 
        calculate the b-value of gwave in units of ms/um^2
        '''
        if gmax is None:
            gmax = self.gmax
        gwave_um = self.gwave/10e3; # mT/um
        return np.sum(np.cumsum(gmax*gamma*gwave_um)**2)*self.dt**3; # ms/um^2

    def calculate_bvalue_from_wave(self):
            ''' 
            calculate the b-value of gwave in units of ms/um^2
            following equation 15 in https://doi.org/10.1002/nbm.1520
            '''
            gwave_um = self.wave/10e3; # mT/um
            return np.sum(np.cumsum(gamma*gwave_um)**2)*self.dt**3; # ms/um^2 

    def set_bvalues(self,b):
        b0 = self.calculate_bvalue(1)
        self.b = b
        self.gmax = np.sqrt(b/b0)

    def set_gmax(self,gmax):
        self.gmax = gmax
        self.b = self.calculate_bvalue(gmax)

class PGDiffWaveform(DiffGradWaveform):
    '''
    class to generate a pulse gradient diffusion waveform
    '''
    def __init__(self,big_delta,little_delta,te,time_step=dt0):
        self.big_delta = big_delta
        self.little_delta = little_delta
        self.te = te
        self.dt = time_step

        self.generate_waveform()

    def generate_waveform(self):
        self.t = np.arange(self.dt,self.te+self.dt,self.dt)
        self.wave = np.zeros_like(self.t)
        
        first_lobe = (self.t > (self.te/2 - self.big_delta/2 - self.little_delta/2)) \
                & (self.t < (self.te/2 - self.big_delta/2 + self.little_delta/2))
        self.wave[first_lobe] = 1

        second_lobe = (self.t > (self.te/2 + self.big_delta/2 - self.little_delta/2)) \
                & (self.t < (self.te/2 + self.big_delta/2 + self.little_delta/2))
        self.wave[second_lobe] = -1

class CosineOGDiffWaveform(DiffGradWaveform):
    '''
    Class to generate a cosine-modulated oscillating gradient diffusion waveform
    '''
    def __init__(self, num_oscillations, frequency, gmax, te, time_step=dt0):
        self.num_oscillations = num_oscillations
        self.frequency = frequency
        self.gmax = gmax
        self.te = te
        self.dt = time_step

        self.generate_waveform()

    def generate_waveform(self):
        self.t = np.arange(self.dt, self.te+self.dt, self.dt)
        self.wave = np.zeros_like(self.t)
        
        # Calculate timing parameters
        oscillation_period = 1.0 / self.frequency * 1000 #
        cosine_duration = self.num_oscillations * oscillation_period
        ramp_duration = oscillation_period / 8  # 1/4 cycle of double frequency
        
        # Total duration of a single waveform (first half of TE)
        half_waveform_duration = cosine_duration + 2 * ramp_duration
        
        # Start time for first waveform (centered in first half of TE)
        t1_start = self.te / 4 - half_waveform_duration / 2
        
        # First waveform (positive)
        for i, t in enumerate(self.t):
            if t1_start <= t < t1_start + ramp_duration:
                # Initial ramp-up (1/4 sine of double frequency)
                phase = (t - t1_start) / ramp_duration * np.pi / 2
                self.wave[i] = self.gmax * np.sin(phase)
            elif t1_start + ramp_duration <= t < t1_start + ramp_duration + cosine_duration:
                # Cosine oscillations
                phase = 2 * np.pi * self.frequency * (t - (t1_start + ramp_duration))
                self.wave[i] = self.gmax * np.cos(phase)
            elif t1_start + ramp_duration + cosine_duration <= t < t1_start + 2 * ramp_duration + cosine_duration:
                # Final ramp-down (1/4 sine of double frequency)
                # Mirror image of ramp-up: gmax to 0
                phase = (t - (t1_start + ramp_duration + cosine_duration)) / ramp_duration * np.pi / 2
                self.wave[i] = self.gmax * np.cos(phase)  # Starts at gmax, ends at 0
        
        # Second waveform (negative) - mirroring the first waveform
        t2_start = 3 * self.te / 4 - half_waveform_duration / 2
        
        for i, t in enumerate(self.t):
            if t2_start <= t < t2_start + ramp_duration:
                # Initial ramp-up (1/4 sine of double frequency) - negative
                phase = (t - t2_start) / ramp_duration * np.pi / 2
                self.wave[i] = -self.gmax * np.sin(phase)
            elif t2_start + ramp_duration <= t < t2_start + ramp_duration + cosine_duration:
                # Cosine oscillations - negative
                phase = 2 * np.pi * self.frequency * (t - (t2_start + ramp_duration))
                self.wave[i] = -self.gmax * np.cos(phase)
            elif t2_start + ramp_duration + cosine_duration <= t < t2_start + 2 * ramp_duration + cosine_duration:
                # Final ramp-down (1/4 sine of double frequency) - negative
                phase = (t - (t2_start + ramp_duration + cosine_duration)) / ramp_duration * np.pi / 2
                self.wave[i] = -self.gmax * np.cos(phase)  # Starts at -gmax, ends at 0