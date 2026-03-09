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
        gmax is in mT/m; wave amplitude of 1 corresponds to 1 mT/m.
        '''
        if gmax is None:
            gmax = self.gmax
        gwave_um = self.gwave / 1e6  # mT/m -> mT/um (1 mT/m = 1e-6 mT/um)
        return np.sum(np.cumsum(gmax*gamma*gwave_um)**2)*self.dt**3; # ms/um^2

    def calculate_bvalue_from_wave(self):
            ''' 
            calculate the b-value of gwave in units of ms/um^2
            following equation 15 in https://doi.org/10.1002/nbm.1520
            Wave amplitude of 1 corresponds to gmax = 1 mT/m.
            1 mT/m = 1e-6 mT/um. More generally,
            b = \int_0^T [ dt * q(t) * q(t) ]
                
                where q(t) = \int_0^t gamma*[ g(t') dt'] (cumulative gradient moment)
                
                = \int_0^T [ dt * ( \int_0^t [gamma * g(t') dt'] )^2 ]

            '''
            gwave_um = self.wave / 10e3  # mT/m -> mT/um (1 mT/m = 1e-6 mT/um)
            # discrete form of \int_0^T [ dt * ( \int_0^t [gamma * g(t') dt'] )^2 ]
            return np.sum(np.cumsum(gamma*gwave_um)**2)*self.dt**3; # ms/um^2 -- 1 ms/um^2 = 1000 sec/mm^2

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

class ApodizedCosineOGSEWaveform(DiffGradWaveform):
    '''
    Class to generate apodized cosine modulated oscillating gradient waveform
    Based on Does et al. 2003 - Oscillating gradient measurements of water diffusion
    
    Structure: 1/4 sine (2x freq) + N cosine cycles + 1/4 sine (2x freq)
    Effective diffusion time: t_eff = T/(4N)
    '''
    def __init__(self, N_cycles, T_duration, te=None, gmax=1.0, time_step=dt0):
        """
        Parameters:
        N_cycles: Number of cosine cycles in the main oscillation
        T_duration: Duration of the active gradient in ms
        te: Total echo time (if None, uses 4*T_duration for encoding+decoding)
        gmax: Maximum gradient amplitude (normalized to 1.0)
        time_step: Time step for discretization
        """
        self.N_cycles = N_cycles
        self.T_duration = T_duration
        self.gmax = gmax
        self.dt = time_step
        
        # Calculate effective diffusion time
        self.t_eff = T_duration / (4 * N_cycles)
        
        # Set total echo time if not provided
        if te is None:
            # Default: time_before + T + time_between + T + time_after
            # Use 0.5*T for spacing
            self.te = 4 * T_duration  # Simple symmetric structure
        else:
            self.te = te
            
        self.generate_waveform()

    def generate_waveform(self):
        """Generate the apodized cosine OGSE waveform"""
        self.t = np.arange(0, self.te + self.dt, self.dt)
        self.wave = np.zeros_like(self.t)
        
        # Simplified approach for apodized cosine OGSE:
        # Structure: 1/4 sine + N cosine cycles + 1/4 sine
        # If apodization frequency = 2 * cosine frequency, then:
        # Total time T = 2*(1/4 cycle at 2f) + N cycles at f = 0.25/f + N/f = (0.25 + N)/f
        # Therefore: f_cosine = (0.25 + N) / T
        
        f_cosine = (0.25 + self.N_cycles) / self.T_duration  # cycles per ms
        f_apod = 2 * f_cosine  # apodization frequency (2x)
        
        # Calculate durations
        apod_duration = 0.25 / f_apod  # Duration of 1/4 cycle at apodization frequency
        cosine_duration = self.N_cycles / f_cosine  # Duration of N cycles at cosine frequency
        
        # Verify total duration
        total_calc = 2 * apod_duration + cosine_duration
        if abs(total_calc - self.T_duration) > 1e-6:
            print(f"Note: Slight duration difference: {total_calc:.6f} vs {self.T_duration:.6f}")
        
        # Generate encoding waveform (positive)
        t_start_encode = self.te/4 - self.T_duration/2  # Center in first quarter of TE
        
        self._generate_single_waveform(t_start_encode, 1.0, apod_duration, cosine_duration)
        
        # Generate decoding waveform (negative) 
        t_start_decode = 3*self.te/4 - self.T_duration/2  # Center in third quarter of TE
        
        self._generate_single_waveform(t_start_decode, -1.0, apod_duration, cosine_duration)
    
    def _generate_single_waveform(self, t_start, polarity, apod_duration, cosine_duration):
        """Generate a single apodized cosine waveform"""
        
        for i, t in enumerate(self.t):
            t_rel = t - t_start  # Time relative to waveform start
            
            if 0 <= t_rel < apod_duration:
                # First 1/4 sine (ramp up from 0 to 1)
                # sin goes from 0 to 1 over π/2 (quarter cycle)
                phase = (t_rel / apod_duration) * (np.pi / 2)
                self.wave[i] = polarity * self.gmax * np.sin(phase)
                
            elif apod_duration <= t_rel < apod_duration + cosine_duration:
                # N cosine cycles, starting and ending at +1
                # To ensure we end at +1, we need exactly N complete cycles
                t_cosine = t_rel - apod_duration
                progress = t_cosine / cosine_duration  # 0 to 1
                phase = 2 * np.pi * self.N_cycles * progress  # Exactly N cycles
                self.wave[i] = polarity * self.gmax * np.cos(phase)
                
            elif apod_duration + cosine_duration <= t_rel < 2*apod_duration + cosine_duration:
                # Final 1/4 sine (ramp down from 1 to 0)
                # cos goes from 1 to 0 over π/2 (quarter cycle)
                t_final = t_rel - (apod_duration + cosine_duration)
                phase = (t_final / apod_duration) * (np.pi / 2)
                self.wave[i] = polarity * self.gmax * np.cos(phase)
    
    def get_effective_diffusion_time(self):
        """Returns the effective diffusion time in ms"""
        return self.t_eff
    
    def get_waveform_info(self):
        """Returns information about the waveform parameters"""
        # Use the simplified frequency calculation
        f_cosine = (0.25 + self.N_cycles) / self.T_duration  # cycles per ms
        f_apod = 2 * f_cosine  # apodization frequency
        
        apod_duration = 0.25 / f_apod
        cosine_duration = self.N_cycles / f_cosine
        
        info = {
            'N_cycles': self.N_cycles,
            'T_duration_ms': self.T_duration,
            'effective_diffusion_time_ms': self.t_eff,
            'total_echo_time_ms': self.te,
            'apodization_duration_ms': apod_duration,
            'cosine_duration_ms': cosine_duration,
            'cosine_frequency_Hz': f_cosine * 1000,
            'apodization_frequency_Hz': f_apod * 1000,
            'frequency_ratio': f_apod / f_cosine
        }
        return info