import numpy as np
from scipy.optimize import linprog

gamma = 267.513; # rad/ms/mT 
# gamma = 2.675*10^8 rad/s/T = 2.675*10^8 * 1e-6 rad/ms/mT = 267.5 rad/ms/mT

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
    '''deprecated'''
    # def calculate_bvalue(self,gmax=None):
    #     ''' 
    #     calculate the b-value of gwave in units of ms/um^2
    #     gmax is in mT/m; wave amplitude of 1 corresponds to 1 mT/m.
    #     '''
    #     if gmax is None:
    #         gmax = self.gmax
    #     gwave_um = self.gwave / 1e6  # mT/m -> mT/um (1 mT/m = 1e-6 mT/um)
    #     return np.sum(np.cumsum(gmax*gamma*gwave_um)**2)*self.dt**3; # ms/um^2

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
            gwave_um = self.wave / 1e6  # mT/m -> mT/um 
            # discrete form of \int_0^T [ dt * ( \int_0^t [gamma * g(t') dt'] )^2 ]
            # gamma = 267.513; # rad/ms/mT 
            # gwave_um in mT/um 
            return np.sum(np.cumsum(gamma*gwave_um)**2)*self.dt**3; # ms/um^2 -- 1 ms/um^2 = 1000 s/mm^2
    '''deprecated'''
    # def set_bvalues(self,b):
    #     b0 = self.calculate_bvalue(1)
    #     self.b = b
    #     self.gmax = np.sqrt(b/b0)
    '''deprecated'''
    # def set_gmax(self,gmax):
    #     self.gmax = gmax
    #     self.b = self.calculate_bvalue(gmax)

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
        # Use a TE-anchored timeline: starts at 0 and includes TE.
        self.t = np.arange(0, self.te + self.dt, self.dt)
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
    
    Structure: half-sine lobe + cosine segment + half-sine lobe,
    where each half-sine lobe is at 2x the cosine frequency and has shape 0->1->0.
    The cosine segment starts at zero with a negative lobe and ends at zero
    on a negative lobe, matching the pattern used in Does et al. Appendix A.
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
        
        # Apodized cosine OGSE with half-sine endcaps at 2f:
        # - each half-sine has duration 1/(4f)
        # - middle cosine segment has duration (N - 0.5)/f so it starts and
        #   ends at zero on negative lobes: 0->-1->0 ... ->-1->0
        # Total: T = 2*(1/(4f)) + (N - 0.5)/f = N/f  =>  f = N/T
        f_cosine = self.N_cycles / self.T_duration  # cycles per ms
        f_apod = 2 * f_cosine  # apodization frequency (2x)
        
        # Calculate durations
        apod_duration = 0.5 / f_apod  # Duration of 1/2 cycle at apodization frequency
        cosine_duration = (self.N_cycles - 0.5) / f_cosine  # Duration of negative-start/negative-end cosine segment
        
        # Verify total duration
        total_calc = 2 * apod_duration + cosine_duration
        if abs(total_calc - self.T_duration) > 1e-6:
            print(f"Note: Slight duration difference: {total_calc:.6f} vs {self.T_duration:.6f}")
        
        # Generate encoding waveform (positive)
        t_start_encode = self.te/4 - self.T_duration/2  # Center in first quarter of TE
        
        self._generate_single_waveform(t_start_encode, 1.0, apod_duration, cosine_duration, f_cosine)
        
        # Generate decoding waveform (negative) 
        t_start_decode = 3*self.te/4 - self.T_duration/2  # Center in third quarter of TE
        
        self._generate_single_waveform(t_start_decode, -1.0, apod_duration, cosine_duration, f_cosine)
    
    def _generate_single_waveform(self, t_start, polarity, apod_duration, cosine_duration, f_cosine):
        """Generate a single apodized cosine waveform"""
        
        for i, t in enumerate(self.t):
            t_rel = t - t_start  # Time relative to waveform start
            
            if 0 <= t_rel < apod_duration:
                # First half-sine endcap at 2f: 0 -> 1 -> 0.
                phase = (t_rel / apod_duration) * np.pi
                self.wave[i] = polarity * self.gmax * np.sin(phase)
                
            elif apod_duration <= t_rel < apod_duration + cosine_duration:
                # N cosine-frequency cycles in between endcaps, starting at zero
                # and beginning with a negative lobe, ending at zero on a
                # negative lobe: 0 -> -1 -> 0 ... -> -1 -> 0.
                t_cosine = t_rel - apod_duration
                phase = 2 * np.pi * (f_cosine * t_cosine)
                self.wave[i] = -polarity * self.gmax * np.sin(phase)
                
            elif apod_duration + cosine_duration <= t_rel < 2*apod_duration + cosine_duration:
                # Final half-sine endcap at 2f: 0 -> 1 -> 0.
                t_final = t_rel - (apod_duration + cosine_duration)
                phase = (t_final / apod_duration) * np.pi
                self.wave[i] = polarity * self.gmax * np.sin(phase)
    
    def get_effective_diffusion_time(self):
        """Returns the effective diffusion time in ms"""
        return self.t_eff
    
    def get_waveform_info(self):
        """Returns information about the waveform parameters"""
        # With negative-start/negative-end cosine segment, T = N/f.
        f_cosine = self.N_cycles / self.T_duration  # cycles per ms
        f_apod = 2 * f_cosine  # apodization frequency
        
        apod_duration = 0.5 / f_apod
        cosine_duration = (self.N_cycles - 0.5) / f_cosine
        
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


class CosineOGSEWaveform(DiffGradWaveform):
    '''
    Non-apodized cosine OGSE waveform with paired bipolar blocks.

    Each active block is a non-apodized cosine with very short linear edge
    ramps: 0->1 (quick), cosine core, then 1->0 (quick). The second block has
    opposite polarity.
    '''

    def __init__(self, N_cycles, T_duration, te=None, gmax=1.0, time_step=dt0):
        self.N_cycles = int(N_cycles)
        self.T_duration = float(T_duration)
        self.gmax = float(gmax)
        self.dt = float(time_step)

        if self.N_cycles <= 0:
            raise ValueError("N_cycles must be a positive integer.")
        if self.T_duration <= 0:
            raise ValueError("T_duration must be positive.")

        self.t_eff = self.T_duration / (4.0 * self.N_cycles)

        if te is None:
            self.te = 4.0 * self.T_duration
        else:
            self.te = float(te)

        self.generate_waveform()

    def generate_waveform(self):
        self.t = np.arange(0.0, self.te + self.dt, self.dt)
        self.wave = np.zeros_like(self.t)

        f_cosine = self.N_cycles / self.T_duration  # cycles/ms
        t_start_encode = self.te / 4.0 - self.T_duration / 2.0
        t_start_decode = 3.0 * self.te / 4.0 - self.T_duration / 2.0

        self._generate_single_waveform(t_start_encode, 1.0, self.T_duration, f_cosine)
        self._generate_single_waveform(t_start_decode, -1.0, self.T_duration, f_cosine)

    def _generate_single_waveform(self, t_start, polarity, cosine_duration, f_cosine):
        # Use a near-instant ramp equal to one simulation step when possible.
        ramp_duration = min(self.dt, max(0.0, 0.25 * cosine_duration))
        core_duration = max(cosine_duration - 2.0 * ramp_duration, 0.0)
        if core_duration > 0.0:
            f_core = self.N_cycles / core_duration
        else:
            f_core = f_cosine

        for i, t in enumerate(self.t):
            t_rel = t - t_start

            if 0.0 <= t_rel < ramp_duration:
                # Quick linear ramp from 0 to block peak.
                amp = (t_rel / max(ramp_duration, 1e-15))
                self.wave[i] = polarity * self.gmax * amp

            elif ramp_duration <= t_rel < (ramp_duration + core_duration):
                # Cosine core starting near +1 and oscillating at the OGSE frequency.
                t_core = t_rel - ramp_duration
                phase = 2.0 * np.pi * (f_core * t_core)
                self.wave[i] = polarity * self.gmax * np.cos(phase)

            elif (ramp_duration + core_duration) <= t_rel < (2.0 * ramp_duration + core_duration):
                # Quick linear ramp from block peak back to zero.
                tau = (t_rel - (ramp_duration + core_duration)) / max(ramp_duration, 1e-15)
                self.wave[i] = polarity * self.gmax * (1.0 - tau)

    def get_effective_diffusion_time(self):
        return self.t_eff

    def get_waveform_info(self):
        f_cosine = self.N_cycles / self.T_duration
        return {
            'N_cycles': self.N_cycles,
            'T_duration_ms': self.T_duration,
            'effective_diffusion_time_ms': self.t_eff,
            'total_echo_time_ms': self.te,
            'cosine_duration_ms': self.T_duration,
            'cosine_frequency_Hz': f_cosine * 1000.0,
        }


class TrapezoidalCosineOGSEWaveform(DiffGradWaveform):
    '''
    Xu-style trapezoidal cosine OGSE waveform with paired bipolar blocks.

    One active block follows a piecewise trapezoid-cosine structure with:
    - rise/fall time tr (= trise)
    - edge plateau time tp (first and last positive peaks)
    - interior timing parameter t3 = tp + tr/2
    - interior OGSE plateaus with duration 2*t3
    - 2N+1 alternating plateaus (+, -, +, ..., +)

    Two identical blocks are generated with opposite polarity. The
    separation_duration is interpreted as start-to-start spacing between
    the positive and negative blocks.

    Effective diffusion time convention: t_eff = T/(4N), where T is the
    active duration of one block.
    '''

    def __init__(
        self,
        N_cycles,
        T_duration,
        separation_duration,
        trise=None,
        tp=None,
        te=None,
        gmax=1.0,
        time_step=dt0,
    ):
        self.N_cycles = int(N_cycles)
        self.T_duration = float(T_duration)
        self.separation_duration = float(separation_duration)
        self.gmax = float(gmax)
        self.dt = float(time_step)

        if self.N_cycles <= 0:
            raise ValueError("N_cycles must be a positive integer.")
        if self.T_duration <= 0:
            raise ValueError("T_duration must be positive.")
        if self.separation_duration < 0:
            raise ValueError("separation_duration must be non-negative.")

        if trise is None and tp is None:
            self.trise = 1.0
        elif trise is None:
            self.trise = np.nan
        else:
            self.trise = float(trise)
        self.tp = None if tp is None else float(tp)

        if self.tp is None:
            if not np.isfinite(self.trise) or self.trise <= 0:
                raise ValueError("When tp is not provided, trise must be positive.")
            # For the implemented block layout (edge tp, interior 2*t3,
            # t3 = tp + tr/2), enforce exact on-time T:
            # T = 4*N*tp + (6*N + 1)*tr.
            self.tp = (self.T_duration - (6.0 * self.N_cycles + 1.0) * self.trise) / (4.0 * self.N_cycles)
        else:
            if self.tp <= 0:
                raise ValueError("tp must be positive.")
            if not np.isfinite(self.trise):
                self.trise = (self.T_duration - 4.0 * self.N_cycles * self.tp) / (6.0 * self.N_cycles + 1.0)

        if self.trise <= 0:
            raise ValueError("Computed trise is non-positive. Adjust T_duration/trise/tp.")
        if self.tp <= 0:
            raise ValueError("Computed tp is non-positive. Adjust T_duration/trise/tp.")

        self.t3 = self.tp + 0.5 * self.trise
        self.single_lobe_duration = 4.0 * self.N_cycles * self.tp + (6.0 * self.N_cycles + 1.0) * self.trise
        if not np.isclose(self.single_lobe_duration, self.T_duration, rtol=0.0, atol=max(1e-9, self.dt)):
            raise ValueError(
                "Inconsistent trapezoid timing: computed lobe on-time does not match T_duration. "
                f"Computed={self.single_lobe_duration:.6f} ms, T_duration={self.T_duration:.6f} ms."
            )

        self.t_eff = self.T_duration / (4.0 * self.N_cycles)

        self.full_gradient_duration = self.separation_duration + self.T_duration

        if te is None:
            # User-requested convention: TE = full_gradient_duration + T_duration.
            self.te = self.full_gradient_duration + self.T_duration
        else:
            self.te = float(te)

        min_te = self.full_gradient_duration
        if self.te < min_te:
            raise ValueError(
                f"te={self.te} is too short; minimum is full_gradient_duration = {min_te}."
            )

        self.generate_waveform()

    def generate_waveform(self):
        self.t = np.arange(0.0, self.te + self.dt, self.dt)
        self.wave = np.zeros_like(self.t)

        active_total = self.full_gradient_duration
        t_margin = 0.5 * (self.te - active_total)

        self.block1_start = t_margin
        self.block2_start = self.block1_start + self.separation_duration

        self._add_trapezoid_cosine_block(self.block1_start, polarity=1.0)
        self._add_trapezoid_cosine_block(self.block2_start, polarity=-1.0)

    def _add_trapezoid_cosine_block(self, t_start, polarity):
        # Build explicit Xu-style sequence: ramp, plateaus, and 2*tr transitions.
        signs = np.array([1.0 if (k % 2 == 0) else -1.0 for k in range(2 * self.N_cycles + 1)], dtype=float)
        plateau_durations = np.full(signs.size, 2.0 * self.t3, dtype=float)
        plateau_durations[0] = self.tp
        plateau_durations[-1] = self.tp

        cursor = t_start

        # Initial ramp: 0 -> +1
        self._add_linear_segment(cursor, self.trise, 0.0, signs[0] * polarity)
        cursor += self.trise

        # Plateau / transition chain.
        for idx in range(signs.size):
            amp = signs[idx] * polarity
            self._add_constant_segment(cursor, plateau_durations[idx], amp)
            cursor += plateau_durations[idx]

            if idx < (signs.size - 1):
                # One sign flip realized as two ramps of duration tr each => 2*tr total.
                next_amp = signs[idx + 1] * polarity
                self._add_linear_segment(cursor, 2.0 * self.trise, amp, next_amp)
                cursor += 2.0 * self.trise

        # Final ramp: +1 -> 0
        self._add_linear_segment(cursor, self.trise, signs[-1] * polarity, 0.0)

    def _add_constant_segment(self, t_start, duration, value):
        if duration <= 0:
            return
        mask = (self.t >= t_start) & (self.t < (t_start + duration))
        self.wave[mask] = self.gmax * value

    def _add_linear_segment(self, t_start, duration, v0, v1):
        if duration <= 0:
            return
        mask = (self.t >= t_start) & (self.t < (t_start + duration))
        if not np.any(mask):
            return
        tau = (self.t[mask] - t_start) / duration
        self.wave[mask] = self.gmax * (v0 + (v1 - v0) * tau)

    def get_effective_diffusion_time(self):
        return self.t_eff

    def get_waveform_info(self):
        return {
            'N_cycles': self.N_cycles,
            'T_duration_ms': self.T_duration,
            'single_lobe_duration_ms': self.single_lobe_duration,
            'separation_duration_ms': self.separation_duration,
            'full_gradient_duration_ms': self.full_gradient_duration,
            'trise_ms': self.trise,
            'tp_ms': self.tp,
            't3_ms': self.t3,
            'ogse_plateau_ms': 2.0 * self.t3,
            'effective_diffusion_time_ms': self.t_eff,
            'total_echo_time_ms': self.te,
            'cosine_frequency_Hz': (self.N_cycles / self.T_duration) * 1000.0,
            'block1_start_ms': self.block1_start,
            'block2_start_ms': self.block2_start,
        }