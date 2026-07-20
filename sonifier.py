import torch
from scipy.io import wavfile

import config
from data import pca_reduce, to_uniform
from masker import Masker


class Sonifier():

    def __init__(self, shape, fs=44100):
        self._INPUT_SHAPE = shape
        self._FS = fs
        self._NOTE_LENGTH = config.NOTE_LENGTH
        self._SAMPLES_PER_NOTE = int(self._FS * self._NOTE_LENGTH)

        self.config = {
            "sonification_type": "freq",
            "freq_lower": config.FREQ_LOWER,
            "freq_upper": config.FREQ_UPPER,
            "max_z": config.MAX_Z,

            "do_interpolate": False,
            "do_diff": False,
            "pca": None,

            "channel_names": config.CHANNEL_SPLIT,
            "gain": config.GAIN,

            "freq_map": None
        }

        self._masker =  Masker(
            shape=(
                int(shape[0] * shape[1] * self._SAMPLES_PER_NOTE), 
                shape[2]
            ),
            pass_size=(shape[1] * self._SAMPLES_PER_NOTE)
        )

    @staticmethod
    def _interpolate(x, scale_factor):
        """ 
        Interpolate a (time, voices)-tensor to shape (time * scale_factor, voices) 
        """
        T, V = x.shape
        x = x.permute(1, 0).view(1, V, T) # (1, V, T)

        x = torch.nn.functional.interpolate(
            x, 
            scale_factor=scale_factor, 
            mode='linear', 
            align_corners=False
        ) # (1, V, T * scale)

        x = x.squeeze().permute(1, 0) # (T * scale, V)
        return x
    
    def _to_freq(self, x):
        lower = self.config["freq_lower"]
        upper = self.config["freq_upper"]
        freq = x.abs() / (self.config["max_z"] * x.std()) * (upper-lower) + lower
        return freq

    def _generate_phase(self, frequencies):
        """
        Given a (timesteps, voices)-tensor, output a (time, voices)-tensor
        of phases where time = timesteps * fs
        """
        delta = 2 * torch.pi * frequencies / float(self._FS)
        initial_phase = torch.rand(frequencies.shape[1]) * 2 * torch.pi
        phase = initial_phase.unsqueeze(0) + torch.cumsum(delta, axis=0)  # start from random phase
        phase = torch.fmod(phase, 2 * torch.pi) # wrap phase to prevent overflow
        return phase

    def _mix(self, audio):
        """ 
        Sum the voice channels of a (channels, time, voices)-tensor, 
        normalize to int16 
        """
        mix = torch.sum(audio, dim=2)
        mix /= torch.max(torch.abs(mix))
        return (mix * 32767 * self.config["gain"]).to(torch.int16)

    def _get_diff_mask(self, states: torch.Tensor):
        """
        Generate a (time, voices) mask to weigh an audio tensor's amplitude
        depending on the interpolated derivative of time
        """
        T, V = states.shape

        diff = torch.abs(torch.diff(states, dim=0))

        diff = torch.cat((
            torch.zeros(1, V), 
            diff, 
            torch.zeros(1, V)
        ), dim=0) # padded first derivative

        diff = self.interpolate(diff, self._SAMPLES_PER_NOTE)
        diff = diff[self._SAMPLES_PER_NOTE // 2 : -self._SAMPLES_PER_NOTE // 2] # crop to fit
        diff = to_uniform(diff, 0, 1)
        return diff

    def _freq_son(self, states: torch.Tensor):
        """
        Sonify a (seq_length, layers, voices)-tensor
        """
        if list(states.shape) != list(self._INPUT_SHAPE):
            raise ValueError(f"input shape {list(states.shape)} does not match with standard shape {list(self._INPUT_SHAPE)}")
        S, L, V = self._INPUT_SHAPE
        states = states.reshape(S * L, V).float() # (time, voices)

        if self.config["pca"] is not None: 
            states = pca_reduce(states, q=self.config["pca"])

        states = self._to_freq(states)

        if self.config["do_interpolate"]:
            freq_samples = self._interpolate(states, self._SAMPLES_PER_NOTE)
        else:
            freq_samples = torch.repeat_interleave(states, self._SAMPLES_PER_NOTE, dim=0)

        phase = self._generate_phase(freq_samples)
        audio = torch.sin(phase)

        if self.config["do_diff"]:
            raise NotImplementedError("config 'do_diff' not yet implemented")
            diff_mask = self._get_diff_mask(states)
            audio *= diff_mask

        stereo = self._masker(audio, self.config["channel_names"])
        stereo = self._mix(stereo)

        stereo = stereo.permute(1, 0) # stupid but I have to do this

        return stereo

    def _gain_son(self, states: torch.Tensor, freq_map: torch.Tensor):

        S, L, V = self._INPUT_SHAPE
        states = states.reshape(S * L, V).float() # (time, voices)
        if self.config["do_abs"]:
            gains = torch.abs(gains)
        gains = self._to_freq(states, lower=0, upper=0.5)

        if len(freq_map.shape) == 1:

            if freq_map.shape[0] != gains.shape[1]:
                raise ValueError("1D freq_map must match voice dimension of gains")
            freq_map = freq_map.unsqueeze(0)
            freq_map = freq_map.repeat(S*L, 1)

        elif list(freq_map.shape) != list(gains.shape): 
            raise ValueError("2D freq_map must match shape of gains")
        
        freq_samples = torch.repeat_interleave(freq_map, self._SAMPLES_PER_NOTE, dim=0)

        if self.config["do_interpolate"]:
            gain_samples = self._interpolate(states, self._SAMPLES_PER_NOTE)
        else:
            gain_samples = torch.repeat_interleave(states, self._SAMPLES_PER_NOTE, dim=0)

        phase = self._generate_phase(freq_samples)
        audio = torch.sin(phase)
        audio *= gain_samples

        stereo = self._masker(audio, self.config["channel_names"])
        stereo = self._mix(stereo)

        stereo = stereo.permute(1, 0) # stupid but I have to do this

        return stereo

    def __call__(self, states: torch.Tensor, freq_map: torch.Tensor=None):

        if self.config["sonification_type"] == "freq":
            return self._freq_son(states)
        
        elif self.config["sonification_type"] == "gain":
            
            if self.config["freq_map"] == None:
                raise ValueError("config[\"freq_map\"] must be defined for gain sonification")
            
            return self._gain_son(states, self.config["freq_map"])
        
        else:
            raise ValueError(f"sonification type \"{self.config["sonification_type"]}\" not defined")
    

if __name__ == "__main__":
    states = torch.randn((3, 17, 2048))
    sonify = Sonifier(states.shape)
    wav = sonify(states).numpy()
    wavfile.write("stereo.wav", 44100, wav)