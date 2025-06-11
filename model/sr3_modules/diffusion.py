import math
import torch
from torch import device, nn, einsum
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from model.ddpm_modules.DPM_Solver import NoiseScheduleVP, model_wrapper, DPM_Solver


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_param_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
        params = [betas]
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        params = [betas]
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
        params = [betas]
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
        params = [betas]
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        params = [betas]
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
        params = [betas]
    elif schedule == "cosine":
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        f_t = timesteps / (1 + cosine_s) * math.pi / 2
        f_t = np.power(np.cos(f_t), 2)
        alphas_cumprod_t = f_t / f_t[0]
        alphas_cumprod = alphas_cumprod_t[1:]
        alphas_cumprod_prev = alphas_cumprod_t[:-1]
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        params = [betas, alphas_cumprod, alphas_cumprod_prev]
    else:
        raise NotImplementedError(schedule)
    return params


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        sample_type='DDPM',
        fast_steps='None',
        schedule_opt=None,
        perception=False
    ):
        super().__init__()
        self.perception = perception
        if self.perception:
            vgg = vgg16(pretrained=True)
            self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
            self.mse_loss = nn.MSELoss(reduction='sum')
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.sample_type = sample_type
        self.fast_steps = fast_steps
        self.conditional = conditional

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        self.to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        self.schedule = schedule_opt['schedule']
        params = make_param_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        params = params.detach().cpu().numpy() if isinstance(params, torch.Tensor) else params
        if self.schedule == 'linear':
            betas = params[0]
            alphas = 1. - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        elif self.schedule == 'cosine':
            betas = params[0]
            alphas = 1. - betas
            alphas_cumprod = params[1]
            alphas_cumprod_prev = params[2]

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', self.to_torch(betas))
        self.register_buffer('alphas_cumprod', self.to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             self.to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             self.to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             self.to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             self.to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             self.to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             self.to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev',
                             self.to_torch(np.sqrt(1 - alphas_cumprod_prev)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             self.to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', self.to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', self.to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', self.to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.noise_level_list = self.sqrt_alphas_cumprod_prev

    def set_ddim_noise_schedule(self, schedule_opt, device):
        self.to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        params = make_param_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        params = params.detach().cpu().numpy() if isinstance(params, torch.Tensor) else params
        betas = params[0]
        self.register_buffer('betas', self.to_torch(betas))
        skip = schedule_opt['n_timestep'] // self.fast_steps
        self.seq = range(0, schedule_opt['n_timestep'], skip)
        self.seq_next = [-1] + list(self.seq[:-1])
        if int(self.fast_steps) == 2000:
            alphas = 1. - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            self.noise_level_list = np.sqrt(np.append(1., alphas_cumprod))
        else:
            self.noise_level_list = np.loadtxt('/home/andyw/SR3/noise_level_' + str(self.fast_steps) + '.txt')

    def set_dpmsolver_noise_schedule(self, schedule_opt, device):
        self.to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        params = make_param_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        params = params.detach().cpu().numpy() if isinstance(params, torch.Tensor) else params
        betas = params[0]
        self.register_buffer('betas', self.to_torch(betas))
        if int(self.fast_steps) == 2000:
            alphas = 1. - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            self.noise_level_list = np.sqrt(np.append(1., alphas_cumprod))
        else:
            self.noise_level_list = np.loadtxt('/home/andyw/SR3/noise_level_' + str(self.fast_steps) + '.txt')

    def compute_alpha(self, beta, t):
        beta= torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        alpha_cumprod = (1-beta).cumprod(dim=0).index_select(0, t+1)
        return alpha_cumprod

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.noise_level_list[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            predict_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
            x_recon = self.predict_start_from_noise(x, t=t, noise=predict_noise)
        else:
            predict_noise = self.denoise_fn(x, noise_level)
            x_recon = self.predict_start_from_noise(x, t=t, noise=predict_noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, predict_noise, x_recon

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, predict_noise, predict_x0 = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x_next = model_mean + noise * (0.5 * model_log_variance).exp()
        return x_next

    def p_sample_ddim(self, x, condition_x, step, alpha_cumprod_t, alpha_cumprod_t_next):
        batch_size = x.shape[0]
        noise_level = torch.tensor(np.float32(self.noise_level_list[step])).to(x.device).repeat(batch_size, 1)
        predict_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        predict_x0 = (x - predict_noise * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
        c = (1 - alpha_cumprod_t_next).sqrt()
        x_next = alpha_cumprod_t_next.sqrt() * predict_x0 + c * predict_noise
        return x_next

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i, t in enumerate(tqdm(reversed(seq), desc='sampling loop time step', total=len(seq))):
                img = self.p_sample_ddpm(img, len(seq)-i-1)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            if self.sample_type == 'DDPM' or self.sample_type == 'DDPM-SRCNN':
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',total=self.num_timesteps):
                    img = self.p_sample_ddpm(img, i, condition_x=x)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
            elif self.sample_type == 'DDIM' or self.sample_type == 'DDIM-SRCNN':
                step = 0
                for i, j in tqdm(zip(reversed(self.seq), reversed(self.seq_next)), desc='sampling loop time step',total=len(self.seq_next)):
                    t = (torch.ones(1) * i).to(device)
                    next_t = (torch.ones(1) * j).to(device)
                    alpha_cumprod_t = self.compute_alpha(self.betas, t.long())
                    alpha_cumprod_t_next = self.compute_alpha(self.betas, next_t.long())
                    img = self.p_sample_ddim(img, x, step, alpha_cumprod_t, alpha_cumprod_t_next)
                    step += 1
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
            elif self.sample_type == 'DPMSolver' or self.sample_type == 'DPMSolver-SRCNN':
                print(self.fast_steps)
                noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
                dpm_solver = DPM_Solver(
                    self.denoise_fn,
                    noise_schedule,
                    algorithm_type='dpmsolver',
                    #correcting_x0_fn="dynamic_thresholding" # if use dpmsolver++, add this
                )
                ret_img = dpm_solver.sample(
                    img,
                    x,
                    self.noise_level_list,
                    steps=self.fast_steps,
                    order=3,
                    skip_type='time_uniform',
                    method='multistep',
                    lower_order_final=False,
                    denoise_to_zero=False,
                    solver_type='dpmsolver',
                    atol=0.00078,
                    rtol=0.05
                )


        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        if self.sample_type == 'DDPM' or self.sample_type =='DDIM':
            x_start = x_in['HR']
        if self.sample_type == 'DDPM-SRCNN' or self.sample_type =='DDIM-SRCNN':
            x_start = x_in['HR'] - x_in['SR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        noise_loss = self.loss_func(noise, x_recon)

        if self.perception:
            x_start_res_predict = self.predict_start_from_noise(x_noisy, t-1, x_recon)
            x_start_predict = x_start_res_predict + x_in['SR']
            x_start_real_rgb = x_in['HR'].repeat(1, 3, 1, 1)
            x_start_predict_rgb = x_start_predict.repeat(1, 3, 1, 1)
            perception_loss = 1e10 * self.mse_loss(self.loss_network(x_start_predict_rgb), self.loss_network(x_start_real_rgb))
            return noise_loss, perception_loss

        return noise_loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
