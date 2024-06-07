#################################
# Retinal Waves Generation
# This script generates retinal waves propagating with directional bias based on the model proposed by Teh et al. (2023).
# Reference: Teh, Kai Lun, et al.
# "Retinal waves align the concentric orientation map in mouse superior colliculus to the center of vision."
# Science Advances 9.19 (2023): eadf4240.
#################################


import math
import cmath
import os.path
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--NrWaves", help="Specify how many waves you want to generate", type=int, default=10)
parser.add_argument("-d", "--directional", help="Specify if you want to generate directional waves",
                    action='store_true',  default=False)
parser.add_argument("-u", "--undirectional", help="Specify if you want to generate UNdirectional waves",
                    action='store_true',  default=False)
parser.add_argument("-p", "--plot",
                    help="Specify if you want to plot the wave initiation positions and propagation directions",
                    action='store_true', default=False)
args = parser.parse_args()

# convention: positions denoted as tuple always have y at the first position and x at the second (like numpy array)

# constants
grid_size = 64  # 40
N_v = 1
tau_IEI = 60  # 30s with 0.5s per frame
q = 2.4
T_wave = 1
r = 0.8
N = 8
tau_ON = 0
tau_OFF = 2  # 1s with 0.5s per frame
b = 0.35
nose_position = (0.5 * grid_size, 0.125 * grid_size)


class ActivityMap:
    def __init__(self, a_init, rho_betas, past_activation_probs, A_dict):
        self.a_init = a_init
        self.rho_betas = rho_betas
        self.frame_map = np.zeros((grid_size, grid_size))
        self.frame_map[a_init[0], a_init[1]] = 1
        self.past_frames = {0: self.frame_map}
        self.new_activated_cells = [(a_init[0], a_init[1])]
        self.activated_in_frames = {}
        self.activation_probs = {}
        self.past_activation_probs = past_activation_probs
        self.A_dict = A_dict

    def update_activation_probs(self, new_frame_nr):
        self.activation_probs = {}
        for y, x in self.new_activated_cells:
            for coords, direction in get_neighbors((y, x)):
                if coords[0] >= grid_size or coords[1] >= grid_size or coords[0] < 0 or coords[1] < 0:
                    continue
                if (coords[0], coords[1]) not in self.activation_probs.keys():
                    self.activation_probs[(coords[0], coords[1])] = []
                self.activation_probs[(coords[0], coords[1])].append(self.rho_betas[direction])
        self.past_activation_probs[new_frame_nr] = self.activation_probs

    def helper_propagate(self, new_frame_nr, new_frame):
        self.new_activated_cells = []
        for y_pixel in range(grid_size):
            for x_pixel in range(grid_size):
                new_frame[y_pixel, x_pixel] = self.V_Pre_P((y_pixel, x_pixel), new_frame_nr)
                if new_frame[y_pixel, x_pixel] > 0:  # cell is activated
                    if self.past_frames[new_frame_nr - 1][y_pixel, x_pixel] == 0:  # cell is new activated
                        self.new_activated_cells.append((y_pixel, x_pixel))
                    if (y_pixel, x_pixel) not in self.activated_in_frames.keys():
                        self.activated_in_frames[(y_pixel, x_pixel)] = []
                    self.activated_in_frames[(y_pixel, x_pixel)].append(new_frame_nr)

    # equation 12
    def V_Pre_P(self, alpha, frame):
        sum_activity = 0
        for t in range(0, (frame + 1)):
            if (alpha[0], alpha[1]) in self.past_activation_probs[t].keys():
                rho_alpha_list = self.past_activation_probs[t][(alpha[0], alpha[1])]
            else:
                rho_alpha_list = [0.0]

            term_a = self.A(alpha, t, rho_alpha_list)
            term_kp = self.K_P((frame - t))
            sum_activity += term_a * term_kp
        return sum_activity

    # equation 4
    def A(self, position, frame, rho_alpha_list):
        if (position, frame) in self.A_dict:
            return self.A_dict[(position, frame)]

        last_activated_in_frame = None
        try:
            last_activated = self.activated_in_frames[position]
            for la in last_activated:
                if la >= frame:
                    break
                else:
                    last_activated_in_frame = la
        except KeyError:
            pass

        if (self.nn_active(position, frame - 1) >= N_v) and (not last_activated_in_frame or
                                                             frame > (last_activated_in_frame + tau_IEI)):
            for rho_alpha in rho_alpha_list:
                activity = np.random.binomial(1, rho_alpha)
                if activity > 0.0:
                    self.A_dict[(position, frame)] = activity
                    return activity
            self.A_dict[(position, frame)] = activity
            return activity
        self.A_dict[(position, frame)] = 0
        return 0

    def nn_active(self, position, frame):
        if frame == -1:
            return 1
        active_count = 0
        for coords, direction in get_neighbors(position):
            if coords[0] >= grid_size or coords[1] >= grid_size or coords[0] < 0 or coords[1] < 0:
                continue
            if self.past_frames[frame][coords[0], coords[1]] > 0:
                active_count += 1
        return active_count

    # equation 9
    def K(self, frame):
        interval = range(0, T_wave + 1)
        return r * self.chi_t(frame, interval)  # gamma * xi(frame), gamma was 0

    def K_P(self, frame):
        return

    # equation 10
    def chi_t(self, frame, interval):
        if frame in interval:
            return 1
        return 0


class OnActivityMap(ActivityMap):
    def __init__(self, a_init, rho_betas, past_activation_probs, A_dict):
        super().__init__(a_init, rho_betas, past_activation_probs, A_dict)
        self.update_activation_probs(1)

    # equation 11
    def K_P(self, frame):
        # print(f'ON looking at frame {frame - tau_ON}')
        return self.K(frame - tau_ON)

    def propagate(self, new_frame_nr):
        # print(f'Propagate frame {new_frame_nr}')
        new_frame = np.zeros((grid_size, grid_size))
        self.helper_propagate(new_frame_nr, new_frame)
        self.past_frames[new_frame_nr] = new_frame
        self.frame_map = new_frame
        self.update_activation_probs(new_frame_nr + 1)


class OffActivityMap(ActivityMap):
    def __init__(self, a_init, rho_betas, past_activation_probs, A_dict):
        super().__init__(a_init, rho_betas, past_activation_probs, A_dict)
        self.past_frames = {0: self.frame_map}

    # equation 11
    def K_P(self, frame):
        return self.K(frame - tau_OFF)

    def propagate(self, new_frame_nr):
        new_frame = np.zeros((grid_size, grid_size))
        if new_frame_nr < (tau_OFF - tau_ON):
            new_frame[self.a_init[0], self.a_init[1]] = 1
        else:
            self.helper_propagate(new_frame_nr, new_frame)
        self.past_frames[new_frame_nr] = new_frame
        self.frame_map = new_frame


class Wave:
    rho_betas = {}

    def __init__(self, a_init, a_ai, sigma_prop, PICTURE_PATH, plots_per_wave):
        self.a_init = a_init
        self.a_ai = a_ai
        self.sigma_prop = sigma_prop
        self.PICTURE_PATH = PICTURE_PATH
        self.plots_per_wave = plots_per_wave
        self.calculate_initial_rho_betas()

        past_activation_probs = {0: {}}
        A_dict = {((a_init[0], a_init[1]), 0): 1.0}
        self.on_map = OnActivityMap(self.a_init, self.rho_betas, past_activation_probs, A_dict)
        self.off_map = OffActivityMap(self.a_init, self.rho_betas, past_activation_probs, A_dict)

        self.dataset = None

        self.frame_nr = 0

    def run(self):
        os.mkdir(f'{self.PICTURE_PATH}/images')
        if self.plots_per_wave:
            os.mkdir(f'{self.PICTURE_PATH}/binary_images')

        while not (np.all(self.on_map.frame_map == 0) and np.all(self.off_map.frame_map == 0)):
            if self.plots_per_wave:
                self.save_blue_and_red_frame()
            self.save_black_and_white_frame()
            if np.all(self.off_map.frame_map == 0):
                print('All zero in OFF map!')
            self.frame_nr += 1
            self.on_map.propagate(self.frame_nr)
            # if self.frame_nr > (tau_OFF - tau_ON):
            self.off_map.propagate(self.frame_nr)
        return self.dataset

    def calculate_initial_rho_betas(self):
        for coords, direction in get_neighbors((1, 1)):
            self.rho_betas[direction] = self.rho_beta_with_theta_wave((1, 1), coords)

    # equation 5
    def rho_beta_with_theta_wave(self, alpha, beta):
        # calculate Theta beta
        nu_x = beta[1] - alpha[1]
        nu_y = beta[0] - alpha[0]
        theta_beta = cmath.phase(complex(nu_x, nu_y))

        return rho_beta_with_angle(theta_beta, self.sigma_prop, self.a_init, self.a_ai, alpha)

    def save_frame_picture(self):
        sum_frame = self.on_map.frame_map + self.off_map.frame_map
        sns.heatmap(sum_frame, vmin=0.0, vmax=4.0, cmap='YlOrRd')
        plt.axis('off')
        plt.savefig(f'{self.PICTURE_PATH}/images/frame_{self.frame_nr}.png', bbox_inches='tight')
        plt.close()

    def save_black_and_white_frame(self):
        heat = np.zeros((grid_size, grid_size))
        cmap = LinearSegmentedColormap.from_list('Custom', ['white', 'black'], 2)
        for y_pixel in range(grid_size):
            for x_pixel in range(grid_size):
                if self.on_map.frame_map[y_pixel, x_pixel] > 0:
                    heat[y_pixel, x_pixel] = 1
                elif self.off_map.frame_map[y_pixel, x_pixel] > 0:
                    heat[y_pixel, x_pixel] = 1
                else:
                    heat[y_pixel, x_pixel] = 0

        if self.dataset is None:
            self.dataset = heat
        else:
            self.dataset = np.dstack((self.dataset, heat))

        sns.heatmap(heat, vmin=0.0, vmax=1.0, cmap=cmap, cbar=False, square=True)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.savefig(f'{self.PICTURE_PATH}/binary_images/frame_binary_{self.frame_nr}.png', bbox_inches='tight')
        plt.close()

    def save_blue_and_red_frame(self):
        heat = np.zeros((grid_size, grid_size))
        cmap = LinearSegmentedColormap.from_list('Custom', ['white', 'blue', 'red', 'green'], 4)
        for y_pixel in range(grid_size):
            for x_pixel in range(grid_size):
                if self.on_map.frame_map[y_pixel, x_pixel] > 0:
                    if self.off_map.frame_map[y_pixel, x_pixel] > 0:
                        heat[y_pixel, x_pixel] = 3
                    else:
                        heat[y_pixel, x_pixel] = 2  # red, ON
                elif self.off_map.frame_map[y_pixel, x_pixel] > 0:
                    heat[y_pixel, x_pixel] = 1  # blue, OFF
                else:
                    heat[y_pixel, x_pixel] = 0
        ax = sns.heatmap(heat, vmin=0.0, vmax=3.0, cmap=cmap)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1.1, 1.9, 2.6])
        colorbar.set_ticklabels(['OFF', 'ON', 'both'])
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        plt.savefig(f'{self.PICTURE_PATH}/frame_{self.frame_nr}.png', bbox_inches='tight')
        plt.close()


def get_neighbors(a):
    return [((a[0] - 1, a[1] - 1), 'top_left'), ((a[0] - 1, a[1]), 'top'),
            ((a[0] - 1, a[1] + 1), 'top_right'), ((a[0] + 1, a[1] - 1), 'bottom_left'),
            ((a[0] + 1, a[1]), 'bottom'), ((a[0] + 1, a[1] + 1), 'bottom_right'),
            ((a[0], a[1] - 1), 'left'), ((a[0], a[1] + 1), 'right')]


def get_optimized_sigma_prop():
    theta = torch.tensor([((2 * k + 1) * math.pi / N) for k in range(N)])
    std_deviation = torch.tensor([1.0])
    std_deviation.requires_grad = True
    sq_errori = []

    for i in range(10):
        # forward pass
        g = torch.exp(- (theta ** 2) / (2 * (std_deviation ** 2)))
        norm_coeff = 1 / (std_deviation * math.sqrt(2 * math.pi))
        g_norm = g * norm_coeff
        g_sum = g_norm.sum()  # q

        p_prop = torch.div(g_norm, g_sum)
        real_part = p_prop * theta.cos()
        imaginary_part = p_prop * theta.sin()
        positions = torch.cat([real_part, imaginary_part]).view(N, 2)

        B = positions.sum().abs()  # local propagation bias (strength of the averaged preferred direction)
        sq_error = (b - B) ** 2

        # backward pass
        std_deviation.grad = None  # gradient to zero
        sq_error.backward()

        # update
        lr = 0.1
        std_deviation.data += -lr * std_deviation.grad
        # track stats
        sq_errori.append(sq_error.item())

    # tested with values 2.0718722343444824 and 1.2, the latter brought especially good results
    return std_deviation.item()


def get_asymm_inhibition_position():
    center = [nose_position[0], nose_position[1]]  # [250, 1000] um and 50um per pixel
    spread = 1  # equivalent to 50um
    cov = [[spread, 0], [0, spread]]
    y, x = np.random.default_rng().multivariate_normal(center, cov).T
    return int(y), int(x)


def get_initiaion_position():
    x = np.random.randint(low=0, high=grid_size - 1)
    y = np.random.randint(low=0, high=grid_size - 1)
    return y, x


# equation 5.1
def rho_beta_with_angle(theta_beta, sigma_prop, a_init, a_ai, alpha):
    # calculate Theta wave
    mu_x = a_init[1] - a_ai[1]
    mu_y = a_init[0] - a_ai[0]
    theta_wave = cmath.phase(complex(mu_x, mu_y))
    theta_wave_prime = ((2 * math.pi) / N) * math.floor((N * theta_wave) / (2 * math.pi) + 0.5)

    return q * p_prop(theta_beta - theta_wave_prime, sigma_prop, alpha)


# equation 6; 1-D Gaussian function
def g(theta, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(math.pow(theta, 2)) / (2 * math.pow(sigma, 2)))


# equation 5.1
def p_prop(theta, sigma, alpha):
    sum_gamma_g = 0
    for gamma, gamma_pos in get_neighbors(alpha):
        v_x = gamma[1] - alpha[1]
        v_y = gamma[0] - alpha[0]
        theta_gamma = cmath.phase(complex(v_x, v_y))
        sum_gamma_g += g(theta_gamma, sigma)
    return g(theta, sigma) / sum_gamma_g


def theta_beta_plot(a_init, a_ai, sigma_prop, PICTURE_PATH):
    np.random.seed(42)
    nr_angels = 20
    angles = np.linspace(-np.pi, np.pi, nr_angels, endpoint=False)
    probabilities = []
    for angle in angles:
        probabilities.append(rho_beta_with_angle(angle, sigma_prop, a_init, a_ai, (1, 1)))

    ax = plt.subplot(projection='polar')
    ax.bar([(-1) * a for a in angles], probabilities, width=2 * np.pi / nr_angels, bottom=0.0, alpha=0.5)
    plt.ylim(0, 1)
    plt.savefig(f'{PICTURE_PATH}/direction_probabilities.png', bbox_inches='tight')
    plt.close()


def initialization_plot(a_init, a_ai, PICTURE_PATH):
    # Nose and inition positions
    fig = sns.heatmap(np.zeros((grid_size, grid_size)), vmin=0.0, vmax=4.0, cmap='binary', cbar=False, square=True)
    plt.scatter(x=nose_position[1], y=nose_position[0], label='Nose', marker='x')
    plt.annotate('Nose', (nose_position[1] + 1, nose_position[0] + 1), fontsize=12)
    plt.scatter(x=a_ai[1], y=a_ai[0], label='Inhibition', marker='x')
    plt.annotate('Inhibition', (a_ai[1] + 1, a_ai[0]), fontsize=12)
    plt.scatter(x=a_init[1], y=a_init[0], label='Init', marker='x')
    plt.annotate('Init', (a_init[1] + 1, a_init[0]), fontsize=12)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for _, spine in fig.spines.items():
        spine.set_visible(True)
    plt.savefig(f'{PICTURE_PATH}/e_init_positions.png', bbox_inches='tight')
    plt.close()


def main():
    if args.directional and args.undirectional:
        raise Exception('You cannot specify both -d and -u')
    if not args.directional and not args.undirectional:
        raise Exception('You need to specify either -d or -u')

    DIRECTIONAL = args.directional
    nr_waves = args.NrWaves
    plots_per_wave = args.plot

    directionality_text = 'Directional' if DIRECTIONAL else 'Undirectional'

    i = 0
    while (os.path.exists(f'data/generated_waves/Trial{i}_Directional')
           or os.path.exists(f'data/generated_waves/Trial{i}_Undirectional')):
        i += 1
    os.makedirs(f'data/generated_waves/Trial{i}_{directionality_text}/numpy_waves')
    TRIAL_PATH = f'data/generated_waves/Trial{i}_{directionality_text}'

    print(f'Creating {nr_waves} {directionality_text} waves...')

    theta_wave_dict = {}
    theta_wave_prime_dict = {}
    wave_dataset = None
    for wave_counter in tqdm(range(nr_waves)):
        PICTURE_PATH = f'{TRIAL_PATH}/W{wave_counter}'
        os.makedirs(PICTURE_PATH)

        a_init = get_initiaion_position()

        if DIRECTIONAL:
            a_ai = get_asymm_inhibition_position()
        else:
            a_ai = get_initiaion_position()

        sigma_prop = get_optimized_sigma_prop()
        if plots_per_wave:
            theta_beta_plot(a_init, a_ai, sigma_prop, PICTURE_PATH)
            initialization_plot(a_init, a_ai, PICTURE_PATH)

        mu_x = a_init[1] - a_ai[1]
        mu_y = a_init[0] - a_ai[0]
        theta_wave = cmath.phase(complex(mu_x, mu_y))
        theta_wave_prime = ((2 * math.pi) / N) * math.floor((N * theta_wave) / (2 * math.pi) + 0.5)
        if theta_wave not in theta_wave_dict.keys():
            theta_wave_dict[theta_wave] = 0
        theta_wave_dict[theta_wave] += 1
        if theta_wave_prime not in theta_wave_prime_dict.keys():
            theta_wave_prime_dict[theta_wave_prime] = 0
        theta_wave_prime_dict[theta_wave_prime] += 1

        wave = Wave(a_init, a_ai, sigma_prop, PICTURE_PATH, plots_per_wave)
        wave_as_numpy = wave.run()
        np.save(f'{TRIAL_PATH}/numpy_waves/wave{wave_counter}.npy', wave_as_numpy)
        if wave_dataset is None:
            wave_dataset = wave_as_numpy
        else:
            wave_dataset = np.dstack((wave_dataset, wave_as_numpy))

    np.save(f'{TRIAL_PATH}/trial{i}.npy', wave_dataset)

    # calculate wave directional bias
    for angles, occurences, name in zip([theta_wave_dict.keys(), theta_wave_prime_dict.keys()],
                                        [theta_wave_dict.values(), theta_wave_prime_dict.values()],
                                        ['theta_wave', 'theta_wave_prime']):
        heights = [(o / nr_waves) for o in occurences]
        ax = plt.subplot(projection='polar')
        ax.bar([(-1) * a for a in angles], heights, width=np.pi / N, bottom=0.0, alpha=0.5)
        plt.ylim(0, 1)
        plt.savefig(f'{TRIAL_PATH}/wave_bias_{name}.png', bbox_inches='tight')
        plt.close()

    with open(f'{TRIAL_PATH}/theta_waves.pickle', 'wb') as file:
        pickle.dump(theta_wave_dict, file)

    # Calculate Wave propagation direction bias B^wave (equation 26)
    # function_values = []
    # normalized_freq = [(o/nr_waves) for o in theta_wave_dict.values()]
    # for nr, angle in enumerate(theta_wave_dict.keys()):
    #     function_values.append(normalized_freq[nr]*math.exp(1j * angle))


if __name__ == "__main__":
    main()
