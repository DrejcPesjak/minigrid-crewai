import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel


from collections import deque

VAL_N = 512  # how many completed episodes to aggregate before logging SR_512
val_returns = deque(maxlen=VAL_N)          # store last 512 episode returns
val_frames_per_ep = deque(maxlen=VAL_N)    # optional, for avg len
total_episodes_seen = 0                    # cumulative (for reporting)
total_successes_seen = 0                   # cumulative successes (return>0)
consecutive_99_count = 0                   # count consecutive evaluations with SR >= 0.99

def _success_from_return(r):
    # works for sparse terminal reward (success => r>0)
    # change if your env uses a different success signal
    return float(r > 0)


# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    # obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    # if "vocab" in status:
    #     preprocess_obss.vocab.load_vocab(status["vocab"])
    # txt_logger.info("Observations preprocessor loaded")

    # --- Load observations preprocessor (FiLM-friendly) ---
    from utils.form import ObssPreprocessor

    # We want symbolic images + tokenized instructions.
    # ObssPreprocessor will expose .obs_space with ints (e.g., image:147, instr:100)
    preprocess_obss = ObssPreprocessor(model_name)   # pass model_name so vocab can be saved under the run dir
    obs_space = preprocess_obss.obs_space

    txt_logger.info("Observations preprocessor loaded")
    txt_logger.info(f"obs_space {obs_space}\n")

    # Load model

    # acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    acmodel = ACModel(obs_space, envs[0].action_space, use_instr=args.text, use_memory=args.mem)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))


    # # --- TensorBoard: model graph export ---
    # acmodel.eval()  # recommended for tracing

    # # Build a minimal dummy obs that matches your model's expected shapes
    # n, m, c = obs_space["image"]  # (H, W, C) from BabyAI
    # dummy_img = torch.zeros(1, n, m, c, dtype=torch.float32, device=device)

    # if args.text:
    #     # one-token dummy text; dtype must be long for nn.Embedding
    #     dummy_text = torch.zeros(1, 1, dtype=torch.long, device=device)
    # else:
    #     dummy_text = None

    # # ACModel expects attributes .image and (optionally) .text on obs
    # class _Obs:
    #     def __init__(self, image, text=None):
    #         self.image = image
    #         if text is not None:
    #             self.text = text

    # dummy_obs = _Obs(dummy_img, dummy_text)

    # # Forward always takes `memory`; pass zeros (ignored if use_memory=False)
    # dummy_mem = torch.zeros(1, acmodel.memory_size, dtype=torch.float32, device=device)

    # try:
    #     tb_writer.add_graph(acmodel, (dummy_obs, dummy_mem))
    #     txt_logger.info("TensorBoard model graph written.")
    # except Exception as e:
    #     txt_logger.info(f"TensorBoard graph export failed: {e}")
    # # --- end graph export ---

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}

        ep_rets = logs.get("return_per_episode", [])
        ep_frames = logs.get("num_frames_per_episode", [])

        # extend buffers (they may be different lengths if some updates have 0 completions)
        for r in ep_rets:
            val_returns.append(r)
        for f in ep_frames:
            val_frames_per_ep.append(f)

        # maintain counters for total episodes/successes (useful meta-stats)
        total_episodes_seen += len(ep_rets)
        total_successes_seen += sum(_success_from_return(r) for r in ep_rets)

        # when we have at least 512 completed episodes in the buffer, compute SR_512 once
        if len(val_returns) == VAL_N:
            sr_512 = sum(_success_from_return(r) for r in val_returns) / VAL_N
            avg_len_512 = (sum(val_frames_per_ep) / VAL_N) if len(val_frames_per_ep) == VAL_N else float('nan')

            # optional: compute an episodes-to-99% estimate at this time
            # episodes so far (approx) from frames:
            # you already track num_frames; prefer exact counter if you have it
            # Here we just report SR_512 cleanly.
            txt_logger.info(f"[VAL{VAL_N}] SR={sr_512:.4f}  avg_len={avg_len_512:.2f}  total_frames={num_frames}  total_eps={total_episodes_seen}")

            # tensorboard
            tb_writer.add_scalar(f"val/SR_{VAL_N}", sr_512, num_frames)
            if len(val_frames_per_ep) == VAL_N:
                tb_writer.add_scalar(f"val/avg_len_{VAL_N}", avg_len_512, num_frames)
            
            # Track consecutive evaluations with SR >= 0.99
            if sr_512 >= 0.99:
                consecutive_99_count += 1
                txt_logger.info(f"Success rate is greater than 99% ({consecutive_99_count}/5 consecutive)")
                if consecutive_99_count >= 5:
                    txt_logger.info("Success rate has been >= 99% for 5 consecutive evaluations. Stopping training.")
                    break
            else:
                consecutive_99_count = 0  # Reset counter if SR drops below 99%


        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            num_episodes = len(logs["return_per_episode"])
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            success_rate_per_episode = utils.synthesize_success_rate(logs["return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration", "episodes"]
            data = [update, num_frames, fps, duration, num_episodes]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["success_rate_" + key for key in success_rate_per_episode.keys()]
            data += success_rate_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | E {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | SR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)
            
            # if success_rate_per_episode['mean'].round(2) >= 0.99:
            #     txt_logger.info("Success rate is greater than 99%")
            #     break

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
