import mdp
import random
import mdp_utils as mu
import mdp_aprox as ma

# convert episode array to string (for debugging and output)
def episode_to_str(m, e, rounding = 2, add_utility = True):
    s = str(e[0]) + ";"
    for i in range(1, len(e) - 1, 3):
        # action, reward, state
        s += str(e[i]) + "," + str(round(e[i + 1], rounding)) + "," + str(e[i + 2]) + ";";
    if add_utility:
        s += " ({})".format(round(mdp.episode_utility(m, e), rounding))
    return s

def run_with_policy(m, policy, descr, cnt = 1000):
    print("\nrunning " + str(cnt) + " episode(s) with " + descr + ":")
    episodes = []
    for i in range(0, cnt):
        episodes.append(policy.create_episode(m.start_state()))
    qvals = ma.qvalues_monte_carlo(episodes, m.discount())
    e = episodes[0]
    s, a = e[0], e[1]
    u = qvals[(s, a)]
    print("average episode utility: " + str(u))
    return episodes

def compare_dictionaries(pd, pe):
    failed = True
    if len(pd) == len(pe):
        failed = False
        for k, v in pd.items():
            if pe[k] != v:
                print("failed for " + str(k) + " - " + str(v) + " vs. " + str(pe[k]))
                failed = True
    else:
        print("mismatch of number of elements")
    return failed


def multi_choise(probs):
    p, v = random.uniform(0.0, 1.0), 0.0
    for i in range(len(probs)):
        if v > p: return i
        v += probs[i]
    return len(probs) - 1

# returns random value from array (uniformly distributed)
def random_choice(arr):
    return arr[random.randint(0, len(arr) - 1)]

def run(m: mdp.mdp_t, p, msg, cnt = 10000):
    return mu.run_with_policy(m, p, msg, cnt)

def separator(msg):
    print("\n===========================================================")
    print(msg)

def dictprn(d):
    print('  ', end='')
    for k, v in sorted(d.items()):
        print(k, end=': ')
        print(v, end=', ')
    print("")
