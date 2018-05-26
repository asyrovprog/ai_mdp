import mdp
import random
import time
import mdp_iter as imdp
import mdp_utils
import mdp_aprox as mc
import mdp_gauss as mg

#
# Simple game: at  initial  state  (s0)  we  have  2  actions  'stay' and 'quit', if we
# 'quit' we  get  $10  and  game  ends  (s1), if we 'stay' than with probability 2/3 we go to
# state s0 with reward $4, or with probability 1/3 we go to end state (s1) with also reward $4.
# For this game discount is default 1.0.
#
# Expectation for <stay> policy is $12 from E = 2/3($4 + E) + 1/3($4)
# Expectation for <quit> policy is obviously $10
# Expectation for random policy is $10.5 which is found from system of equations:
#   E  = 1/2($10) + 1/2(E')
#   E' = 2/3($4 + E) + 1/3($4)
#
class stay_quit_t(mdp.mdp_t):
    def actions(self, s):
        # we return same actions for all states, since s1 is end state and we have just 2 states
        return ['stay', 'quit']
    def start_state(self):
        return 'in'
    def is_end(self, s):
        return s == 'end'
    def transitions(self, s, a):
        if s == 'in' :
            return [(2.0/3.0, 4.0, 'in'), (1.0 - 2.0/3.0, 4.0, 'end')] if a == 'stay' else [(1.0, 10.0, 'end')]
        return [] # for s1 there are no transitions
    def size(self):
        return 2
    def get_state(self, id):
        return 'in' if id == 0 else 'end'

class stay_policy_t(mdp.policy_t):
    def action(self, s):
        return 'stay'

class quit_policy_t(mdp.policy_t):
    def action(self, s):
        return 'quit'

if __name__ == '__main__':
    random.seed(time.time())
    m = stay_quit_t()
    mdp_utils.run_with_policy(m, mdp.random_policy_t(m), 'random policy (should be close to 10.5)', 10000)
    mdp_utils.run_with_policy(m, stay_policy_t(m), '<stay> policy (should be close to 12)', 10000)
    mdp_utils.run_with_policy(m, quit_policy_t(m), '<quit> policy (must be exactly 10)', 10000)

    # use policy evaluation to estimate utility of each policy
    print("\n<stay> policy evaluation (must be very close to 12):")
    print(imdp.policy_evaluation(m, stay_policy_t(m)))
    print("\n<quit> policy evaluation (must be very close to 10):")
    print(imdp.policy_evaluation(m, quit_policy_t(m)))
    print("\nrandom policy evaluation (must be very close to 10.5):")
    mdp_utils.run_with_policy(m, mdp.random_policy_t(m), '', 1000)

    # calculate optimal policy
    print("\noptimal policy for each state:")
    print(imdp.value_iterator(m))

    # calculate optimal policy
    print("\napproximate transitions/rewards and then do optimal policy evaluation")
    ea = mdp_utils.run_with_policy(m, mdp.random_policy_t(m), '', 10000)
    me = mc.approximate_model(ea, 'in', 1.0)
    pme = imdp.value_iterator(me)
    print("\ncalculated policy:")
    print(pme)
    pe = mdp.dict_policy_t(me, pme)
    print("\npolicy evaluation:")
    print(imdp.policy_evaluation(me, pe))

    print("\nPolicy evaluation using Gaussian elimination")
    print("Optimal policy (stay) :")
    print(mg.policy_evaluation(m, stay_policy_t(m)))
    print("Suboptimal policy (quit) :")
    print(mg.policy_evaluation(m, quit_policy_t(m)))
