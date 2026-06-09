# Open-Ended Interestingness-Based Learning

An LLM (Qwen/Qwen2.5-0.5B-Instruct) is prompted to generate question-answer pairs. The same LLM is also prompted to rank the interestingness of each token within its context. GRPO reinforcement learning is applied on a per token basis to maximize interestingness. See intrinsic_rl.py and training_log.txt for details.

# ~~Differentiable Creativity~~ (Negative Results)

~~A language model is trained on its own sharpened distribution thereby providing the test-time synaptic feedback hypothesized to be necessary for creativity.~~

# Motivation & Creativity

Intrinsically-motivated behavior is behavior done for it's own sake. Extrinsically-motivated behavior is behavior done for some externally supplied reward (https://people.cs.umass.edu/~barto/IMCleVer-chapter-totypeset2.pdf).

In short, intrinsic motivation usually generates internal rewards that facilitate learning or performance. For example, the intrinsic motive known as "curiosity" has the agent try to predict the next frame or state and then supply an internal reward proportional to the error of this prediction. In this way the agent is encouraged to explore that which is unfamiliar or difficult to predict.

In this paper (https://arxiv.org/abs/1705.05363) they use pure curiosity (with no external rewards) to solve the first level of Super Mario Bros. 

As an aside, exploration is important in reinforcement learning. For example, epsilon-greedy methods will take the reward-maximizing action with P() = 1 - epsilon and take a random action with P() = epsilon. Such random actions are necessary for some algorithms to work well in practice (https://github.com/seungeunrho/minimalRL/blob/master/dqn.py) as well as to converge to the optimal policy in theory (http://incompleteideas.net/book/RLbook2020.pdf).

Take for example the game of Tetris with only line clears being externally rewarded. As an agent, how long must you perform what are essentially random actions before you encounter a line clear? And how long will it take you to generalize these specific line clears that you encounter to the concept of line clearing in general?

Empowerment is another intrinsic motive (https://arxiv.org/abs/1310.1863). It can be viewed as maximizing the entropy you control while minimizing the entropy you don't control.

Creativity has been studied at the computational level (https://computationalcreativity.net/iccc21/wp-content/uploads/2021/09/ICCC_2021_paper_14.pdf) but many questions remain.

For example, what is a good intrinsic motive for creativity?

# Qualia = Intermediate Representations

We conjecture that qualia are the intermediate representations of a neural system.

Does this mean a vector of numbers is conscious and has subjective experience? No, beauty is in the eye of the beholder and that vector of numbers must be processed or perceived by an actual system in order to be a qualia percept.

The study of intermediate representations and how to arrive at effective ones is the subject of representation learning (https://arxiv.org/abs/1206.5538).

What does this conjecture mean for consciousness? That for a fixed neural system you can mess with its qualia by messing with it’s neuronal activations. This is unsurprising really. Moreover, as a neural system evolves and adapts so too do its qualia. Also unsurprising in retrospect.

What does this mean for the hard problem of consciousness? Consciousness is the processing of intermediate representations by a neural system. It “feels like something” to be a particular neural system because different neural systems process or interpret representations differently.
