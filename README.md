# Open-Ended Interestingness-Based Learning

An LLM (Qwen/Qwen2.5-0.5B-Instruct) is prompted to generate question-answer pairs. The same LLM is also prompted to rank the interestingness of each token within its context. GRPO-based reinforcement learning is applied to maximize interestingness. See intrinsic_rl.py and training_log.txt for details.

## Sample Output

[ALIGNED] **Research(1.34) Question:**(-1.50) How(-1.25) can(2.94) we(3.25) optimize(3.44) the(3.50) use(3.56) of(3.62) renewable(3.31) energy(1.94) sources(2.31) in(2.44) urban(3.25) areas(0.06) to(2.56) improve(-1.41) air(-2.06) quality,(-3.11) reduce(-1.09) greenhouse(-0.16) gas(-1.72) emissions,(0.28) and(0.50) enhance(0.56) public(2.00) health(-1.91) outcomes?(-0.88) **Answer:**(-3.02) One(-2.19) fascinating(-1.09) approach(1.34) to(0.50) optimizing(1.00) the(-1.00) use(-3.50) of(-0.16) renewable(1.41) energy(2.06) sources(1.22) in(2.75) urban(1.38) areas(1.62) is(2.00) through(-0.97) the(1.16) development(-0.75) of(3.25) advanced(-1.69) smart(1.69) grid(0.00) systems(2.31) that(0.28) not(1.25) only(2.94) manage(1.34) the(0.53) distribution(1.41) of(3.00) electricity(2.31) but(3.25) also(2.06) efficiently(3.31) integrate(0.69) diverse(1.12) renewable(0.78) energy(2.19) sources(1.50) like(1.09) solar,(1.16) wind,(-2.88) and(-0.59) geothermal(0.41) power(2.12) into(0.56) the(-4.06) grid.(-2.39) These(-1.50) systems(-1.69) could(-1.53) be(2.12) equipped(1.66) with(1.34) real-time(0.09) monitoring(1.25) devices(-0.94) to(2.94) predict(3.31) and(3.50) respond(1.25) quickly(-0.16) to(-1.09) changes(1.94) in(-1.28) demand,(0.56) ensuring(-0.38) a(0.88) seamless(3.25) integration(0.88) of(0.94) renewable(-2.36) energy(-3.14) sources.(-0.03) To(-1.16) test(-0.34) this(1.88) concept,(-0.88) a(1.44) multi-year(0.78) study(-1.53) involving(-2.06) multiple(1.44) cities(0.78) could(1.22) involve:(1.47) 1.(1.50) **Data(3.50) Collection**:(0.28) Collecting(2.75) comprehensive(-0.53) data(1.34) on(-0.78) daily(0.66) renewable(3.12) energy(2.81) production(2.50) from(3.44) solar(3.19) panels,(3.50) wind(1.31) turbines,(-3.41) and(-3.11) other(2.06) renewable(1.47) energy(3.06) sources.(2.44) 2.(2.75) **Real-Time(1.66) Monitoring**:(2.00) Implementing(2.38) sensors(2.00) that(2.31) continuously(-0.50) monitor(0.44) these(1.69) energy(1.94) sources(1.69) for(-3.02) any(-3.47) anomalies(-1.12) or(-2.89) inefficiencies.(-0.22) 3.(-1.25) **Grid(0.91) Management**:(-0.94) Using(-1.34) advanced(-1.38) algorithms(-0.19) to(-1.12) dynamically(-1.12) adjust(0.00) the(-0.06) load(2.56) management(1.22) system,(2.19) which(1.47) adjusts(2.44) the(3.06) grid(-2.81) frequency(3.50) to(2.69) balance(-0.44) out(1.22) supply(0.22) and(-0.75) demand.(-0.75) 4.(-0.50) **Public(-3.38) Engagement**:(-1.62) Inviting(3.44) local(-3.19) communities(-3.31) and(0.06) businesses(0.69) to(1.69) participate(-1.78) in(1.94) the(-1.31) development(3.06) and(3.25) testing(1.00) phases(2.31) of(2.56) the(2.56) smart(3.50) grid(-0.25) system,(0.34) offering(0.50) feedback(-0.91) on(-0.69) its(3.38) effectiveness.(3.56) By(1.41) leveraging(0.53) technology(1.94) such(2.75) as(2.31) IoT(1.81) (Internet(3.25) of(0.53) Things)(-1.91) devices(-3.78) and(0.50) predictive(0.97) analytics,(1.44) this(1.41) innovative(1.94) smart(-4.44) grid(3.25) system(2.94) would(1.00) enable(2.06) cities(1.06) to(3.25) become(2.31) more(1.06) resilient(2.75) against(2.12) fluctuations(1.69) in(-0.84) renewable(-2.69) energy(-0.19) supply(2.62) and(2.94) to(-1.22) more(-3.78) accurately(-0.72) plan(0.97) resource(-1.94) allocation.(0.91) This(2.19) approach(0.56) could(-1.81) lead(-2.25) to(2.94) significant(-0.25) improvements(2.81) in(3.50) air(-2.19) quality,(-4.28) reduced(-0.62) greenhouse(-4.00) gas(0.94) emissions,(-1.50) and(0.78) enhanced(3.25) public(-0.72) health(0.38) outcomes(1.34) by(1.69) better(-0.53) integrating(1.50) renewable(2.69) energy(2.06) sources(3.38) into(2.56) existing(0.88) infrastructure.(2.94)

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
