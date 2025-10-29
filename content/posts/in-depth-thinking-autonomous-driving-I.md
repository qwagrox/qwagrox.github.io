---
title: "A note of thinking on end-to-end autonomous driving"
date: 2025-10-26T22:10:26+08:00
math: true
draft: false
---

### Introduction: A Debate About the Soul of Autonomous Driving

Recently, Tesla's AI lead Ashok Elluswamy explained why "End-to-End" autonomous driving is necessary:

> Why end-to-end?
> 
> Even though Tesla strongly believes in end-to-end neural networks, it is by no means the consensus approach to self-driving. Most other
>
>entities developing self-driving have a sensor-heavy, modular approach to driving. While such systems may be easier to develop and debug in 
>
>the beginning, there are several complexities with such a system. The end-to-end approach offers several benefits over that baseline. To 
>
>name a few:
>
>Codifying human values is incredibly difficult. It is easier to learn them from data.
>
>Interface between perception, prediction and planning is ill-defined. In end-to-end, the gradients flow all the way from controls to sensor
> 
>inputs, thus optimizing the entire network holistically.
>
>Easily scalable to handle the fat and long tail of real-world robotics.
>
>Homogenous compute with deterministic latency.
>
>Overall, on the correct side of scaling w.r.t. the bitter lesson.
>
>Here are a few examples that illustrate this.
>
>Example 1:
>
>In the below example, the AI needs to decide between going over the large puddle or entering the oncoming lane. Typically, entering >
>
>oncoming lanes can be very bad and potentially dangerous. However, in this situation there's enough visibility to know that there wouldn't >
>
>be an oncoming car in the foreseeable future. Secondly, that the puddle is rather large and is better avoided. Such trade-offs cannot be 
>
>easily written down in traditional programming logic, whereas it's rather straightforward for a human looking at the scene. 

While impressed by the power of Tesla's FSD, I have my own different perspective on the "End-to-End" approach. Specifically, I believe that human driving decisions are not simply the result of "big data training," and that the end-to-end method is essentially about finding a continuous function amenable to gradient optimization for complex systems, rather than being the only or inevitable path to achieving general artificial intelligence in driving.

I will attempt to articulate my viewpoint using my limited knowledge and thinking, conducting a deep reflection on the end-to-end and modular technical approaches from multiple dimensions including cognitive science, machine learning theory, and engineering practice.

### The Gap Between Human Experience and Machine Training: Perspectives on Cognition and Causality

#### The Dual-Process Model of Human Driving Decisions: Intuition Is Not Simply "End-to-End"

Human decision-making processes are not a single computational mode. Nobel laureate in Economics Daniel Kahneman's **Dual Process Theory** provides us with a powerful explanatory framework. This theory posits that human thinking comprises two distinctly different systems [1]:

*   **System 1**: Fast, automatic, intuitive, and effortless. It relies on experience, emotion, and memory association, handling most of our daily decisions, such as the operations of skilled drivers under normal road conditions.
*   **System 2**: Slow, deliberate, requiring focused attention for logical reasoning. When encountering new problems, complex situations, or high-risk decisions, System 2 is activated to conduct conscious analysis and weighing of options.

The fact that human driving often appears to be "making decisions instantly after accumulating experience" is precisely the manifestation of System 1's efficient operation. For an experienced driver, driving skills have become highly "internalized" or "automated," requiring no constant complex logical analysis. However, this by no means implies that this is a simple "input-output" mapping. When facing the "puddle" case mentioned by Ashok Elluswamy—a complex scenario requiring a trade-off between "going over a large puddle" and "entering the oncoming lane"—this is precisely a typical moment when System 2 intervenes. The driver conducts a series of rapid, conscious risk assessments:

> "There's enough visibility to know that there wouldn't be an oncoming car in the foreseeable future." — This is a **prediction** of future states and an assessment of **safety boundaries**.
> "The puddle is rather large and is better avoided." — This is based on an understanding of the **abstract concept** of "puddle" and **causal knowledge** of its potential risks (such as vehicle loss of control or damage).

Human "intuitive" decisions are actually the result of System 1 and System 2 working collaboratively based on long-term experience. They are built upon a deep understanding of **abstract, causal models** of the world, rather than purely statistical correlations in data. Academician Zheng Nanning of the Chinese Academy of Engineering has also pointed out that human intuitive responses seek global optimal solutions, with **cost functions** and **memory-based decision structures** behind them [2].

#### "Causal Confusion" in End-to-End Models: Correlation Does Not Equal Causation

In stark contrast to the complex cognitive architecture of humans, current end-to-end autonomous driving models, especially those based on Imitation Learning, fundamentally learn **statistical correlations** between input data (sensor signals) and output data (expert driving behavior). This leads to a fatal flaw: **"Causal Confusion"** [3, 4].

A classic example is: the model observes that in countless training instances, when the brake lights of the vehicle ahead light up, the expert driver applies the brakes. The model thus learns the rule "brake when seeing the front vehicle's brake lights." However, in many cases, the **root cause** of braking is actually the traffic light further ahead turning red. The front vehicle's braking is merely an **intermediate correlated variable**, not the **root cause**. If in a new scenario, the front vehicle brakes for other reasons (such as driver error) while traffic conditions do not require deceleration, a purely correlation-dependent end-to-end model might make an unnecessary emergency brake, creating danger.

Returning to the "puddle" example, an end-to-end model might learn from data: when a large area of dark reflective region appears in the sensor image, the vehicle executes a steering operation. The model does not "understand" what a puddle is, nor does it understand the physical laws by which waterlogged road surfaces might cause vehicle loss of control. It merely establishes a fragile statistical link between pixels and behavior. If it encounters an object with similar appearance but completely different nature (such as a huge black plastic sheet), the model might make the same but completely wrong evasive maneuver.

The following table clearly contrasts the essential differences between human decision-making and current end-to-end models:

| Feature Dimension | Human Driving Decisions (Based on Dual Process Theory) | End-to-End Autonomous Driving Models | 
| :--- | :--- | :--- |
| **Learning Paradigm** | Based on understanding and constructing **causal models** of the world | Based on fitting **statistical correlations** in large-scale data |
| **Knowledge Representation** | Abstract, symbolic concepts (such as "puddle," "danger") | High-dimensional, distributed vector representations ("black box") |
| **Decision Mechanism** | **System 1** (intuition) and **System 2** (reasoning) working in synergy | Single, unified neural network forward propagation |
| **Generalization Ability** | Can handle entirely new scenarios through logical reasoning and imagination | Depends on training data coverage, poor generalization to out-of-distribution scenarios |
| **Core Issues** | Cognitive load, fatigue, irrational emotional interference | **Causal confusion**, poor interpretability, difficult safety verification |

Therefore, the end-to-end approach is essentially about constructing a continuous function in hopes of better gradient optimization. It is a powerful mathematical tool, but on the path toward general artificial intelligence, it sidesteps the core challenge of constructing causal models of the world. As research points out, even end-to-end methods need to "identify policy-relevant context and discard irrelevant details" [5], which itself suggests that the purely end-to-end paradigm may need to evolve toward a more structured direction.


### Engineering Trade-offs: The Dilemma Between Scalability and Verifiability

In Ashok Elluswamy's arguments, two points are extremely persuasive and reflect engineering reality: **(1) Easy to scale to handle long-tail problems; (2) Homogeneous compute with deterministic latency.** These are indeed the core engineering advantages of the end-to-end approach. However, behind these advantages lies the sacrifice of the system's **interpretability, verifiability, and debuggability**, which are precisely the cornerstones of Safety-Critical Systems.

#### Advantages of the Modular Approach: Transparent, Controllable, Verifiable

The traditional modular approach decomposes the autonomous driving task into independent modules such as perception, prediction, planning, and control. The advantages of this architecture include:

*   **Specialization and Decoupling**: Each module can be developed and optimized by specialized teams using the most suitable technology for that task. For example, the perception module can use convolutional neural networks, while the planning module can use search-based or optimization-based algorithms.
*   **Interpretability and Debugging**: When the system encounters problems, it is relatively easy to pinpoint which module's output does not meet expectations. Engineers can inspect the intermediate outputs of each module (such as detection boxes, predicted trajectories, planned paths), which is crucial for debugging and attribution.
*   **Verifiability**: Clear input-output interfaces and performance indicators (KPIs) can be defined for each module, and independent unit testing, integration testing, and regression testing can be conducted. This makes safety verification of the system possible.

However, its drawbacks, as Ashok pointed out, include difficult interface definition between modules and the tendency for "error accumulation" effects—small errors in upstream modules may be amplified in downstream modules, leading to final decision failures. Optimizing the entire system also becomes very difficult.

#### The "Devil's Bargain" of the End-to-End Approach: Trading Verifiability for Scalability

The end-to-end approach "fuses" all modules together through a massive neural network, achieving direct mapping from sensors to control. This brings significant engineering advantages:

*   **Simplified Architecture**: Reduces complex inter-module interfaces and manual rules, making the system more computationally homogeneous, facilitating hardware acceleration and achieving deterministic latency.
*   **Data-Driven Scaling**: Facing "long-tail problems" (i.e., countless rare but potentially fatal driving scenarios in the real world), theoretically performance can be improved by continuously "feeding" the model new data without modifying complex code logic. This gives the system strong **scalability**.

But this is essentially a "devil's bargain." The cost is that the system becomes an almost incomprehensible "black box." When an end-to-end system fails in a certain scenario, engineers find it difficult to answer the question "why." Is it due to minor perturbations in sensor input? Abnormal activation values in a certain network layer? Or some bias in the training data? This **lack of interpretability** makes debugging extremely difficult and renders formal **safety verification** nearly impossible.

Research clearly indicates that both modular and end-to-end methods face common challenges of generalization, interpretability, and robustness [6]. Blindly believing in either approach cannot solve fundamental problems. The following table summarizes the core trade-offs between the two approaches in engineering practice:

| Engineering Dimension | Modular Approach | End-to-End Approach |
| :--- | :--- | :--- |
| **System Architecture** | Heterogeneous, complex, multi-module | Homogeneous, simple, single network |
| **Development and Debugging** | Modules developed independently, easy to locate and debug | Trained holistically, difficult to debug, like "alchemy" |
| **Interpretability** | Strong, intermediate results clearly visible | Weak, decision process is a "black box" |
| **Safety Verification** | Relatively feasible, independent verification of each module possible | Extremely difficult, hard to conduct formal verification |
| **Handling Long-tail Problems** | Relies on manual rules and scenario libraries, poor scalability | Relies on data-driven approach, theoretically strong scalability |
| **System Optimization** | Local optimization, difficult to achieve global optimum | Global joint optimization, theoretically higher performance ceiling |

Therefore, Ashok's statement that the end-to-end approach is "on the correct side of scaling" only tells half the story (in my personal opinion). It may have advantages in addressing the "breadth" of long-tail distributions, but in ensuring the "depth" (i.e., reliability, verifiability) of safety-critical systems, it brings enormous, possibly insurmountable challenges.

### Conclusion: Beyond the Route Debate, Returning to the Essence of Intelligence

Through the above analysis, we can roughly conclude: there is a huge gap between the current mainstream deep learning paradigm and achieving truly general artificial intelligence. Equating the complex decision-making process of humans based on experience, causality, and abstract models with a data-fitting engine designed to optimize continuous functions is an oversimplified and misleading analogy.

**Summary:**

1.  **Human Intelligence Is Not "End-to-End"**: Human driving decisions are complex cognitive activities based on "dual process theory," relying on abstract and causal models we construct of the world, rather than simple input-output mappings. Equating human intuition with the black-box computation of neural networks ignores the structured knowledge and reasoning processes behind it.

2.  **End-to-End Learning Has Fundamental Flaws**: "Causal confusion" is a fatal weakness of current imitation learning-based end-to-end methods. Models learn fragile statistical correlations rather than robust causal relationships, making them inherently unreliable when facing "long-tail" scenarios outside the training data distribution.

3.  **Engineering Choices Are a "Devil's Bargain"**: The end-to-end approach sacrifices the system's interpretability, debuggability, and verifiability in exchange for architectural simplicity and data-driven scalability. For safety-critical systems like autonomous driving, the risks of this bargain are enormous and may even be unacceptable.

So, where does the future of autonomous driving lie? We should not fall into the binary opposition of "modular" versus "end-to-end." The real way forward may lie in **hybrid architectures that combine the advantages of both**. Future systems may exhibit the following characteristics:

*   **Macro-level Modularity with Causality as the Backbone**: The system maintains a clear modular structure at the top level (such as perception, world modeling, value judgment, planning), ensuring the system's interpretability and verification framework.
*   **Micro-level End-to-End Within Modules**: For certain well-defined subtasks (such as extracting vehicle pose from LiDAR point clouds and images), the powerful feature extraction capabilities of end-to-end learning can be utilized, but the output is structured, symbolic information that can be understood by downstream modules, rather than direct control signals.
*   **Introducing World Models and Symbolic Reasoning**: The system needs an explicit "world model" capable of modeling and reasoning about entities, relationships, and physical laws in the environment. This may be the key bridge connecting perception and decision-making, and the fundamental approach to solving the causal confusion problem.

Ultimately, the ultimate solution for autonomous driving may not be finding a more powerful "alchemy furnace" to fit an all-encompassing function, but rather returning to the essence of intelligence and constructing a system that truly understands the world and makes decisions based on causal relationships. This requires going beyond the current deep learning paradigm and conducting more arduous and in-depth exploration at the intersection of neuroscience, cognitive science, and computer science.

### References

[1] Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

[2] Zheng, N. (2018). *Intuitive AI and Autonomous Driving*. OFweek.

[3] Chen, L., Wu, P., Chitta, K., Jaeger, B., Geiger, A., & Li, H. (2024). End-to-end Autonomous Driving: Challenges and Frontiers. *arXiv preprint arXiv:2306.16927v2*.

[4] Li, J., Li, H., Liu, J., Zou, Z., Ye, X., Wang, F., & Huang, J. (2024). Exploring the Causality of End-to-End Autonomous Driving. *arXiv preprint arXiv:2407.06546v1*.

[5] Leong, E. (2022). *Bridging the Gap Between Modular and End-to-end Autonomous Driving Systems*. EECS Department, University of California, Berkeley, Tech. Rep. UCB/EECS-2022-79.

[6] Attrah, S. (2025). Autonomous driving: Modular pipeline Vs. End-to-end and LLMs. *Medium*.


