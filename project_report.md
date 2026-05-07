# Project Keyword spotting

Keyword spotting: identifying target words in noisy audio streams through real-time temporal pattern (correlation) detection.


- Reproduce correlation-based EI+ WTA network
- Validate filtering behavior under diﬀerentinputs
- Analyze robustness to noise and distractors
- Applyto keyword spotting task
- Evaluate accuracyand limitations
- Writereportand prepare presentation



### Guidance 

Audio: Clear Goldstandard audio stream // may from Whisper for Audio: 
Noise: Noise sampling for difficulty 



Example: 
"Can you bring some milk from the grocery store" .. like alexa it listens the hole time and when __ALEXA__ or __Hey Google__ is said it activates. Everything else is ignored.

"Can you bring some $\color{red}\boxed{milk}$ from the grocery store"

$\rarr$ we need to learn the __digital equivalent of the keyword__

### Idea:





### Tools: 

__FFT__: 

- decompose of the signal to split or pre-process 
- Linearity by Taylor approx. 

### Learning: 

- the keyword must be given in an classification task.
  - we wanna he

---

MIRCO:

1. We do a corraclation detection of a pure frequency 
2. we build different population which are listening to differnt frequencies 
   1. hpw do i make out which winns 





---

### Generating Input Spike Pattern @ Lab02 

The language of neuromorphic chips is spike trains. Whether we want to process the analog or digital signal, we need to convert it to spikes first. Inside DYNAP-SE1, we have such converters, following a single protocol: AER (Address-Event Representation). AER uses (spike source address, timing) pair to define a single spike. In this exercise, we will not use AER but the Dirac delta function $\rho(t)=\sum_{i=1}^{k} \delta\left(t-t_{i}\right)$ to define spike trains.











TODO:

- connection to dynapse 
- get stad config 
- 4 Channels of å 16 population size for $E \ \ \& \ I $ 
  - pruduce a spike pattern 4 diff for all 4 channels 
  - like the same but shifted 
- Produce a refrency sequence with same frequency $1 \over T $ 
- align pattern for channel and reference with an error term $\epsilon$ 
  - ![Bildschirmfoto 2026-04-30 um 10.32.12](../../../../../Library/Application Support/typora-user-images/Bildschirmfoto 2026-04-30 um 10.32.12.png)
- after this WTA should kick and drive 1 Population to the qinner which returns the reference channel 







- spike_id listen for given population/neurons Lin.space 
- withpout reference cant reach the 
