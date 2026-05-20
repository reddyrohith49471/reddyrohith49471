<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Mono&weight=600&size=16&pause=1400&color=FFFFFF&background=00000000&center=true&vCenter=true&width=720&lines=Reddy+Rohith+Kosinepalli;ML+Researcher+%7C+Engineer+%7C+Scientist;%E2%88%82(self)%2F%E2%88%82t+%3D+curiosity+%C2%B7+%E2%88%87(knowledge)" alt="typing" />

</div>

<br/>

---

> *Some people write about themselves.*
> *I'd rather let the mathematics do it.*

---

<br/>

## The Beginning

I didn't start with machine learning. I started with a question —
**why do systems fail when the data gets harder?**

That question led me to **IIIT Agartala** (B.Tech CSE, 2027), then to a research internship at **IIIT Allahabad**, and eventually to a stage at the **Springer MICA 2025 International Conference** where I presented my first research paper.

Along the way, I became a **Hackathon Finalist** at **IIT Kharagpur** and the **University of Hyderabad**.

But more than credentials — I became someone who reads loss curves the way others read stories.
Every dip is a chapter. Every plateau is a problem worth solving.

<br/>

---

## The Research

At IIIT Allahabad, I worked on a problem that looked simple on paper but wasn't:
**predicting heart disease from clinical data where the positive class barely existed.**

Most models would learn to ignore it entirely.
High accuracy. Zero usefulness.
That's the quiet deception of imbalanced learning —

$$\mathbb{E}\\left[\left(\hat{f}(x)-f(x)\right)^2\right] \=\ \underbrace{\text{Bias}^2[\hat{f}]}_{\text{ignoring the minority}} \+\ \underbrace{\text{Var}[\hat{f}]}_{\text{noise sensitivity}} \+\ \sigma_\varepsilon^2$$

I refused to accept that tradeoff.

The result was a paper — **"Stacking Inside Voting: A Hybrid Ensemble for Heart Disease Prediction"** — published in **Springer Proceedings at MICA 2025**.
A hybrid architecture that turned a broken $F_1$ score into something meaningful:

$$F_1 \=\ \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \:\quad 0.05 \\longrightarrow\ 0.93$$

<br/>

---

## How I Think

I reason like Bayes — I don't arrive at conclusions, I update toward them.
Every new piece of data shifts the posterior:

$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \cdot P(\theta)}{P(\mathcal{D})}$$

I believe most models fail not because they can't fit — but because they fit too well.
Memorisation dressed as intelligence.
Regularisation is the discipline that separates the two:

$$\mathcal{L}_{\text{reg}} \=\ \mathcal{L}_{\text{data}} \+\ \lambda_1\left\|\theta\right\|_1 \+\ \lambda_2\left\|\theta\right\|_2^2$$

And I pay attention to what a model *doesn't* know as much as what it does.
Uncertainty isn't a weakness — it's information:

$$H(X) \=\ -\sum_{x\\in\\mathcal{X}} p(x)\log_2 p(x)$$

A confident wrong answer is worse than an honest uncertain one.

<br/>

---

## The Stack

I work across the full depth of the ML pipeline —
from raw tensors to deployed APIs, from research notebooks to production containers.

$$\mathcal{S} \=\ \mathcal{S}_{\text{foundations}} \\oplus\ \mathcal{S}_{\text{deep learning}} \\oplus\ \mathcal{S}_{\text{LLM}} \\oplus\ \mathcal{S}_{\text{MLOps}}$$

| Layer | Tools |
|:---|:---|
| **Languages** | Python · C · C++ · SQL |
| **Deep Learning** | PyTorch · TensorFlow · HuggingFace · CNNs · RNNs · Transformers |
| **LLM Fine-Tuning** | LoRA · QLoRA · PEFT · Unsloth · BitsAndBytes · TRL SFTTrainer |
| **Classical ML** | Scikit-learn · XGBoost · Ensemble Methods · SMOTE · Feature Engineering |
| **MLOps** | MLflow · Docker · GitHub Actions · AWS EC2/ECR · DagsHub · CI/CD |
| **Serving** | FastAPI · Flask · Streamlit · REST APIs |
| **Data** | NumPy · Pandas · Matplotlib · Seaborn · MongoDB · SQL |

<br/>

---

## The Next Chapter

There is one problem in machine learning I keep returning to —
the one that sits at the boundary between memory and adaptation.

**Catastrophic forgetting.**

When a model learns something new, it quietly destroys what it already knew.
Train on $\mathcal{T}_2$ and $\mathcal{T}_1$ is gone. That's not intelligence. That's replacement.

$$\mathcal{L}_{\text{forgetting}} \=\ \sum_{t=1}^{T}\\sum_{k < t}\ \mathbb{E}_{(x,y)\sim\mathcal{T}_k}\\left[\\ell\\left(f_{\theta_t}(x)\,y\right)\right]$$

This should go to zero. It rarely does.

I'm working on **Dynamic Self-Growing Continual Learning (DSGCL)** —
a graph-based architecture where the model doesn't replace old knowledge.
It grows around it.

New tasks add new nodes. Edge gates $\alpha_{m,k}$ decide exactly how much
the past informs the present. Reuse scores $\beta_m$ decide what persists.

$$\mathcal{L}_{\text{DSGCL}} \=\ \mathcal{L}_{\text{task}} \+\ \lambda_1 \underbrace{\left\|\alpha\right\|_1}_{\text{gate sparsity}} \+\ \lambda_2 \underbrace{\sum_{m}(1-\beta_m)\cdot\mathbf{1}[\beta_m < \phi]}_{\text{reuse penalty}}$$

Plasticity and stability. Not a tradeoff. A single objective.

The belief I build everything on:

$$\frac{d\ \text{knowledge}}{d\ \text{time}} \>\ 0 \quad \forall\ t$$

<br/>

---

## Let's Connect

I'm actively looking for **ML Engineer**, **Data Scientist**, and **Applied Scientist** roles
where research depth meets production reality.

<div align="center">

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-000000?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/reddy-rohith-kosinepalli-b06613218/)
[![Gmail](https://img.shields.io/badge/Gmail-000000?style=flat-square&logo=gmail&logoColor=white)](mailto:reddyrohith20061902@gmail.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-000000?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/reddyrohith49471)

<br/><br/>

```
"The model is only as good as its loss function. Choose yours carefully."
```

</div>
