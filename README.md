# My Bachelor's Thesis: Analyzing the Consistency of Semantical Capabilities of Large Language Models - a Word-in-Context Benchmark Evaluation Framework and Utility Library
You can test the semantical sentence-understanding capabilities of **any\* Hugging Face model**

# [src/Framework](https://github.com/Fabbernat/Thesis/tree/master/src/Framework) - The module where it happens
[src/Framework module](https://github.com/Fabbernat/Thesis/tree/master/src/Framework)
### Input:
- Any amount of records from the [Word in Context dataset](https://pilehvar.github.io/wic/) (or records in the same format, of course 🙂)
- Any\* [Hugging Face](https://huggingface.co/models) model
### Output: 
- Detailed statistics and analytics of the model's answers to the input. Unfortunately no plots yet, but I'm working on automatic plotting the results using [TikZ](https://tikz.net/) right now.

\* almost any, qwen and google models are the most compatible. You need to make your own scripts to test unsupported models. The framework has been thoroughly tested on 
1. [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), so this and [similar models](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) will grantedly work.
2. [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) has been tested a lot too, so this and similar models will work. Note: As gemma is a [gated model](https://huggingface.co/docs/hub/en/models-gated), you'll need to log in to use it.

### How to run the app?
##### In PowerShell ( or your favorite terminal that supports `git` and `pip`)
1. clone the repository to a folder e.g.
   ```powershell
   cd ~\PycharmProjects
   git clone https://github.com/Fabbernat/Thesis
   ```
   Install required packages (may vary based on the chosen model)
   ```powershell
   cd Thesis
   pip install torch transformers accelerate huggingface_hub
   ```
   
   Then run the modules one by one:
    ```powershell
   py -3.13 -m src.Framework.ModelInputPreparer.main
    
   py -3.13 -m src.Framework.HuggingFaceModelInferencer.main

   py -3.13 -m src.Framework.ModelOutputProcessor.main
    ```
    Or run all three:
    ```powershell
   py -3.13 -m src.Framework.globalMain
    ```

##### In PyCharm
1. Clone the Repo. Python interpreter needed. It is recommended to use PyCharm
2. navigate to `src/Framework/ModelInputPreparer/main.py` and run `main()` (in PyCharm just click the green triangle)
3. You see the results in the `.out` files
4. do the same with the `HuggingFaceModelInferencer` and the `ModelOutputProcessor` modules, or just run the `src/Framework/globalMain.py` to execute all three modules at once
5. Check the results in the .out files
6. That's it!

## The paper [Overleaf Project]:
[Analyzing the Consistency of Semantical Capabilities of Large Language Models](https://www.overleaf.com/read/pfzywbczdsfb#057a56)
## Pdf TeX Source:
[GitHub Thesis-paper](https://github.com/Fabbernat/Thesis-paper)  

## My home page:
[Bernát Fábián](https://fabbernat.github.io/)


## Word in Context (WiC) Task
By design, word embeddings are unable to model the dynamic nature of words' semantics, i.e., the property of words to correspond to potentially different meanings. To address this limitation, dozens of specialized meaning representation techniques such as sense or contextualized embeddings have been proposed. However, despite the popularity of research on this topic, very few evaluation benchmarks exist that specifically focus on the dynamic semantics of words. In this paper we show that existing models have surpassed the performance ceiling of the standard evaluation dataset for the purpose, i.e., Stanford Contextual Word Similarity, and highlight its shortcomings. To address the lack of a suitable benchmark, Pilehvar and his team put forward a large-scale Word in Context dataset, called WiC, based on annotations curated by experts, for generic evaluation of context-sensitive representations. WiC is released in https://pilehvar.github.io/wic/.

This repository contains an algorithm to achieve as much accuracy as possible on the WiC
binary classification task. Each instance in WiC
has a target word w for which two contexts are
provided, each invoking a specific meaning of w.
The task is to determine whether the occurrences
of w in the two contexts share the same meaning
or not, clearly requiring an ability to identify the
word’s semantic category. The WiC task is defined
over supersenses (Pilehvar and Camacho-Collados,
2019) – the negative examples include a word used
in two different supersenses and the positive ones
include a word used in the same supersense.


#### Research results (Hungarian) [updated: March 13, 2026.]:

<img width="549" height="342" alt="image" src="https://github.com/user-attachments/assets/c5f36ff8-caf9-4736-8125-508daed081de" />

<img width="629" height="351" alt="image" src="https://github.com/user-attachments/assets/667d3019-dfe9-43ba-9e20-bc0e94574f29" />

<img width="600" height="511" alt="image" src="https://github.com/user-attachments/assets/a93b0c37-04a5-49eb-87c5-924883422659" />

<img width="442" height="266" alt="image" src="https://github.com/user-attachments/assets/45ae62f4-7776-45ca-b2d2-c4de3b426e5b" />


#### The Google Colab Notebook used for GPU-powered run [updated: January 10, 2026.]:
[REAL Phi_4_mini_instruct.ipynb inferrer](https://colab.research.google.com/drive/1PCGSFj5bzKXs9_k0Hfd-BZf5ko3WP-za?usp=sharing)



#### Usage of the scripts [outdated]:
![image](https://github.com/user-attachments/assets/ae79159c-16c2-4018-a51c-c483ade90183)
![image](https://github.com/user-attachments/assets/768bc99d-7a77-4ab5-877d-1e5578afb8f1)
![image](https://github.com/user-attachments/assets/d3221ae1-f9a1-4295-bfb5-8f6db278d777)
![image](https://github.com/user-attachments/assets/f99d52f4-4af6-4f3b-9571-0b4d8ba01170)


- The Google Colab notebook running the models can be found [at this link](https://colab.research.google.com/drive/1yA8IAd5z2oreKUXha-16Du2YrNhemNiU?usp=sharing).
- This software can be downloaded from the [github.com/Fabbernat/Thesis](https://github.com/Fabbernat/Thesis) GitHub repository.
- Testing and evaluation of language models can be viewed in the [Generative Language Models](https://docs.google.com/spreadsheets/d/1y49lg52LHVFmTom-0ibCqYqWA1pKKhiUny-Pf3KVTIg/edit?usp=sharing) spreadsheet.
