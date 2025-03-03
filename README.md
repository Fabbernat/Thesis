# Word in Context (WiC) Task
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

# Example usages:
![Képernyőkép 2025-02-06 124831](https://github.com/user-attachments/assets/1c0691c6-2bb7-4cdf-a1be-06793b9c09b3)

![Képernyőkép 2025-02-06 124751](https://github.com/user-attachments/assets/baa43460-712b-48f2-a1ed-cf8a6c694c62)

![Képernyőkép 2025-02-06 124646](https://github.com/user-attachments/assets/93443df3-46f4-48fd-91bd-003a85000fa2)

![Képernyőkép 2025-02-06 124555](https://github.com/user-attachments/assets/d5c03a1d-81f3-4b7e-a5f8-53c50e17bdc5)

![Képernyőkép 2025-02-06 124250](https://github.com/user-attachments/assets/5ed3bec3-f99c-413a-8e0a-fb576512cdfa)
