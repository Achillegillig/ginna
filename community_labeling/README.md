# Resting-State Network Atlas: Labeling Initiative

## Overview
Welcome to the community-driven initiative to refine and assess the cognitive labeling of GINNA 33 resting-state networks ! This repository contains an atlas of networks with both **objective labels** derived from meta-analytic terms and **subjective labels** based on expert interpretation and PCA analysis.

## Goals
1. Assess the proposed subjective labels.
2. Propose alternative labels with motivated justifications.
3. Vote on the most appropriate labels based on community consensus.

## How to Contribute
1. **Assess Labels**:
   - Visit the [Issues](../../../issues) tab.
   - Comment on existing issues for the network you want to assess.

2. **Propose a New Label**:
   - Visit the [Issues](../../../issues) tab.
   - Use the provided [**Label Assessment Template**]((../../../issues/new?assignees=&labels=&projects=&template=label-assessment.md&title=%5BLABELING%5D+RSNXX+-+LABEL))

3. **Vote on Labels**:
   - Use GitHub reactions (`👍`, `👎`, `🧐`) on comments to vote.
   - In addition to opening new labeling issues, feel free to comment on any existing one!

## Instructions for assigning a cognitive label
   1. **Access the [detailed results of the target RSN](/decoding_results)**. Each file contains a) a representation of the RSN, b) the terms significantly associated with the RSN (if none, corresponding p-values are reported as a table), c) results of a Principal Component analysis (two first PCs) on the terms space and d) on the RSN space (see article for more details).

   2. **Evaluate the potential cognitive process(es) that the decoded terms represent.**
      Make sure to use:
        - The whole set of significantly associated terms
        - The strength of the association between each tearm and the RSN (Pearson r)
        - The results of the PCA (terms with comparable activation profile share a high loading on a same PC).
      **Be careful not to**:
         - Deduce a cognitive process from the RSN topology (reverse inference)
         -  over-interpret the meaning of Neurosynth terms (you can inspect the studies that load the most on any meta-analytic map from the [Neurosynth Studies tab](https://www.neurosynth.org/analyses/terms/face)
           
   3. **Assign one or multiple cognitive process(es) from the Cognitive Atlas**
      Cognitive Atlas concepts sorted by category are accessible [here](https://www.cognitiveatlas.org/concepts/categories/all). Ctrl + F can be used to quickly search through the page.

   4. **Share your interpretation with us!** See [How to contribute](##-How-to-Contribute).
      

## Files and Directories (**work in progress**)
- `data/`: Contains the atlas and descriptions.
- `proposed_labels/`: Stores current labels with justifications.
- `submissions/`: Community-submitted label proposals.
- `voting_results/`: Summaries of voting outcomes.

## Label Assignment Template
To propose a new label and/or comment on existing labels, [open an issue with the label assignment template](../../../issues/new?assignees=&labels=&projects=&template=label-assessment.md&title=%5BLABELING%5D+RSNXX+-+LABEL).
