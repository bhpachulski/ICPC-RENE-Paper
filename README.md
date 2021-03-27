# What is the Vocabulary of Flaky Tests? An Extended Replication

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Bruno Henrique Pachulski Camara <sup>1</sup>, <sup>2</sup>, <br />
Marco Aure ́lio Graciotto Silva <sup>3</sup>, <br />
Andre T. Endo <sup>4</sup>, <br />
Silvia Regina Vergilio <sup>2</sup>. <br />

<sup>1</sup> Centro Universitário Integrado, Campo Mourão, PR, Brazil <br />
<sup>2</sup> Department of Computer Science, Federal University of Parana ́, Curitiba, PR, Brazil <br />
&nbsp; &nbsp; &nbsp; bhpachulski@ufpr.br, silvia@inf.ufpr.br <br />
<sup>3</sup> Department of Computing, Federal University of Technology - Parana ́, Campo Mourão, PR, Brazil <br />
&nbsp; &nbsp; &nbsp; magsilva@utfpr.edu.br <br />
<sup>4</sup> Department of Computing, Federal University of Technology - Parana ́, Cornélio Procópio, PR, Brazil <br />
&nbsp; &nbsp; &nbsp; andreendo@utfpr.edu.br <br />

This paper has been submitted for publication in *ICPC 2021 - Replications and Negative Results (RENE)*.

This experimental package is organized by research questions. For each of the questions, some files can be executed to obtain the data that are presented in the paper.

## Abstract

> Software systems have been continuously evolved and delivered with high quality due to the widespread adoption of automated tests. A recurring issue hurting this scenario is the presence of flaky tests, a test case that may pass or fail non-deterministically. A promising, but yet lacking more empirical evidence, approach is to collect static data of automated tests and use them to predict their flakiness. In this paper, we conducted an empirical study to assess the use of code identifiers to predict test flakiness. To do so, we first replicate most parts of the previous study of Pinto~et~al.~(MSR~2020). This replication was extended by using a different ML Python platform (Scikit-learn) and adding different learning algorithms in the analyses. Then, we validated the performance of trained models using  datasets with other flaky tests and from different projects.  We successfully replicated the results of Pinto~et~al.~(2020), with minor differences using Scikit-learn; different algorithms had performance similar to the ones used previously. Concerning the validation, we noticed that the recall of the trained models was smaller, and classifiers presented a varying range of decreases. This was observed in both intra-project and inter-projects test flakiness prediction. 

Keywords: test flakiness, regression testing, replication studies, machine learning

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg