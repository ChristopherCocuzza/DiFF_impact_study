# DiFF_impact_study

Repository for performing impact studies on DiFFs and transversity PDFs.

* <ins>baseline</ins>: This folder contains the baseline replicas from [our latest paper][paper].  They are contained in the folder msr.  It also contains the input file for running the fit.

* <ins>analysis</ins>: This folder contains the code used to analyze the replicas.  This includes doing predictions, plotting data vs. theory, and plotting the distributions.

* <ins>driver.py</ins>: Running driver.py <wdir> will analyze the replicas using the codes in the analysis folder.  Doing so also requires [Diffpack][Diffpack].

* <ins>pseudo.xlsx</ins>: This Excel file contains made up pseudo data for testing.

* <ins>sim.py</ins>: This is the file that needs to be set up to fill in pseudo.xlsx with the prediction from the baseline replicas.

[paper]: https://arxiv.org/abs/2306.12998

[Diffpack]: https://github.com/QCDHUB/Diffpack/tree/version1
