# bali-subrahmanyam-wen-2021-replication
Code to replicate the Bali, Subrahmanyam and Wen (2021) JFQA paper.

This code repository is written in Python and contains 3 script files which replicate the results in the paper entitled
"The Macroeconomic Uncertainty Premium in the Corporate Bond Market" by Bali, Subrahmanyam and Wen (2021), henceforth BSW.

WRDS Bond Returns Corporate Bond Database Replication Script
The file entitled "wrds_bali_subrahmanyam_wen_2021_replication.py" replicates their paper with the publicly available 
bond dataset from thw WRDS Bond Returns Module and requires an academic (or otherwise) subscription to WRDS.
The script directly queries the WRDS Bond Returns database and downloads bond returns, amount outstanding, maturity,
bond prices and ratings that are required for the replication.

Once the data has been downloaded, the BBW 4-Factors are downloaded (these are the original factors fom Turan Bali's website
-- https://sites.google.com/a/georgetown.edu/turan-bali/data-working-papers?authuser=0 ).
Note that the DRF and CRF factors have a lead error for most of the sample and the LRF factor has a lag error from 2014-2016.
Besides this, the factors cannot be replicated using any corporate bond database.
The BBW original MKTbond factor also exactly mimics the ICE Investment Grade Index. The original MKTbond factor is used in
an attempt to be as close as possible to the original paper.

The script then estimates a rolling tim-series regression for each bond in the sample using 36-Months of data,
requiring a bond to have a minimum of 24-Months of data for a beta to be computed. This means betas begin at the end of 
Month 24 in the sample.

Results are then replicated using a univariate portfolio sort and Fama-MacBeth regressions.
